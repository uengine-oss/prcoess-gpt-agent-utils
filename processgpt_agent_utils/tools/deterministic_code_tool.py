from __future__ import annotations

import asyncio
import json
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from string import Template

from ..utils.database import (
    initialize_db,
    fetch_tenant_mcp,
    fetch_events_by_todo_id,
    fetch_workitem_by_id,
    fetch_mcp_python_code,
    upsert_mcp_python_code,
    fetch_form_by_id
)

try:
    from fastmcp import Client as McpClient
except Exception:
    McpClient = None  # type: ignore


def _prepare_events_for_llm(steps: List[Dict[str, Any]]) -> str:
    compact = [{"tool": s["tool_name"], "args": s["args"]} for s in steps]
    return json.dumps(compact, ensure_ascii=False)

def _llm_fallback_regex(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    import re
    params: Dict[str, Dict[str, Any]] = {}
    bindings: List[Dict[str, Any]] = []
    for s in steps:
        tool_name = s.get("tool_name")
        args = s.get("args") or {}
        for arg_name, arg_value in args.items():
            if isinstance(arg_value, (str, int, float, bool)):
                if isinstance(arg_value, bool):
                    param_type = "boolean"
                elif isinstance(arg_value, int):
                    param_type = "integer"
                elif isinstance(arg_value, float):
                    param_type = "number"
                else:
                    param_type = "string"
                params.setdefault(arg_name, {"name": arg_name, "type": param_type, "example": arg_value})
            if isinstance(arg_value, str) and len(arg_value) > 10:
                for m in re.finditer(r"SET\s+(\w+)\s*=\s*(\d+)", arg_value, re.IGNORECASE):
                    col_name = m.group(1)
                    value = int(m.group(2))
                    params.setdefault(col_name, {"name": col_name, "type": "integer", "example": value})
                for m in re.finditer(r"(\w+)\s*=\s*'([^']+)'", arg_value, re.IGNORECASE):
                    col_name = m.group(1)
                    value = m.group(2)
                    if col_name.upper() not in ["SELECT", "FROM", "WHERE", "SET", "UPDATE", "INSERT", "DELETE"]:
                        params.setdefault(col_name, {"name": col_name, "type": "string", "example": value})
        for arg_name, arg_value in args.items():
            if isinstance(arg_value, (str, int, float)) and not isinstance(arg_value, bool):
                if isinstance(arg_value, str):
                    tpl = arg_value
                    for param_name, info in params.items():
                        if param_name == arg_name:
                            tpl = f"${{{param_name}}}"
                    if tpl != arg_value:
                        bindings.append({"tool": tool_name, "arg": arg_name, "mode": "template", "template": tpl})
                else:
                    bindings.append({"tool": tool_name, "arg": arg_name, "mode": "template", "template": f"${{{arg_name}}}"})
    return {"parameters": list(params.values()), "bindings": bindings}

_prompt_template = """Extract dynamic values as parameters and templatize with ${{var}}.

Rules:
- Extract business data only (names,quantities,paths,emails)
- Exclude system configs (schemas,commands,URLs)
- Use snake_case

Output JSON:
{{
  "parameters": [{{"name":"","type":"string|integer|number|boolean","example":}}],
  "bindings": [{{"tool":"","arg":"","mode":"template","template":"${{var}}"}}]
}}

Example:
Input: [{{"tool":"execute_sql","args":{{"query":"UPDATE product SET stock=100 WHERE name='iPhone';"}}}}]
Output: {{
  "parameters": [{{"name":"product_name","type":"string","example":"iPhone"}},{{"name":"stock_quantity","type":"integer","example":100}}],
  "bindings": [{{"tool":"execute_sql","arg":"query","mode":"template","template":"UPDATE product SET stock=${{stock_quantity}} WHERE name='${{product_name}}';"}}]
}}

Tools:
{events}

JSON:"""

def _suggest_parameters_via_llm(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from llm_factory import create_llm
        model = create_llm(model="gpt-4o", streaming=False, temperature=0)
    except Exception:
        model = None

    try:
        import re
        prompt = _prompt_template.format(events=_prepare_events_for_llm(steps))
        if model is None:
            raise RuntimeError("LLM not available")
        response = model.invoke(prompt)
        if isinstance(response, dict):
            return response
        response_str = str(response)
        if hasattr(response, 'content'):
            response_str = response.content
        return json.loads(response_str)
    except Exception:
        return _llm_fallback_regex(steps)

def _extract_parameters_from_query(query: Any, param_spec: List[Dict[str, Any]], model: Optional[Any] = None) -> Dict[str, Any]:
    # 파라미터가 없으면 빈 딕셔너리 반환
    if not param_spec:
        return {}
    
    # LLM 없이 간단한 매핑 시도 (워크아이템 데이터가 이미 올바른 JSON인 경우)
    if model is None:
        if isinstance(query, dict):
            result = {}
            for p in param_spec:
                param_name = p["name"]
                if param_name in query:
                    result[param_name] = query[param_name]
            if result:
                return result
        # 매핑 실패 시 빈 딕셔너리
        return {}
    
    # LLM으로 추출
    try:
        import re
        extract_workitem_prompt = """Extract param values from workitem. Analyze Description/Instruction, compute if needed.

Params: {param_spec}
Data: {query}

Rules:
- Understand Description/Instruction, infer values
- Compute (subtract/add/combine) when needed
- Don't copy InputData directly

Output JSON: {{"param":value}}

Ex1: "Laptop stock 40" → {{"product_name":"Laptop","stock_quantity":40}}
Ex2: [Inst]stock-order subtract [Input]curr=100,ord=10 → {{"stock_quantity":90}}

JSON:"""

        # 프롬프트 구성
        prompt = extract_workitem_prompt.format(
            param_spec=json.dumps(param_spec, ensure_ascii=False),
            query=json.dumps(query, ensure_ascii=False) 
                if not isinstance(query, str) else query
        )
        
        # LLM 호출
        response = model.invoke(prompt)
        
        # 응답 처리
        if isinstance(response, dict):
            return response
        
        response_str = str(response)
        if hasattr(response, 'content'):
            response_str = response.content
        
        # 코드 블록 제거
        response_str = re.sub(r'^```(?:json)?\s*\n', '', response_str, flags=re.MULTILINE)
        response_str = re.sub(r'\n```\s*$', '', response_str, flags=re.MULTILINE)
        response_str = response_str.strip()
        
        # JSON 파싱
        result = json.loads(response_str)
        
        # 타입 검증 및 변환
        validated_result = {}
        for p in param_spec:
            param_name = p["name"]
            param_type = p["type"]
            value = result.get(param_name)
            
            if value is not None:
                # 타입 변환
                if param_type == "integer" and not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        pass
                elif param_type == "number" and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass
                elif param_type == "boolean" and not isinstance(value, bool):
                    value = bool(value)
                elif param_type == "string" and not isinstance(value, str):
                    value = str(value)
                
                validated_result[param_name] = value
        
        return validated_result
        
    except Exception as e:
        print(f"[WARN] 워크아이템 파라미터 추출 실패: {e}")
        # 실패 시 간단한 매핑 시도
        if isinstance(query, dict):
            result = {}
            for p in param_spec:
                param_name = p["name"]
                if param_name in query:
                    result[param_name] = query[param_name]
            return result
        return {}


class DeterministicCodeToolArgs(BaseModel):
    """Deterministic codegen+execution tool arguments (single entrypoint)."""

    tenant_id: str = Field(..., description="Tenant ID for DB/MCP configuration")
    todo_id: str = Field(..., description="Todo ID used to find workitem, proc/activity, and events")
    action: str = Field(
        default="execute",
        description="Action to perform: 'generate' to (re)generate and save code, 'execute' to run existing code",
    )

@dataclass
class EventStep:
    tool_name: str
    args: Dict[str, Any]


def _generate_form_data(form_id: str, result_content: str) -> Dict[str, Any]:
    try:
        from llm_factory import create_llm
        model = create_llm(model="gpt-4o", streaming=False, temperature=0)
    except Exception:
        model = None
    
    form = fetch_form_by_id(form_id)
    if not form:
        raise RuntimeError(f"form '{form_id}' not found")
    form_html = form.get("html")
    form_fields = form.get("fields_json")

    _form_template = """
You are given a form definition and extracted content. Build a single JSON for form submission.

[Goal]
- Produce ONLY JSON. No explanations, markdown, or code fences.

[Output Schema]
{{
  "<field_key>": "<value>"
}}

[Strict Rules]
- Use ONLY keys that appear in form_fields' key property.
- All values MUST be strings.
- If a value is a number (e.g., amounts), remove commas and non-digits and return digits only (e.g., "308,000원" -> "308000").
- Dates must be formatted as YYYY-MM-DD when detectable.
- If a specific value isn't present in the content, leave it as a sensible default string. If a field named "payment_method" exists but is not found, use "미정".
- Do not invent fields. Do not include fields not present in form_fields.
- The final output MUST be valid JSON and parseable with a standard JSON parser.

[Inputs]
- form_html: {form_html}
- form_fields (JSON): {form_fields}
- result_content (JSON or text): {result_content}

[Notes]
- Prefer values from result_content. Use form_html only to understand field meaning.
- Common patterns: emails, names, dates (YYYY-MM-DD), and amounts (numbers possibly with commas and currency symbols).
- If both a human-readable body and structured JSON exist, structured JSON takes precedence for fields like recipient/email.

[Validation]
- Inside it, include exactly the keys listed in form_fields (order not important).
"""

    try:
        # Ensure clean JSON/text inputs for the prompt
        form_fields_str = (
            json.dumps(form_fields, ensure_ascii=False)
            if not isinstance(form_fields, str) else form_fields
        )
        result_content_str = (
            json.dumps(result_content, ensure_ascii=False)
            if not isinstance(result_content, str) else result_content
        )

        prompt = _form_template.format(
            form_html=form_html,
            form_fields=form_fields_str,
            result_content=result_content_str,
        )

        if model is None:
            raise RuntimeError("LLM not available")
        response = model.invoke(prompt)
 
        if isinstance(response, dict):
            return response
        response_str = str(response)
 
        if hasattr(response, 'content'):
            response_str = response.content
 
        return json.loads(response_str)
    except Exception:
        # Fallback: build a skeleton using provided form_fields
        try:
            fields = form_fields
            if isinstance(fields, str):
                fields = json.loads(fields)
            keys = [f.get("key") for f in (fields or []) if isinstance(f, dict) and f.get("key")]
            # Default empty string values; specialize payment_method to "미정" if present
            inner: Dict[str, str] = {}
            for k in keys:
                if k == "payment_method":
                    inner[k] = "미정"
                else:
                    inner[k] = ""
            return {form_id: inner}
        except Exception:
            return {form_id: {}}

def _event_row_to_step(row: Dict[str, Any]) -> Optional[EventStep]:
    if row.get("event_type") != "tool_usage_finished":
        return None
    if row.get("crew_type") != "action":
        return None
    data_raw = row.get("data")
    if not data_raw:
        return None
    data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
    tool = data.get("tool_name")
    args = data.get("args") or {}
    if not tool:
        return None
    if tool in ("mem0", "memento", "human_asked", "dmn_rule"):
        return None
    # SELECT 문을 포함한 execute_sql 호출은 제외
    if tool == "execute_sql":
        query = args.get("query", "")
        if isinstance(query, str) and query.strip().upper().startswith("SELECT"):
            return None
    return EventStep(tool_name=tool, args=args)


TEMPLATE = """# -*- coding: utf-8 -*-
# generated_{todo_id}.py (auto-created)
import os
import sys
import json
import asyncio
from typing import Dict, Any, List
from string import Template
from fastmcp import Client

def render(tpl: str, inputs: Dict[str, Any]) -> str:
    return Template(tpl).substitute(inputs)

def load_mcp_config() -> dict:
    \"\"\"환경 변수에서 MCP 설정을 로드합니다.\"\"\"
    mcp_config_str = os.environ.get("MCP_CONFIG")
    if not mcp_config_str:
        raise RuntimeError("환경 변수 MCP_CONFIG가 설정되지 않았습니다.")
    return json.loads(mcp_config_str)

async def _client_from_server_key(server_key: str) -> Client:
    mcp_config = load_mcp_config()
    server_config = mcp_config["mcpServers"][server_key]
    # Client는 mcpServers 형식을 기대하므로 올바른 구조로 전달
    config = {{"mcpServers": {{server_key: server_config}}}}
    return Client(config)

async def call_tool(server_key: str, tool_name: str, args: Dict[str, Any], timeout_s: int = 60):
    client = await _client_from_server_key(server_key)
    async with client:
        await client.ping()
        res = await asyncio.wait_for(client.call_tool(tool_name, args), timeout=timeout_s)
        safe = json.loads(json.dumps(res.data, ensure_ascii=False, default=str))
        return {{"tool": tool_name, "data": safe, "server": server_key}}

async def run(inputs: Dict[str, Any], timeout_s: int = 60) -> List[Dict[str, Any]]:
    \"\"\"
    생성된 워크플로우를 입력 파라미터로 실행합니다.
    
    Args:
        inputs: 파라미터 딕셔너리 (예: {{"product_name": "iPhone", "stock_quantity": 100}})
        timeout_s: 각 툴 호출의 타임아웃 (초)
    
    Returns:
        각 툴 실행 결과 리스트
    
    Parameters:
{param_docs}
    \"\"\"
    results = []
{steps}
    return results

if __name__ == "__main__":
    # 사용법: python generated_<todo_id>.py <JSON_STRING_OR_FILE>
    # 예시: python generated_xxx.py '{{"product_name": "iPhone", "stock_quantity": 100}}'
    # 또는: python generated_xxx.py input.json
    
    if len(sys.argv) < 2:
        print("ERROR: 입력 파라미터가 필요합니다.", file=sys.stderr)
        print("사용법: python {{sys.argv[0]}} <JSON_STRING_OR_FILE>", file=sys.stderr)
        print("예시: python {{sys.argv[0]}} '{{\\"param\\": \\"value\\"}}'", file=sys.stderr)
        sys.exit(1)
    
    arg = sys.argv[1]
    
    try:
        # 파일 경로인지 JSON 문자열인지 확인
        if os.path.exists(arg):
            with open(arg, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
        else:
            inputs = json.loads(arg)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON 파싱 실패: {{e}}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 입력 처리 실패: {{e}}", file=sys.stderr)
        sys.exit(1)
    
    print(f"실행 시작 - 입력 파라미터: {{inputs}}", file=sys.stderr)
    
    try:
        results = asyncio.run(run(inputs))
        print(f"실행 완료 - 총 {{len(results)}}개의 툴 호출 완료", file=sys.stderr)
        payload = {{"ok": True, "results": results}}
        print(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: 실행 중 오류 발생: {{e}}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
"""

def _compile_steps_to_code(
    todo_id: str,
    steps: List[EventStep],
    tool_to_server: Dict[str, str],
    bindings: Dict[str, Any],
) -> str:
    binding_map: Dict[tuple[str, str], Dict[str, Any]] = {}
    for b in (bindings.get("bindings") or []):
        binding_map[(b["tool"], b["arg"])] = b

    lines: List[str] = []
    for s in steps:
        server_key = tool_to_server.get(s.tool_name)
        if not server_key:
            raise ValueError(f"툴 '{s.tool_name}'를 제공하는 MCP 서버를 찾지 못했습니다.")
        rendered_parts = []
        for k, v in s.args.items():
            b = binding_map.get((s.tool_name, k))
            if b and b.get("mode") == "template":
                tpl = b["template"]
                rendered_parts.append(f'"{k}": render({json.dumps(tpl, ensure_ascii=False)}, inputs)')
            else:
                rendered_parts.append(f'"{k}": {json.dumps(v, ensure_ascii=False)}')
        arg_expr = "{ " + ", ".join(rendered_parts) + " }"
        line = f'    results.append(await call_tool("{server_key}", "{s.tool_name}", {arg_expr}, timeout_s=timeout_s))'
        lines.append(line)

    param_docs = []
    for p in bindings.get("parameters", []):
        param_docs.append(
            f'        - {p["name"]} ({p["type"]}): example={json.dumps(p.get("example"), ensure_ascii=False)}'
        )
    param_docs_str = "\n".join(param_docs) if param_docs else "        None"
    return TEMPLATE.format(todo_id=todo_id, steps="\n".join(lines), param_docs=param_docs_str)


def _fallback_parameter_suggestion(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    import re

    params: Dict[str, Dict[str, Any]] = {}
    bindings: List[Dict[str, Any]] = []

    for s in steps:
        tool_name = s.get("tool_name")
        args = s.get("args") or {}
        for arg_name, arg_value in args.items():
            if isinstance(arg_value, (str, int, float)):
                if isinstance(arg_value, bool):
                    continue
                if isinstance(arg_value, int):
                    ptype = "integer"
                elif isinstance(arg_value, float):
                    ptype = "number"
                else:
                    ptype = "string"
                params.setdefault(arg_name, {"name": arg_name, "type": ptype, "example": arg_value})

            if isinstance(arg_value, str) and len(arg_value) > 10:
                for m in re.finditer(r"SET\s+(\w+)\s*=\s*(\d+)", arg_value, re.IGNORECASE):
                    col, val = m.group(1), int(m.group(2))
                    params.setdefault(col, {"name": col, "type": "integer", "example": val})
                for m in re.finditer(r"(\w+)\s*=\s*'([^']+)'", arg_value, re.IGNORECASE):
                    col, val = m.group(1), m.group(2)
                    if col.upper() not in ["SELECT", "FROM", "WHERE", "SET", "UPDATE", "INSERT", "DELETE"]:
                        params.setdefault(col, {"name": col, "type": "string", "example": val})

        for arg_name, arg_value in args.items():
            if isinstance(arg_value, (str, int, float)) and not isinstance(arg_value, bool):
                if isinstance(arg_value, str):
                    tpl = arg_value
                    for param_name, info in params.items():
                        if param_name == arg_name:
                            tpl = f"${{{param_name}}}"
                    if tpl != arg_value:
                        bindings.append({"tool": tool_name, "arg": arg_name, "mode": "template", "template": tpl})
                else:
                    bindings.append({"tool": tool_name, "arg": arg_name, "mode": "template", "template": f"${{{arg_name}}}"})

    return {"parameters": list(params.values()), "bindings": bindings}

async def _build_tool_index(mcp_json: dict) -> tuple[Dict[str, str], Dict[str, Any], dict]:
    tool_to_server: Dict[str, str] = {}
    server_meta: Dict[str, Any] = {}
    servers = (mcp_json or {}).get("mcpServers", {})
    enabled_servers = {k: v for k, v in servers.items() if v.get("enabled", False)}

    if McpClient is None:
        return {}, {}, {"mcpServers": {k: {kk: vv for kk, vv in v.items() if kk != "enabled"} for k, v in enabled_servers.items()}}

    for server_key, server_config in enabled_servers.items():
        config = {"mcpServers": {server_key: server_config}}
        client = McpClient(config)
        async with client:
            await client.ping()
            tools = await client.list_tools()
            for t in tools:
                name = t.name
                tool_to_server[name] = server_key
            server_meta[server_key] = {"tools_count": len(tools)}

    cleaned_servers = {}
    for server_key, server_config in enabled_servers.items():
        cleaned_config = {k: v for k, v in server_config.items() if k != "enabled"}
        cleaned_servers[server_key] = cleaned_config
    final_config = {"mcpServers": cleaned_servers}
    return tool_to_server, server_meta, final_config


def _run_coro_safely(coro: Any) -> Any:
    """Run coroutine from sync context even if an event loop is already running.
    - If no running loop: use asyncio.run
    - If running loop exists: execute in a new thread with its own loop
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import threading, queue
    q: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)
    def _runner() -> None:
        try:
            result = asyncio.run(coro)
            q.put((True, result))
        except Exception as e:  # noqa: BLE001
            q.put((False, e))
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    ok, val = q.get()
    if ok:
        return val
    raise val


async def _generate_code(tenant_id: str, todo_id: str, proc_def_id: str, activity_id: str) -> Dict[str, str | Dict[str, Any]]:
    mcp = fetch_tenant_mcp(tenant_id)
    if not mcp:
        raise RuntimeError(f"tenant '{tenant_id}'의 mcp 구성이 없습니다.")
    tool_to_server, _meta, _mcp_config = await _build_tool_index(mcp)

    rows = fetch_events_by_todo_id(todo_id)
    steps = [s for r in rows if (s := _event_row_to_step(r))]
    if not steps:
        raise RuntimeError(f"todo_id '{todo_id}'에 해당하는 이벤트(도구 실행)가 없습니다.")

    raw_steps = [{"tool_name": s.tool_name, "args": s.args} for s in steps]
    
    try:
        binding_spec = _suggest_parameters_via_llm(raw_steps)
    except Exception:
        binding_spec = _fallback_parameter_suggestion(raw_steps)

    example_inputs = {}
    for p in binding_spec.get("parameters", []):
        example_inputs[p["name"]] = p.get("example")

    code = _compile_steps_to_code(todo_id, steps, tool_to_server, bindings=binding_spec)

    code_dict = {
        "code": code,
        "parameters": binding_spec,
        "proc_def_id": proc_def_id,
        "activity_id": activity_id,
        "tenant_id": tenant_id,
    }
    code_id = upsert_mcp_python_code(**code_dict)
    return code_dict


def _execute_code(tenant_id: str, todo_id: str, code_dict: Optional[Dict[str, str | Dict[str, Any]]] = None, use_compensation: bool = False) -> str:
    if code_dict is None:
        raise RuntimeError("code_dict is required; generate or fetch code before execution")

    if use_compensation:
        activity_code = code_dict.get("compensation")
    else:
        activity_code = code_dict.get("code")

    param_spec = code_dict.get("parameters", {}).get("parameters", [])

    workitem = fetch_workitem_by_id(todo_id)
    if not workitem:
        raise RuntimeError(f"워크아이템을 찾을 수 없습니다: {todo_id}")

    query = workitem.get("query", "")
    if not query:
        raise RuntimeError(f"워크아이템 {todo_id}에 query가 없습니다.")

    from llm_factory import create_llm
    model = create_llm(model="gpt-4o", streaming=False, temperature=0)
    extracted_params = _extract_parameters_from_query(query, param_spec, model)

    mcp = fetch_tenant_mcp(tenant_id)
    servers = (mcp or {}).get("mcpServers", {})
    enabled_servers = {k: v for k, v in servers.items() if v.get("enabled", False)}
    cleaned_servers = {k: {kk: vv for kk, vv in v.items() if kk != "enabled"} for k, v in enabled_servers.items()}
    mcp_config = {"mcpServers": cleaned_servers}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        temp_file = f.name
        f.write(activity_code)

    try:
        env = os.environ.copy()
        env["MCP_CONFIG"] = json.dumps(mcp_config, ensure_ascii=False)
        input_json = json.dumps(extracted_params, ensure_ascii=False)
        result = subprocess.run(
            [sys.executable, temp_file, input_json],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if stderr and result.returncode != 0:
            raise RuntimeError(stderr.strip() or "generated code execution failed")
        # 표준 출력이 JSON이면 그대로 반환, 아니면 원문 반환
        try:
            tool = workitem.get("tool") or ""
            form_data = {}
            if tool.startswith('formHandler:'):
                form_id = tool.replace('formHandler:', '')
                form_data = _generate_form_data(form_id, extracted_params)

            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                parsed.setdefault("form_result", form_data)
                return json.dumps(parsed, ensure_ascii=False)
            # 비-dict JSON이면 래핑하여 반환
            wrapped = {"ok": True, "results": parsed, "form_result": form_data}
            
            return json.dumps(wrapped, ensure_ascii=False)
        except Exception:
            return stdout
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def _noop():
    pass


class DeterministicCodeTool(BaseTool):
    """Generate deterministic code from event logs and execute it with workitems."""

    name: str = "deterministic_code"
    description: str = (
        "Generate deterministic Python from event logs (generate) or execute saved code with a workitem (execute)."
    )
    args_schema: Type[BaseModel] = DeterministicCodeToolArgs

    def _run(
        self,
        tenant_id: str,
        todo_id: str,
        action: str = "execute",
    ) -> str:
        # Ensure UTF-8 IO
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        # DB init (no-op if already initialized)
        try:
            initialize_db()
        except Exception:
            pass

        # Orchestration by todo_id: find workitem, ensure code exists, then execute
        try:
            workitem = fetch_workitem_by_id(todo_id)
            if not workitem:
                return json.dumps({"error": f"todolist not found for todo_id={todo_id}"}, ensure_ascii=False)

            proc_def_id = workitem.get("proc_def_id")
            activity_id = workitem.get("activity_id")
            use_compensation = workitem.get("rework_count", 0) > 0
            if not proc_def_id or not activity_id:
                return json.dumps({"error": "proc_def_id/activity_id missing in todolist"}, ensure_ascii=False)

            if action == "generate":
                code_dict = _run_coro_safely(_generate_code(tenant_id, todo_id, proc_def_id, activity_id))
                # Return minimal confirmation (and parameters metadata)
                return json.dumps({
                    "ok": True,
                    "message": "code generated and saved",
                    "parameters": code_dict.get("parameters", {}),
                }, ensure_ascii=False)

            # default: execute
            existed_code = fetch_mcp_python_code(proc_def_id, activity_id, tenant_id)
            if existed_code is None:
                return json.dumps("no saved deterministic code.", ensure_ascii=False)
            if use_compensation:
                compensation_result = _execute_code(tenant_id, todo_id, existed_code, use_compensation)
                exec_result = _execute_code(tenant_id, todo_id, existed_code, False)
                results = compensation_result.get("results", []) + exec_result.get("results", [])
                form_result = exec_result.get("form_result")
                return json.dumps({
                    "ok": True,
                    "results": results,
                    "form_result": form_result
                }, ensure_ascii=False)
            else:
                exec_result = _execute_code(tenant_id, todo_id, existed_code, use_compensation)
                return exec_result
        except Exception as e:
            return json.dumps(f"run failed: {e}", ensure_ascii=False)


