import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field, ConfigDict, model_validator
from crewai.tools import BaseTool

# A2A SDK (동일 인터페이스 가정)
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    SendMessageRequest,
    MessageSendParams,
    MessageSendConfiguration,
    Message,
    TextPart,
    Part,
    Role,
    Task,
)

logger = logging.getLogger(__name__)


# ===========================
# Models
# ===========================
class AgentEndpoint(BaseModel):
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)


class A2AAgentToolInput(BaseModel):
    """툴 입력 스키마(알 수 없는 키는 payload로 흡수)."""

    model_config = ConfigDict(extra="allow")

    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    skill: Optional[str] = None
    accepted_output_modes: List[str] = Field(default_factory=lambda: ["text"])  # e.g., ["text", "html", "image"]
    blocking: bool = True
    timeout_sec: int = 60

    @model_validator(mode="after")
    def _ensure_any_input(self) -> "A2AAgentToolInput":
        if self.message or self.payload:
            return self
        # 넘어온 기타 키들을 payload로 자동 흡수
        extra = getattr(self, "__pydantic_extra__", None) or {}
        reserved = {"message", "payload", "skill", "accepted_output_modes", "blocking", "timeout_sec"}
        inferred = {k: v for k, v in extra.items() if k not in reserved}
        if inferred:
            self.payload = inferred
            return self
        raise ValueError("Either 'message' or 'payload' must be provided.")


# ===========================
# Utils
# ===========================

def _pick_last_agent_text(task: Optional[Task]) -> str:
    if not task or not getattr(task, "history", None):
        return ""
    for m in reversed(task.history):
        if getattr(m, "role", None) == Role.agent:
            for p in getattr(m, "parts", []) or []:
                root = getattr(p, "root", None)
                if isinstance(root, TextPart) and getattr(root, "text", None):
                    return root.text
    return ""


def _compact_history(task: Optional[Task]) -> Tuple[str, List[Dict[str, Optional[str]]], Optional[str]]:
    if not task or not getattr(task, "history", None):
        return "", [], None

    history: List[Dict[str, Optional[str]]] = []
    for m in task.history:
        role_name = getattr(m.role, "value", str(m.role))
        # parts 안의 TextPart만 이어 붙임
        txt = "".join(
            getattr(p.root, "text", "")
            for p in (getattr(m, "parts", []) or [])
            if isinstance(getattr(p, "root", None), TextPart)
        ) or None
        history.append({"role": role_name, "text": txt})

    result_text = _pick_last_agent_text(task)
    state = getattr(getattr(task, "status", None), "state", None)
    task_state = getattr(state, "value", str(state)) if state else None
    return result_text, history, task_state


def _format_payload_to_message(skill: Optional[str], payload: Dict[str, Any]) -> str:
    """스킬별 포맷팅(필요 시 추가) → 기본은 JSON 문자열."""
    if (skill or "").lower() == "airbnb_search":
        loc = payload.get("location")
        checkin = payload.get("checkin")
        checkout = payload.get("checkout")
        adults = payload.get("adults")
        if loc and checkin and checkout and adults:
            return f"Please find a room in {loc}, {checkin}, checkout date is {checkout}, {adults} adults"
    return json.dumps(payload, ensure_ascii=False)


async def _fetch_agent_card(endpoint: AgentEndpoint, timeout_sec: int) -> Optional[Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout_sec, headers=endpoint.headers or None) as c:
            resolver = A2ACardResolver(httpx_client=c, base_url=endpoint.url)
            return await resolver.get_agent_card()
    except Exception as e:
        logger.debug("AgentCard 조회 실패: %s", e)
        return None


def _build_description(prefix: str, card: Optional[Any]) -> str:
    lines = [
        prefix,
        "",
        "⚠️ 중요: 아래 예시는 형식 참고용입니다. 현재 작업의 실데이터를 입력하세요.",
        "예시의 위치/날짜/인원을 그대로 쓰지 마세요.",
        "",
    ]

    if not card:
        lines.append("(참고: AgentCard 조회 실패로 스킬 정보가 표시되지 않습니다.)")
        return "\n".join(lines)

    title = f"[Agent] {getattr(card, 'name', '')}"
    if getattr(card, "version", None):
        title += f" (v{card.version})"
    lines.append(title)
    if getattr(card, "description", None):
        lines.append(f"- {card.description}")
    if getattr(card, "url", None):
        lines.append(f"- URL: {card.url}")

    lines.append("\n[사용 가능한 Skills]")
    skills = getattr(card, "skills", None) or []
    if not skills:
        lines.append("(스킬 정보 없음)")
    else:
        for idx, s in enumerate(skills, 1):
            skill_name = getattr(s, "name", "(no name)")
            lines.append(f"\n{idx}. Skill: {skill_name}")
            sd = getattr(s, "description", None)
            if sd:
                lines.append(f"   설명: {sd}")
            exs = getattr(s, "examples", None)
            if exs:
                lines.append("   사용 예시 (참고용, 반드시 문맥에 맞게 수정):")
                if isinstance(exs, str):
                    lines.append(f"      - {exs}")
                elif isinstance(exs, list):
                    for e in exs:
                        lines.append(f"      - {e}")

    lines += [
        "\n💡 사용 시 주의사항:",
        "   - message 또는 payload 형태로 입력 가능",
        "   - 허용 출력모드(accepted_output_modes)를 명시하면 멀티 모달 처리 가능",
    ]
    return "\n".join(lines)


# ===========================
# Tool
# ===========================
class A2AAgentTool(BaseTool):
    """A2A 기반으로 다른 에이전트에 메시지를 보내는 CrewAI Tool."""

    name: str = "A2A Agent Tool"
    description: str = (
        "다른 에이전트에 요청을 보내는 A2A 기반의 툴입니다.\n"
        "아래에는 연결된 에이전트의 skills/설명/예시가 자동으로 채워집니다.\n"
    )
    args_schema: type = A2AAgentToolInput

    _endpoint: AgentEndpoint

    def __init__(self, **data: Any):
        super().__init__(**data)

    @classmethod
    async def create(
        cls,
        endpoint: AgentEndpoint,
        name: Optional[str] = None,
        timeout_sec: int = 60,
    ) -> "A2AAgentTool":
        """엔드포인트만 받아 툴을 구성. 클라이언트는 매 호출 시 생성/종료.
        동기/비동기 이중 구현을 피하기 위해 비동기 본체만 유지하고,
        동기 컨텍스트에서 호출되면 내부에서 안전히 실행한다.
        """
        tool = cls()
        tool._endpoint = endpoint
        if name:
            tool.name = name

        card = await _fetch_agent_card(endpoint, timeout_sec)
        tool.description = _build_description(
            "다른 에이전트에 요청을 보내는 A2A 기반의 툴입니다.",
            card,
        )
        return tool

    # -----------------------
    # Public entrypoint (single path)
    # -----------------------
    async def _arun(self, **kwargs) -> str:  # CrewAI 표준 비동기 진입점
        params = self.args_schema(**kwargs) if not isinstance(kwargs, self.args_schema) else kwargs

        message = params.message
        payload = params.payload
        skill = params.skill
        accepted_output_modes = params.accepted_output_modes or ["text"]
        blocking = params.blocking
        timeout_sec = params.timeout_sec

        if not message and payload is not None:
            message = _format_payload_to_message(skill, payload)
        elif not message and payload is None:
            return json.dumps({"error": "Either 'message' or 'payload' is required."}, ensure_ascii=False)

        a2a_msg = Message(
            message_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=message, kind="text"))],
            role=Role.user,
        )
        req = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(
                message=a2a_msg,
                configuration=MessageSendConfiguration(
                    acceptedOutputModes=accepted_output_modes,
                    blocking=blocking,
                ),
            ),
        )

        try:
            async with httpx.AsyncClient(timeout=timeout_sec, headers=self._endpoint.headers or None) as httpx_client:
                client = A2AClient(httpx_client=httpx_client, url=self._endpoint.url)
                resp = await client.send_message(req)
                task = getattr(getattr(resp, "root", None), "result", None)
                result_text, history, task_state = _compact_history(task)
                return json.dumps(
                    {
                        "result": result_text,
                        "meta": {"task_state": task_state, "history": history},
                    },
                    ensure_ascii=False,
                )
        except httpx.ConnectError as e:
            logger.info("Endpoint 연결 실패(%s): %s", self._endpoint.url, e)
            return json.dumps(
                {
                    "result": "",
                    "meta": {"task_state": "simulated", "note": f"Endpoint 연결 실패: {self._endpoint.url}"},
                    "error": str(e),
                },
                ensure_ascii=False,
            )
        except Exception as e:  # 모든 예외를 구조화
            logger.exception("A2A 호출 실패")
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    # CrewAI가 동기 호출만 지원하는 실행 경로에서도 사용 가능하도록 thin wrapper 제공
    def _run(self, **kwargs) -> str:  # noqa: D401
        """동기 컨텍스트에서 안전하게 비동기 본체 실행."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 이미 이벤트 루프가 돌고 있으면 그 루프에 태스크로 붙임
            return asyncio.run_coroutine_threadsafe(self._arun(**kwargs), loop).result()

        # 루프가 없으면 새로 생성하여 실행
        return asyncio.run(self._arun(**kwargs))
