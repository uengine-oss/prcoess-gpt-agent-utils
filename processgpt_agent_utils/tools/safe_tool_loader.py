from __future__ import annotations

import os
import subprocess
import time
import logging
from typing import List, Optional, Dict

import anyio
from mcp.client.stdio import StdioServerParameters
from crewai_tools import MCPServerAdapter

from .knowledge_manager import Mem0Tool, MementoTool
from .human_query_tool import HumanQueryTool

from processgpt_agent_utils.utils.context_manager import proc_inst_id_var, task_id_var, users_email_var

# === A2A 전용 임포트 (로컬 모듈 사용) ======================================
# 프로젝트 내 제공되는 a2a_client_tool에서 직접 임포트
try:
    from .a2a_client_tool import A2AAgentTool, AgentEndpoint  # 리팩토링된 A2A 툴
except Exception:  # 모듈 미존재 시 A2A는 건너뛰도록 처리
    A2AAgentTool = None  # type: ignore
    AgentEndpoint = None  # type: ignore
# ============================================================================

logger = logging.getLogger(__name__)

class SafeToolLoader:
    """도구 로더 클래스"""
    adapters = []  # MCPServerAdapter 인스턴스 등록

    def __init__(self, tenant_id: str = None, user_id: str = None, agent_name: str = None, mcp_config: dict = None):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.agent_name = agent_name
        self.mcp_config = mcp_config or {}
        self.local_tools = ["mem0", "memento", "human_asked"]
        logger.info("\n\n✅ SafeToolLoader 초기화 완료 | tenant_id=%s, user_id=%s, local_tools=%s", tenant_id, user_id, self.local_tools)

    def warmup_server(self, server_key: str):
        """npx 기반 서버의 패키지를 미리 캐시에 저장"""
        logger.debug("🔥 서버 워밍업 시작 | server_key=%s", server_key)
        cfg = self._get_mcp_config(server_key)
        if not cfg or cfg.get("command") != "npx":
            logger.debug("⏭️ 서버 워밍업 생략: npx 명령어 아님 | server_key=%s", server_key)
            return

        npx = self._find_npx_command()
        if not npx:
            logger.debug("⏭️ 서버 워밍업 생략: npx 명령어 찾을 수 없음 | server_key=%s", server_key)
            return

        args = cfg.get("args", [])
        if not (args and args[0] == "-y"):
            logger.debug("⏭️ 서버 워밍업 생략: -y 플래그 없음 | server_key=%s", server_key)
            return

        pkg = args[1]
        try:
            subprocess.run([npx, "-y", pkg, "--help"], capture_output=True, timeout=10, shell=True)
            logger.info("✅ NPX 패키지 캐시 성공 (빠른) | server_key=%s pkg=%s", server_key, pkg)
            return
        except subprocess.TimeoutExpired:
            logger.debug("⏰ NPX 패키지 캐시 타임아웃 (빠른) | server_key=%s pkg=%s", server_key, pkg)
            pass
        except Exception as e:
            logger.debug("⚠️ NPX 패키지 캐시 실패 (빠른, 무시) | server_key=%s pkg=%s err=%s", server_key, pkg, str(e))
            pass

        try:
            subprocess.run([npx, "-y", pkg, "--help"], capture_output=True, timeout=60, shell=True)
            logger.info("✅ NPX 패키지 캐시 성공 (느린) | server_key=%s pkg=%s", server_key, pkg)
        except Exception as e:
            logger.debug("⚠️ NPX 패키지 캐시 실패 (느린, 무시) | server_key=%s pkg=%s err=%s", server_key, pkg, str(e))
            pass

    def _find_npx_command(self) -> str:
        try:
            import shutil
            npx_path = shutil.which("npx") or shutil.which("npx.cmd")
            if npx_path:
                logger.debug("✅ NPX 명령어 발견 | path=%s", npx_path)
                return npx_path
        except Exception as e:
            logger.debug("⚠️ NPX 명령어 찾기 실패 (기본값 사용) | err=%s", str(e))
            pass
        logger.debug("📝 NPX 명령어 기본값 사용 | path=npx")
        return "npx"

    # ----------------------- [변경] agent_type / a2a_endpoints 인자 추가 -----------------------
    def create_tools_from_names(self, tool_names: List[str], agent_type: Optional[str] = None, a2a_endpoints: Optional[Dict[str, Dict]] = None) -> List:
        """tool_names 리스트에서 실제 Tool 객체 생성
        - agent_type: 'a2a' 또는 None/'' (None/'' 이면 A2A 건너뜀)
        - a2a_endpoints: {'airbnb': {'url': '...', 'headers': {...}}, ...}
        """
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        logger.info("🛠️ 도구 생성 요청 시작 | tool_names=%s agent_type=%s", tool_names, agent_type)

        tools = []
        
        # 기본 로컬 도구들 로드 (항상)
        logger.info("📦 기본 로컬 도구들 로드 시작 | local_tools=%s", self.local_tools)
        mem0_tools = self._load_mem0()
        memento_tools = self._load_memento()
        human_asked_tools = self._load_human_asked()
        tools.extend(mem0_tools)
        tools.extend(memento_tools)
        tools.extend(human_asked_tools)
        logger.info("✅ 기본 로컬 도구들 로드 완료 | mem0=%d memento=%d human_asked=%d total=%d", 
                   len(mem0_tools), len(memento_tools), len(human_asked_tools), len(tools))

        # ------------------------------
        # [추가] A2A 툴 로드: agent_type == 'a2a' 인 경우만 수행
        # 규칙: tool_names 중 'a2a:' 프리픽스가 붙은 이름만 A2A 후보로 추출
        # ------------------------------
        if (agent_type or "").lower() == "a2a":
            a2a_candidates: List[str] = []
            for name in tool_names:
                if isinstance(name, str) and name.lower().startswith("a2a:"):
                    a2a_candidates.append(name.split(":", 1)[1].strip())
            if a2a_candidates:
                logger.info("🚀 A2A 도구 로드 시작 | candidates=%s", a2a_candidates)
                tools.extend(self._load_a2a_tools(a2a_candidates, a2a_endpoints))
                logger.info("✅ A2A 도구 로드 완료 | total_tools=%d", len(tools))
            else:
                logger.info("⏭️ A2A 도구 로드 생략: 'a2a:' 프리픽스 없음")

        # ------------------------------
        # MCP 도구 로드: JSON 설정이 있을 경우에만 (기존 로직 유지)
        # ------------------------------
        logger.info("🔧 요청된 도구들 처리 시작 | requested_tools=%s", tool_names)
        for name in tool_names:
            key = name.strip().lower()
            logger.info("🔍 도구 처리 중: %s", key)
            
            if key in self.local_tools:
                logger.info("⏭️ 도구 처리 생략: 이미 로컬 도구로 로드됨 | key=%s", key)
                continue
            if key.startswith("a2a:"):
                logger.info("⏭️ 도구 처리 생략: A2A는 상단 분기에서 처리 | key=%s", key)
                continue

            # MCP: 설정이 있을 경우에만 로딩
            logger.info("🚀 MCP 도구 로드 시작 | key=%s", key)
            self.warmup_server(key)
            mcp_tools = self._load_mcp_tool(key)
            tools.extend(mcp_tools)
            logger.info("✅ MCP 도구 로드 완료 | key=%s tools_count=%d", key, len(mcp_tools))

        logger.info("🎉 도구 생성 완료 | total_tools=%d tool_names=%s", len(tools), [t.name if hasattr(t, 'name') else str(t) for t in tools])
        return tools

    # ======================================================================
    # 개별 도구 로더
    # ======================================================================
    def _load_mem0(self) -> List:
        logger.debug("🧠 Mem0Tool 로드 시작 | user_id=%s", self.user_id)
        try:
            if not self.user_id:
                logger.info("⏭️ Mem0Tool 로드 생략: user_id 없음")
                return []
            tool = Mem0Tool(tenant_id=self.tenant_id, user_id=self.user_id)
            logger.info("✅ Mem0Tool 로드 완료 | user_id=%s", self.user_id)
            return [tool]
        except Exception as e:
            logger.error("❌ Mem0Tool 로드 실패 | tenant_id=%s user_id=%s err=%s", self.tenant_id, self.user_id, str(e), exc_info=True)
            raise

    def _load_memento(self) -> List:
        logger.debug("🔒 MementoTool 로드 시작 | tenant_id=%s", self.tenant_id)
        try:
            if not self.tenant_id:
                logger.info("⏭️ MementoTool 로드 생략: tenant_id 없음")
                return []
            tool = MementoTool(tenant_id=self.tenant_id)
            logger.info("✅ MementoTool 로드 완료 | tenant_id=%s", self.tenant_id)
            return [tool]
        except Exception as e:
            logger.error("❌ MementoTool 로드 실패 | tenant_id=%s err=%s", self.tenant_id, str(e), exc_info=True)
            raise

    def _load_human_asked(self) -> List:
        logger.debug("👤 HumanQueryTool 로드 시작 | tenant_id=%s agent_name=%s", self.tenant_id, self.agent_name)
        try:
            if not self.tenant_id:
                logger.info("⏭️ HumanQueryTool 로드 생략: tenant_id 없음")
                return []
            if not self.agent_name:
                logger.info("⏭️ HumanQueryTool 로드 생략: agent_name 없음")
                return []

            tool = HumanQueryTool(
                proc_inst_id=proc_inst_id_var.get(),
                task_id=task_id_var.get(),
                tenant_id=self.tenant_id,
                agent_name=self.agent_name,
                user_ids_csv=users_email_var.get(),
            )
            logger.info("✅ HumanQueryTool 로드 완료 | tenant_id=%s agent_name=%s", self.tenant_id, self.agent_name)
            return [tool]
        except Exception as e:
            logger.error("❌ HumanQueryTool 로드 실패 | tenant_id=%s agent_name=%s err=%s", self.tenant_id, self.agent_name, str(e), exc_info=True)
            raise

    def _load_mcp_tool(self, tool_name: str) -> List:
        """MCP 도구 로드 (timeout & retry 지원)"""
        logger.info("🔧 MCP 도구 로드 시작 | tool_name=%s", tool_name)
        self._apply_anyio_patch()

        server_cfg = self._get_mcp_config(tool_name)
        if not server_cfg:
            logger.warning("⚠️ MCP 도구 로드 생략: 설정 없음 | tool_name=%s", tool_name)
            return []

        logger.info("📋 MCP 서버 설정 확인 완료 | tool_name=%s config_keys=%s", tool_name, list(server_cfg.keys()))

        env_vars = os.environ.copy()
        env_vars.update(server_cfg.get("env", {}))
        timeout = server_cfg.get("timeout", 40)

        max_retries = 2
        retry_delay = 5

        for attempt in range(1, max_retries + 1):
            try:
                cmd = server_cfg["command"]
                if cmd == "npx":
                    cmd = self._find_npx_command() or cmd

                safe_args = [str(a) for a in server_cfg.get("args", [])]
                safe_env = {k: str(v) for k, v in (env_vars or {}).items()}

                logger.info("🚀 MCP 서버 시작 시도 %d/%d | tool_name=%s cmd=%s args=%s timeout=%d", 
                           attempt, max_retries, tool_name, cmd, safe_args, timeout)

                params = StdioServerParameters(
                    command=str(cmd),
                    args=safe_args,
                    env=safe_env,
                    timeout=int(timeout),
                )

                adapter = MCPServerAdapter(params)
                SafeToolLoader.adapters.append(adapter)
                tool_names = [t.name for t in adapter.tools]
                logger.info("✅ MCP 서버 연결 성공 | tool_name=%s tools_count=%d tool_names=%s", 
                           tool_name, len(adapter.tools), tool_names)
                return adapter.tools

            except Exception as e:
                logger.warning("⚠️ MCP 서버 연결 실패 (시도 %d/%d) | tool_name=%s err=%s", 
                              attempt, max_retries, tool_name, str(e), exc_info=True)
                if attempt < max_retries:
                    logger.info("⏳ MCP 서버 재시도 대기 | tool_name=%s delay=%ds", tool_name, retry_delay)
                    time.sleep(retry_delay)
                else:
                    logger.error("❌ MCP 서버 최종 연결 실패 | tool_name=%s 모든 재시도 소진", tool_name)
                    raise

    # ======================================================================
    # 헬퍼
    # ======================================================================
    def _apply_anyio_patch(self):
        """anyio stderr 패치 적용"""
        logger.debug("🔧 anyio stderr 패치 적용 시작")
        from anyio._core._subprocesses import open_process as _orig

        async def patched_open_process(*args, **kwargs):
            stderr = kwargs.get("stderr")
            if not (hasattr(stderr, "fileno") and stderr.fileno()):
                kwargs["stderr"] = subprocess.PIPE
            return await _orig(*args, **kwargs)

        anyio.open_process = patched_open_process
        anyio._core._subprocesses.open_process = patched_open_process
        logger.debug("✅ anyio stderr 패치 적용 완료")

    def _get_mcp_config(self, tool_name: str) -> dict:
        """인자로 전달받은 MCP 설정에서 특정 도구 설정 반환"""
        logger.debug("🔍 MCP 설정 검색 시작 | tool_name=%s", tool_name)
        try:
            if not self.mcp_config:
                logger.warning("⚠️ MCP 설정 검색 실패: 설정 없음 | tool_name=%s", tool_name)
                return {}
            
            mcp_servers = self.mcp_config.get("mcpServers", {})
            if not mcp_servers:
                logger.warning("⚠️ MCP 설정 검색 실패: mcpServers 섹션 없음 | tool_name=%s", tool_name)
                return {}
            
            tool_config = mcp_servers.get(tool_name, {})
            if not tool_config:
                logger.warning("⚠️ MCP 설정 검색 실패: 서버 설정 없음 | tool_name=%s available_servers=%s", tool_name, list(mcp_servers.keys()))
                return {}
            
            logger.info("✅ MCP 설정 발견 | tool_name=%s config_keys=%s", tool_name, list(tool_config.keys()))
            return tool_config
            
        except Exception as e:
            logger.error("❌ MCP 설정 검색 실패 | tool_name=%s err=%s", tool_name, str(e), exc_info=True)
            raise

    # ======================= [추가] A2A 헬퍼 ==========================
    def _load_a2a_tools(self, a2a_names: List[str], a2a_endpoints: Optional[Dict[str, Dict]] = None) -> List:
        """
        A2A 도구 로드:
        - a2a_names: ['airbnb', 'jira', ...]
        - a2a_endpoints (선택): {'airbnb': {'url': 'http://...', 'headers': {...}}, ...}
          (없으면 환경변수 A2A_{NAME}_URL / A2A_{NAME}_HEADERS 에서 조회)
        """
        if A2AAgentTool is None or AgentEndpoint is None:
            logger.warning("⏭️ A2A 로드 생략: a2a_tools 모듈을 찾을 수 없음")
            return []

        loaded = []

        import asyncio
        async def _create_all():
            for name in a2a_names:
                endpoint = self._resolve_a2a_endpoint(name, a2a_endpoints)
                if not endpoint:
                    logger.warning("⚠️ A2A 엔드포인트 누락 → 스킵 | name=%s", name)
                    continue
                try:
                    tool = await A2AAgentTool.create(endpoint=endpoint, name=f"A2A:{name}", timeout_sec=60)
                    loaded.append(tool)
                    logger.info("✅ A2A 로드 완료 | name=%s url=%s", name, endpoint.url)
                except Exception as e:
                    logger.error("❌ A2A 로드 실패 | name=%s err=%s", name, str(e), exc_info=True)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(_create_all(), loop)
                fut.result()
            else:
                asyncio.run(_create_all())
        except Exception as e:
            logger.error("❌ A2A 로드 실행 실패 | err=%s", str(e), exc_info=True)

        return loaded

    def _resolve_a2a_endpoint(self, name: str, a2a_endpoints: Optional[Dict[str, Dict]] = None) -> Optional[AgentEndpoint]:
        """
        이름 기반 A2A 엔드포인트 해석 우선순위:
        1) a2a_endpoints 딕셔너리
        2) 환경변수 A2A_{NAME}_URL (필수), A2A_{NAME}_HEADERS (선택 JSON)
        """
        # 1) 명시 딕셔너리 우선
        if a2a_endpoints and name in a2a_endpoints:
            cfg = a2a_endpoints[name] or {}
            url = cfg.get("url")
            headers = cfg.get("headers") or {}
            if url:
                return AgentEndpoint(url=url, headers=headers)

        # 2) 환경변수
        key = name.upper().replace("-", "_")
        url = os.getenv(f"A2A_{key}_URL")
        if not url:
            return None
        headers_raw = os.getenv(f"A2A_{key}_HEADERS")
        headers: Dict[str, str] = {}
        if headers_raw:
            try:
                import json as _json
                headers = _json.loads(headers_raw) or {}
            except Exception:
                logger.warning("⚠️ A2A 헤더 JSON 파싱 실패 → 무시 | name=%s", name)
        return AgentEndpoint(url=url, headers=headers)
    # ===================================================================

    @classmethod
    def shutdown_all_adapters(cls):
        """모든 MCPServerAdapter 연결 종료"""
        logger.info("🔌 MCP 어댑터 종료 시작 | adapters_count=%d", len(cls.adapters))
        for i, adapter in enumerate(cls.adapters):
            try:
                logger.debug("🔌 MCP 어댑터 종료 시도 %d/%d", i+1, len(cls.adapters))
                adapter.stop()
                logger.debug("✅ MCP 어댑터 종료 성공 %d/%d", i+1, len(cls.adapters))
            except Exception as e:
                logger.error("❌ MCP 어댑터 종료 실패 %d/%d | err=%s", i+1, len(cls.adapters), str(e), exc_info=True)
                raise
        logger.info("✅ 모든 MCP 어댑터 종료 완료 | adapters_count=%d", len(cls.adapters))
        cls.adapters.clear()
