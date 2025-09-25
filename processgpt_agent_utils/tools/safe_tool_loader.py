from __future__ import annotations

import os
import subprocess
import time
import logging
from typing import List

import anyio
from mcp.client.stdio import StdioServerParameters
from crewai_tools import MCPServerAdapter

from .knowledge_manager import Mem0Tool, MementoTool
from .human_query_tool import HumanQueryTool\

from processgpt_agent_utils.utils.context_manager import proc_inst_id_var, task_id_var, users_email_var

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
        logger.info("🔧 SafeToolLoader 초기화 완료 | tenant_id=%s, user_id=%s", tenant_id, user_id)

    def warmup_server(self, server_key: str):
        """npx 기반 서버의 패키지를 미리 캐시에 저장"""
        cfg = self._get_mcp_config(server_key)
        if not cfg or cfg.get("command") != "npx":
            return

        npx = self._find_npx_command()
        if not npx:
            return

        args = cfg.get("args", [])
        if not (args and args[0] == "-y"):
            return

        pkg = args[1]
        try:
            subprocess.run([npx, "-y", pkg, "--help"], capture_output=True, timeout=10, shell=True)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            logger.debug("npx 패키지 캐시 실패 (무시): %s", str(e))
            pass

        try:
            subprocess.run([npx, "-y", pkg, "--help"], capture_output=True, timeout=60, shell=True)
        except Exception as e:
            logger.debug("npx 패키지 캐시 실패 (무시): %s", str(e))
            pass

    def _find_npx_command(self) -> str:
        try:
            import shutil
            npx_path = shutil.which("npx") or shutil.which("npx.cmd")
            if npx_path:
                return npx_path
        except Exception as e:
            logger.debug("npx 명령어 찾기 실패 (기본값 사용): %s", str(e))
            pass
        return "npx"

    def create_tools_from_names(self, tool_names: List[str]) -> List:
        """tool_names 리스트에서 실제 Tool 객체 생성"""
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        logger.info("🛠️ 도구 생성 요청: %s", tool_names)

        tools = []
        tools.extend(self._load_mem0())
        tools.extend(self._load_memento())
        tools.extend(self._load_human_asked())

        for name in tool_names:
            key = name.strip().lower()
            if key in self.local_tools:
                continue
            else:
                self.warmup_server(key)
                tools.extend(self._load_mcp_tool(key))

        logger.info("✅ 도구 생성 완료: 총 %d개", len(tools))
        return tools

    # ======================================================================
    # 개별 도구 로더
    # ======================================================================
    def _load_mem0(self) -> List:
        try:
            if not self.user_id:
                logger.info("⏭️ mem0 도구 로드 생략: user_id 없음")
                return []
            return [Mem0Tool(tenant_id=self.tenant_id, user_id=self.user_id)]
        except Exception as e:
            logger.error("❌ mem0 도구 로드 실패 | tenant_id=%s user_id=%s err=%s", self.tenant_id, self.user_id, str(e), exc_info=True)
            raise

    def _load_memento(self) -> List:
        try:
            if not self.tenant_id:
                logger.info("⏭️ memento 도구 로드 생략: tenant_id 없음")
                return []
            return [MementoTool(tenant_id=self.tenant_id)]
        except Exception as e:
            logger.error("❌ memento 도구 로드 실패 | tenant_id=%s err=%s", self.tenant_id, str(e), exc_info=True)
            raise

    def _load_human_asked(self) -> List:
        try:
            if not self.tenant_id:
                logger.info("⏭️ human_asked 도구 로드 생략: tenant_id 없음")
                return []
            if not self.agent_name:
                logger.info("⏭️ human_asked 도구 로드 생략: agent_name 없음")
                return []

            return [HumanQueryTool(
                proc_inst_id=proc_inst_id_var.get(),
                task_id=task_id_var.get(),
                tenant_id=self.tenant_id,
                agent_name=self.agent_name,
                user_ids_csv=users_email_var.get(),
            )]
        except Exception as e:
            logger.error("❌ human_asked 도구 로드 실패 | tenant_id=%s user_id=%s err=%s", self.tenant_id, self.user_id, str(e), exc_info=True)
            raise

    def _load_mcp_tool(self, tool_name: str) -> List:
        """MCP 도구 로드 (timeout & retry 지원)"""
        self._apply_anyio_patch()

        server_cfg = self._get_mcp_config(tool_name)
        if not server_cfg:
            return []

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

                params = StdioServerParameters(
                    command=str(cmd),
                    args=safe_args,
                    env=safe_env,
                    timeout=int(timeout),
                )

                adapter = MCPServerAdapter(params)
                SafeToolLoader.adapters.append(adapter)
                logger.info("✅ %s MCP 서버 연결 성공 | 도구 %d개: %s", tool_name, len(adapter.tools), [t.name for t in adapter.tools])
                return adapter.tools

            except Exception as e:
                logger.warning("⚠️ %s MCP 서버 연결 실패 (시도 %d/%d) | err=%s", tool_name, attempt, max_retries, str(e), exc_info=True)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.error("❌ %s MCP 서버 최종 연결 실패 | 모든 재시도 소진", tool_name)
                    raise

    # ======================================================================
    # 헬퍼
    # ======================================================================
    def _apply_anyio_patch(self):
        """anyio stderr 패치 적용"""
        from anyio._core._subprocesses import open_process as _orig

        async def patched_open_process(*args, **kwargs):
            stderr = kwargs.get("stderr")
            if not (hasattr(stderr, "fileno") and stderr.fileno()):
                kwargs["stderr"] = subprocess.PIPE
            return await _orig(*args, **kwargs)

        anyio.open_process = patched_open_process
        anyio._core._subprocesses.open_process = patched_open_process

    def _get_mcp_config(self, tool_name: str) -> dict:
        """인자로 전달받은 MCP 설정에서 특정 도구 설정 반환"""
        try:
            if not self.mcp_config:
                return {}
            return self.mcp_config.get("mcpServers", {}).get(tool_name, {}) or {}
        except Exception as e:
            logger.error("❌ MCP 설정 로드 실패 | tool=%s err=%s", tool_name, str(e), exc_info=True)
            raise

    @classmethod
    def shutdown_all_adapters(cls):
        """모든 MCPServerAdapter 연결 종료"""
        for adapter in cls.adapters:
            try:
                adapter.stop()
            except Exception as e:
                logger.error("❌ MCP 어댑터 종료 실패 | err=%s", str(e), exc_info=True)
                raise
        logger.info("🔌 모든 MCPServerAdapter 연결 종료 완료")
        cls.adapters.clear()
