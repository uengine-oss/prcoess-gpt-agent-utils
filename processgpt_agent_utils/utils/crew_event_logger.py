from __future__ import annotations

import json
import logging
import asyncio
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

# CrewAI 이벤트 임포트 (신/구 버전 호환)
try:
    from crewai.events import CrewAIEventsBus
    from crewai.events import (
        TaskStartedEvent,
        TaskCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
    )
except ImportError:  # 구버전
    from crewai.utilities.events import CrewAIEventsBus
    from crewai.utilities.events.task_events import TaskStartedEvent, TaskCompletedEvent
    from crewai.utilities.events import ToolUsageStartedEvent, ToolUsageFinishedEvent

# context_var import
from .context_manager import task_id_var, proc_inst_id_var, crew_type_var

# database 저장 함수 import
from .database import save_event


class CrewAIEventLogger:
    """CrewAI 이벤트 → events 테이블 저장 (단순/가독성 우선)"""

    # --------- 공개 엔트리포인트 ---------
    def on_event(self, event: Any, source: Any = None) -> None:
        """
        이벤트 수신 → (job_id, event_type, data) 추출 → DB 저장
        - 동기 메서드이며 내부에서 asyncio.run(...)으로 비동기 저장 1회 수행
        - 모든 예외는 상위로 전파
        """
        try:
            event_type = self._extract_event_type(event)
            # 지원 타입만 처리
            if event_type not in ("task_started", "task_completed", "tool_usage_started", "tool_usage_finished"):
                return

            job_id = self._extract_job_id(event, source)
            data = self._extract_data(event, event_type)

            # context_var
            todo_id = task_id_var.get()
            proc_inst_id = proc_inst_id_var.get()
            crew_type = crew_type_var.get()

            event_id = asyncio.run(
                save_event(
                    job_id=job_id,
                    todo_id=todo_id,
                    proc_inst_id=proc_inst_id,
                    crew_type=crew_type,
                    data=data,
                    event_type=event_type,
                    status=None,
                )
            )
            logger.info(
                "✅ 이벤트 저장: id=%s job_id=%s type=%s crew_type=%s todo_id=%s proc_inst_id=%s",
                event_id, job_id, event_type, str(crew_type), str(todo_id), str(proc_inst_id)
            )

        except Exception as e:
            logger.error("❌ on_event 실패: %s", e, exc_info=True)
            # 경계에서 예외 전파
            raise

    # --------- 헬퍼(가독성 유지용 최소) ---------
    def _extract_job_id(self, event: Any, source: Any = None) -> str:
        try:
            if hasattr(event, "task") and hasattr(event.task, "id"):
                return str(event.task.id)
            if source and hasattr(source, "task") and hasattr(source.task, "id"):
                return str(source.task.id)
            if hasattr(event, "job_id"):
                return str(getattr(event, "job_id"))
        except Exception as e:
            logger.warning("⚠️ job_id 추출 경고: %s", str(e), exc_info=True)
        return "unknown"

    def _extract_event_type(self, event: Any) -> str:
        try:
            if hasattr(event, "type") and isinstance(event.type, str):
                return event.type
        except Exception as e:
            logger.debug("이벤트 타입 추출 실패 (기본값 사용): %s", str(e))
            pass
        name = event.__class__.__name__.lower()
        if "taskstarted" in name:
            return "task_started"
        if "taskcompleted" in name:
            return "task_completed"
        if "toolusagestarted" in name:
            return "tool_usage_started"
        if "toolusagefinished" in name:
            return "tool_usage_finished"
        return "unknown"

    def _extract_data(self, event: Any, event_type: str) -> Dict[str, Any]:
        try:
            if event_type == "task_started":
                agent = getattr(getattr(event, "task", None), "agent", None)
                return {
                    "role": getattr(agent, "role", None) or "Unknown",
                    "goal": getattr(agent, "goal", None) or "Unknown",
                    "agent_profile": getattr(agent, "profile", None) or "/images/chat-icon.png",
                    "name": getattr(agent, "name", None) or "Unknown",
                }

            if event_type == "task_completed":
                # output 우선순위: event.output.raw -> event.output(str) -> event.result
                output = getattr(event, "output", None)
                text = getattr(output, "raw", None)
                if text is None:
                    text = output if isinstance(output, str) else getattr(event, "result", None)
                parsed = self._safe_json(text)

                # ✅ planning 포맷(list_of_plans_per_task) → Markdown 축약 (기존과 동일)
                if isinstance(parsed, dict) and "list_of_plans_per_task" in parsed:
                    md = self._format_plans_md(parsed["list_of_plans_per_task"])
                    return {"plans": md}

                # 개인정보 가능성 있는 폼데이터 제거 (기존 정책 유지)
                if isinstance(parsed, dict) and "폼_데이터" in parsed:
                    parsed = {k: v for k, v in parsed.items() if k != "폼_데이터"}

                return {"result": parsed}

            if event_type in ("tool_usage_started", "tool_usage_finished"):
                tool_name = getattr(event, "tool_name", None)
                tool_args = getattr(event, "tool_args", None)
                args = self._safe_json(tool_args)
                query = args.get("query") if isinstance(args, dict) else None
                return {"tool_name": tool_name, "query": query, "args": args}

            return {"info": f"Unhandled event type: {event_type}"}

        except Exception as e:
            logger.error("❌ 데이터 추출 실패: %s", str(e), exc_info=True)
            raise

    # --------- 단순 유틸 ---------
    def _safe_json(self, value: Any) -> Any:
        if value is None or isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except Exception as e:
            logger.debug("JSON 파싱 실패 (원본 반환): %s", str(e))
            return value

    def _format_plans_md(self, plans: List[Dict[str, Any]]) -> str:
        """list_of_plans_per_task → Markdown 문자열로 축약"""
        lines: List[str] = []
        for idx, item in enumerate(plans, 1):
            task = item.get("task", "")
            plan = item.get("plan", "")
            lines.append(f"## {idx}. {task}")
            lines.append("")
            if isinstance(plan, list):
                for line in plan:
                    lines.append(str(line))
            elif isinstance(plan, str):
                lines.extend(plan.splitlines())
            else:
                lines.append(str(plan))
            lines.append("")
        return "\n".join(lines).strip()


class CrewConfigManager:
    """CrewAI 설정 관리자"""
    
    def __init__(self):
        self.logger = CrewAIEventLogger()
        logger.info("🔧 CrewConfigManager 초기화 완료")
    
    def setup_crew_logging(self, crew_instance):
        """Crew 인스턴스에 이벤트 로깅 설정"""
        try:
            if hasattr(crew_instance, 'events_bus'):
                crew_instance.events_bus.subscribe(self.logger.on_event)
                logger.info("✅ Crew 이벤트 로깅 설정 완료")
            else:
                logger.warning("⚠️ Crew 인스턴스에 events_bus가 없습니다")
        except Exception as e:
            logger.error("❌ Crew 로깅 설정 실패: %s", str(e), exc_info=True)
            raise
