from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional, List, Type, Dict, Any, Literal

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from ..utils.context_manager import get_context_snapshot
from ..utils.database import (
    fetch_human_response_sync,
    save_notification_sync,
    save_event_sync,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------
class HumanQuerySchema(BaseModel):
    """사용자 확인/추가정보 요청 스키마 (간결 버전)"""
    role: str = Field(..., description="질의 대상(예: user, manager)")
    text: str = Field(..., description="질의 내용")
    type: Literal["text", "select", "confirm"] = Field(default="text", description="질의 유형")
    options: Optional[List[str]] = Field(default=None, description="type이 select일 때 선택지")


# ---------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------
class HumanQueryTool(BaseTool):
    """사람에게 질문을 보내고, DB(events)에서 응답을 감지하는 도구."""

    name: str = "human_asked"
    description: str = (
        "👀 질문은 반드시 '매우 구체적이고 세부적'으로 작성해야 합니다.\n"
        "- 목적, 대상, 범위/경계, 입력/출력 형식, 성공/실패 기준, 제약조건(보안/권한/시간/용량),\n"
        "  필요한 식별자/예시/반례까지 모두 명시하세요. 추측으로 진행하지 말고 누락 정보를 반드시 질문하세요.\n\n"
        "[1] 언제 사용해야 하나\n"
        "1. 보안에 민감한 정보(개인정보/인증정보/비밀키 등)를 다루거나 외부로 전송할 때\n"
        "2. 데이터베이스에 '저장/수정/삭제' 작업을 수행할 때 (읽기 전용 조회는 제외)\n"
        "3. 요구사항 및 작업지시사항이 모호·불완전·추정에 의존하거나, 전제조건/매개변수가 불명확할 때\n"
        "4. 외부 시스템 연동, 파일 생성/이동/삭제 등 시스템 상태를 바꾸는 작업일 때\n"
        "⛔ 위 조건에 해당하면 이 도구 없이 진행 금지\n\n"
        "[2] 응답 타입과 작성 방식 (항상 JSON으로 질의 전송)\n"
        "- 공통 형식: { role: <누구에게>, text: <질의>, type: <text|select|confirm>, options?: [선택지...] }\n"
        "- 질의 작성 가이드(반드시 포함): 5W1H, 목적/맥락, 선택 이유 또는 승인 근거, 기본값/제약,\n"
        "  입력/출력 형식과 예시, 반례/실패 시 처리, 보안/권한/감사 로그 요구사항, 마감/우선순위\n\n"
        "// 1) type='text' — 정보 수집(모호/불완전할 때 필수)\n"
        "{\n"
        '  "role": "user",\n'
        '  "text": "어떤 DB 테이블/스키마/키로 저장할까요? 입력값 예시/형식, 실패 시 처리, 보존 기간까지 구체히 알려주세요.",\n'
        '  "type": "text"\n'
        "}\n\n"
        "// 2) type='select' — 여러 옵션 중 선택(옵션은 상호배타적, 명확/완전하게 제시)\n"
        "{\n"
        '  "role": "system",\n'
        '  "text": "배포 환경을 선택하세요. 선택 근거(위험/롤백/감사 로그)를 함께 알려주세요.",\n'
        '  "type": "select",\n'
        '  "options": ["dev", "staging", "prod"]\n'
        "}\n\n"  
        "// 3) type='confirm' — 보안/DB 변경 등 민감 작업 승인(필수)\n"
        "{\n"
        '  "role": "user",\n'
        '  "text": "DB에서 주문 상태를 shipped로 업데이트합니다. 대상: order_id=..., 영향 범위: ...건, 롤백: ..., 진행 승인하시겠습니까?",\n'
        '  "type": "confirm"\n'
        "}\n\n"
        "타입 선택 규칙\n"
        "- text: 모호/누락 정보가 있을 때 먼저 세부사항을 수집 (여러 번 질문 가능)\n"
        "- select: 옵션이 둘 이상이면 반드시 options로 제시하고, 선택 기준을 text에 명시\n"
        "- confirm: DB 저장/수정/삭제, 외부 전송, 파일 조작 등은 승인 후에만 진행\n\n"
        "[3] 주의사항\n"
        "- 이 도구 없이 민감/변경 작업을 임의로 진행 금지.\n"
        "- select 타입은 반드시 'options'를 포함.\n"
        "- confirm 응답에 따라: ✅ 승인 → 즉시 수행 / ❌ 거절 → 즉시 중단(건너뛰기).\n"
        "- 애매하면 추가 질문을 반복하고, 충분히 구체화되기 전에는 실행하지 말 것.\n"
        "- 민감 정보는 최소한만 노출하고 필요 시 마스킹/요약.\n"
        "- 예시를 그대로 사용하지 말고 컨텍스트에 맞게 반드시 자연스러운 질의를 재작성하세요.\n"
        "- 타임아웃/미응답 시 '사용자 미응답 거절'을 반환하며, 후속 변경 작업을 중단하는 것이 안전.\n"
        "- 한 번에 하나의 주제만 질문(여러 주제면 질문을 분리). 한국어 존댓말 사용, 간결하되 상세하게.")

    args_schema: Type[HumanQuerySchema] = HumanQuerySchema

    def __init__(
        self,
        *,
        proc_inst_id: str,
        task_id: str,
        tenant_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        user_ids_csv: Optional[str] = None,  # 알림 대상 (CSV)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._proc_inst_id = proc_inst_id
        self._task_id = task_id
        self._tenant_id = tenant_id
        self._agent_name = agent_name
        self._user_ids_csv = user_ids_csv

        logger.info("\n\n✅ HumanQueryTool 초기화 완료 | proc_inst_id=%s task_id=%s tenant_id=%s agent_name=%s user_ids_csv=%s", proc_inst_id, task_id, tenant_id, agent_name, user_ids_csv)

    # CrewAI Tool 규약: 동기 실행 (내부 비동기 작업은 sync 래퍼 사용)
    def _run(self, role: str, text: str, type: str = "text", options: Optional[List[str]] = None) -> str:
        logger.info("\n\n👤 사용자 확인 요청 시작 | role=%s type=%s", role, type)
        
        # 1) 컨텍스트 정보 가져오기
        ctx = get_context_snapshot()
        crew_type = ctx.get("crew_type")

        # 2) 메시지 페이로드 구성
        payload: Dict[str, Any] = {
            "role": role,
            "text": text,
            "type": type,
            "options": options or [],
        }

        # 3) job_id 발급
        job_id = f"human_asked_{uuid.uuid4()}"

        # 4) 이벤트를 DB에 직접 저장
        try:
            save_event_sync(
                job_id=job_id,
                todo_id=self._task_id,
                proc_inst_id=self._proc_inst_id,
                crew_type=crew_type,
                data=payload,
                event_type="human_asked",
            )
            logger.info("✅ 사용자 확인 이벤트 DB 저장 완료 | proc=%s task=%s job_id=%s", self._proc_inst_id, self._task_id, job_id)
        except Exception as e:
            logger.error("❌ 사용자 확인 이벤트 DB 저장 실패 | proc=%s task=%s job_id=%s err=%s", self._proc_inst_id, self._task_id, job_id, str(e), exc_info=True)
            raise

        # 5) 알림 저장 (있으면)
        try:
            if self._user_ids_csv and self._user_ids_csv.strip():
                save_notification_sync(
                    title=text,
                    notif_type="workitem_bpm",
                    description=self._agent_name,
                    user_ids_csv=self._user_ids_csv,
                    tenant_id=self._tenant_id,
                    url=f"/todolist/{self._task_id}" if self._task_id else None,
                    from_user_id=self._agent_name,
                )
                logger.info("✅ 사용자 알림 저장 완료 | user_ids_csv=%s", self._user_ids_csv)
            else:
                logger.info("⏭️ 사용자 알림 저장 생략: user_ids_csv 비어있음")
        except Exception as e:
            logger.error("❌ 사용자 알림 저장 실패 | user_ids_csv=%s err=%s", self._user_ids_csv, str(e), exc_info=True)
            raise

        # 6) DB에서 사람 응답 폴링
        logger.info("\n\n⏳ 사용자 응답 대기 시작 | job_id=%s", job_id)
        answer = self._wait_for_response(job_id)
        logger.info("✅ 사용자 응답 수신 완료 | job_id=%s answer_length=%d", job_id, len(answer) if answer else 0)
        return answer

    # -----------------------------------------------------------------
    # 응답 폴링 (DB events 테이블)
    # -----------------------------------------------------------------
    def _wait_for_response(self, job_id: str, timeout_sec: int = 180, poll_interval_sec: int = 5) -> str:
        deadline = time.time() + timeout_sec
        error_count = 0

        while time.time() < deadline:
            try:
                event = fetch_human_response_sync(job_id=job_id)
                if event:
                    data = (event.get("data") or {})
                    answer = data.get("answer")
                    if isinstance(answer, str):
                        logger.info("✅ 사용자 응답 수신 성공 | job_id=%s", job_id)
                        return answer
                    return json.dumps(data, ensure_ascii=False)
                error_count = 0  # 성공 시 에러 카운트 리셋
            except Exception as e:
                logger.error("❌ 사용자 응답 폴링 오류 | job_id=%s err=%s", job_id, str(e), exc_info=True)
                error_count += 1
                if error_count >= 3:
                    logger.error("💥 사용자 응답 폴링 중단 | job_id=%s 연속 오류 3회", job_id)
                    raise RuntimeError("human_asked polling aborted after 3 consecutive errors") from e
                logger.warning("⚠️ 사용자 응답 폴링 재시도 | job_id=%s error_count=%d", job_id, error_count)
            
            time.sleep(poll_interval_sec)

        logger.warning("⏰ 사용자 응답 타임아웃 | job_id=%s timeout=%ds", job_id, timeout_sec)
        return "사용자 미응답 거절"
