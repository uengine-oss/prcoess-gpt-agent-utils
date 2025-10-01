from __future__ import annotations

import os
import asyncio
import logging
import random
from typing import Any, Dict, Optional, Callable, TypeVar, List
import uuid
from dotenv import load_dotenv
from supabase import Client, create_client

T = TypeVar("T")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Retry utility (same policy/style as database.py)
# -----------------------------------------------------------------------------
async def _async_retry(
    fn: Callable[[], Any],
    *,
    name: str,
    retries: int = 3,
    base_delay: float = 0.8,
    fallback: Optional[Callable[[], Any]] = None,
) -> Any:
    """
    - 각 시도 실패: warning 로깅(시도/지연/에러 포함)
    - 최종 실패: FATAL 로깅(스택 포함), 예외를 상위로 전파
    - fallback 이 있으면 실행(실패 시에도 로깅 후 예외 전파)
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            # Supabase 파이썬 SDK는 동기이므로 to_thread로 감쌉니다.
            return await asyncio.to_thread(fn)
        except Exception as e:
            last_err = e
            jitter = random.uniform(0, 0.3)
            delay = base_delay * (2 ** (attempt - 1)) + jitter
            logger.warning(
                "⏳ 재시도 지연: name=%s attempt=%d/%d delay=%.2fs error=%s",
                name, attempt, retries, delay, str(e)
            )
            await asyncio.sleep(delay)

    # 최종 실패 - 예외 전파
    if last_err is not None:
        logger.error(
            "❌ 재시도 최종 실패: name=%s retries=%s error=%s",
            name, retries, str(last_err), exc_info=last_err
        )
        raise last_err

    if fallback is not None:
        try:
            return fallback()
        except Exception as fb_err:
            logger.error("❌ fallback 실패: name=%s error=%s", name, str(fb_err), exc_info=fb_err)
            raise fb_err
    
    # 이 지점에 도달하면 안 되지만, 안전을 위해 RuntimeError 발생
    raise RuntimeError(f"Unexpected state in _async_retry: name={name}")

# -----------------------------------------------------------------------------
# DB Client (same style/policy as database.py)
# -----------------------------------------------------------------------------
_db_client: Optional[Client] = None

def initialize_db() -> None:
    """
    Supabase 클라이언트 초기화.
    - production 이외 환경에서는 .env 로드
    - 환경변수 키 우선순위: (URL) SUPABASE_URL | SUPABASE_KEY_URL / (KEY) SUPABASE_KEY | SUPABASE_ANON_KEY
    - 실패 시 로깅 후 예외 전파(초기화 실패는 치명적)
    """
    global _db_client
    if _db_client is not None:
        return
    try:
        if os.getenv("ENV") != "production":
            load_dotenv()

        supabase_url = os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_KEY_URL")
        supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise RuntimeError("SUPABASE_URL 및 SUPABASE_KEY가 필요합니다")

        _db_client = create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.error("❌ DB 초기화 실패: %s", str(e), exc_info=e)
        raise

def get_db_client() -> Client:
    """DB 미초기화 시 명확한 에러를 발생시키는 정책을 유지합니다."""
    if _db_client is None:
        raise RuntimeError("DB 미초기화: initialize_db() 먼저 호출")
    return _db_client

# -----------------------------------------------------------------------------
# Query helpers (functions originally present in this file)
# -----------------------------------------------------------------------------
async def fetch_human_response(job_id: str) -> Optional[Dict[str, Any]]:
    """
    events 테이블에서 동일 job_id의 'human_response' 단일 레코드 조회
    - 네트워크/일시 오류에 대해 _async_retry 적용
    - 결과가 없으면 None
    """
    if not job_id:
        logger.error("❌ fetch_human_response 잘못된 job_id: %s", str(job_id))
        return None

    def _call():
        client = get_db_client()
        resp = (
            client.table("events")
            .select("*")
            .eq("job_id", job_id)
            .eq("event_type", "human_response")
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None

    return await _async_retry(_call, name="fetch_human_response")


async def save_event(
    *,
    job_id: str,
    todo_id: Optional[str] = None,
    proc_inst_id: Optional[str] = None,
    crew_type: Optional[str] = None,
    data: Dict[str, Any],
    event_type: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """
    events 테이블에 이벤트 저장
    - id: 자동 생성 UUID
    - timestamp: 자동 설정 (now())
    - data: JSONB 형태로 저장
    """
    try:
        event_id = str(uuid.uuid4())
        
        row = {
            "id": event_id,
            "job_id": job_id,
            "todo_id": todo_id,
            "proc_inst_id": proc_inst_id,
            "crew_type": crew_type,
            "data": data,
            "event_type": event_type,
            "status": status,
        }

        def _insert_call() -> str:
            client = get_db_client()
            client.table("events").insert(row).execute()
            return event_id

        result_id = await _async_retry(
            _insert_call,
            name="save_event.insert",
            retries=3,
            base_delay=0.8,
        )
        return result_id

    except Exception as e:
        logger.error("❌ 이벤트저장오류: job_id=%s event_type=%s error=%s", job_id, event_type, str(e), exc_info=e)
        raise


async def save_notification(
    *,
    title: str,
    notif_type: str,
    description: Optional[str] = None,
    user_ids_csv: Optional[str] = None,
    tenant_id: Optional[str] = None,
    url: Optional[str] = None,
    from_user_id: Optional[str] = None,
) -> None:
    """
    notifications 테이블에 알림 저장
    - user_ids_csv: 쉼표로 구분된 사용자 ID 목록. 비어있으면 저장 생략
    - 테이블 스키마 가정: id, user_id, tenant_id, title, description, type, url, from_user_id
    - 정책:
    """
    try:
        # 대상 사용자가 없으면 작업 생략
        if not user_ids_csv:
            logger.info("⏭️ 알림 저장 생략: 대상 사용자 없음 (user_ids_csv=%r)", user_ids_csv)
            return

        # 사용자 ID 파싱/정제
        user_ids: List[str] = [uid.strip() for uid in user_ids_csv.split(",") if uid and uid.strip()]
        if not user_ids:
            logger.info("⏭️ 알림 저장 생략: 유효한 사용자 ID 없음 (user_ids_csv=%r)", user_ids_csv)
            return

        # 행 구성
        rows: List[Dict[str, Any]] = [
            {
                "id": str(uuid.uuid4()),
                "user_id": uid,
                "tenant_id": tenant_id,
                "title": title,
                "description": description,
                "type": notif_type,
                "url": url,
                "from_user_id": from_user_id,
            }
            for uid in user_ids
        ]

        def _insert_call() -> int:
            client = get_db_client()
            client.table("notifications").insert(rows).execute()
            return len(rows)

        inserted = await _async_retry(
            _insert_call,
            name="save_notification.insert",
            retries=3,
            base_delay=0.8,
        )

        if inserted and inserted > 0:
            logger.info("✅ 알림 저장 완료: %d건 (tenant_id=%r, type=%r)", inserted, tenant_id, notif_type)
        else:
            logger.warning(
                "⚠️ 알림 저장 실패 또는 생략: 대상 %d건 (tenant_id=%r, type=%r)",
                len(rows), tenant_id, notif_type
            )

    except Exception as e:
        logger.error("❌ 알림저장오류: %s", str(e), exc_info=e)
        raise


def save_event_sync(
    *,
    job_id: str,
    todo_id: Optional[str] = None,
    proc_inst_id: Optional[str] = None,
    crew_type: Optional[str] = None,
    data: Dict[str, Any],
    event_type: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """
    동기 버전 save_event - 새로운 이벤트 루프에서 실행
    CrewAI 이벤트 리스너에서 사용하기 위한 동기 래퍼
    """
    import threading
    import queue
    
    result_queue = queue.Queue()
    
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                save_event(
                    job_id=job_id,
                    todo_id=todo_id,
                    proc_inst_id=proc_inst_id,
                    crew_type=crew_type,
                    data=data,
                    event_type=event_type,
                    status=status,
                )
            )
            result_queue.put(('success', result))
        except Exception as e:
            result_queue.put(('error', e))
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_async)
    thread.start()
    thread.join(timeout=10)  # 10초 타임아웃
    
    if thread.is_alive():
        logger.error("❌ save_event_sync 타임아웃")
        raise TimeoutError("save_event_sync 타임아웃")
    
    if result_queue.empty():
        logger.error("❌ save_event_sync 결과 없음")
        raise RuntimeError("save_event_sync 결과 없음")
    
    status, result = result_queue.get()
    if status == 'error':
        raise result
    return result

