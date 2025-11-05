from __future__ import annotations

import os
import asyncio
import logging
import random
from typing import Any, Dict, Optional, Callable, TypeVar, List
import time
import uuid
from dotenv import load_dotenv
from supabase import Client, create_client

T = TypeVar("T")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Retry utility (sync, single source of truth)
# -----------------------------------------------------------------------------
def _retry_sync(
    fn: Callable[[], T],
    *,
    name: str,
    retries: int = 3,
    base_delay: float = 0.8,
    fallback: Optional[Callable[[], T]] = None,
) -> T:
    """
    동기 재시도 유틸리티.
    - 각 실패 시 지수 백오프 + 지터 적용 후 재시도
    - 최종 실패 시 예외 전파
    - fallback 제공 시 마지막에 실행
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            jitter = random.uniform(0, 0.3)
            delay = base_delay * (2 ** (attempt - 1)) + jitter
            logger.warning(
                "⏳ 재시도 지연: name=%s attempt=%d/%d delay=%.2fs error=%s",
                name, attempt, retries, delay, str(e)
            )
            time.sleep(delay)

    if last_err is not None:
        logger.error(
            "❌ 재시도 최종 실패: name=%s retries=%s error=%s",
            name, retries, str(last_err), exc_info=last_err
        )
        if fallback is not None:
            try:
                return fallback()
            except Exception as fb_err:
                logger.error("❌ fallback 실패: name=%s error=%s", name, str(fb_err), exc_info=fb_err)
                raise fb_err
        raise last_err

    raise RuntimeError(f"Unexpected state in _retry_sync: name={name}")

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
    """비동기 호환 래퍼: 동기 구현을 스레드에서 실행"""
    return await asyncio.to_thread(fetch_human_response_sync, job_id=job_id)


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
    """비동기 호환 래퍼: 동기 구현을 스레드에서 실행"""
    return await asyncio.to_thread(
        save_event_sync,
        job_id=job_id,
        todo_id=todo_id,
        proc_inst_id=proc_inst_id,
        crew_type=crew_type,
        data=data,
        event_type=event_type,
        status=status,
    )


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
    """비동기 호환 래퍼: 동기 구현을 스레드에서 실행"""
    return await asyncio.to_thread(
        save_notification_sync,
        title=title,
        notif_type=notif_type,
        description=description,
        user_ids_csv=user_ids_csv,
        tenant_id=tenant_id,
        url=url,
        from_user_id=from_user_id,
    )


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
    동기 저장 구현(단일 소스). Supabase SDK는 동기이므로 직접 호출 + 동기 재시도.
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

        return _retry_sync(_insert_call, name="save_event.insert", retries=3, base_delay=0.8)

    except Exception as e:
        logger.error("❌ 이벤트저장오류: job_id=%s event_type=%s error=%s", job_id, event_type, str(e), exc_info=e)
        raise


def fetch_human_response_sync(*, job_id: str) -> Optional[Dict[str, Any]]:
    """
    동기 버전 fetch_human_response - 새로운 이벤트 루프에서 실행
    human_query_tool 폴링 루틴에서 사용하기 위한 동기 래퍼
    """
    if not job_id:
        logger.error("❌ fetch_human_response 잘못된 job_id: %s", str(job_id))
        return None

    def _call() -> Optional[Dict[str, Any]]:
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

    return _retry_sync(_call, name="fetch_human_response")


def save_notification_sync(
    *,
    title: str,
    notif_type: str,
    description: Optional[str] = None,
    user_ids_csv: Optional[str] = None,
    tenant_id: Optional[str] = None,
    url: Optional[str] = None,
    from_user_id: Optional[str] = None,
) -> None:
    try:
        if not user_ids_csv:
            logger.info("⏭️ 알림 저장 생략: 대상 사용자 없음 (user_ids_csv=%r)", user_ids_csv)
            return

        user_ids: List[str] = [uid.strip() for uid in user_ids_csv.split(",") if uid and uid.strip()]
        if not user_ids:
            logger.info("⏭️ 알림 저장 생략: 유효한 사용자 ID 없음 (user_ids_csv=%r)", user_ids_csv)
            return

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

        inserted = _retry_sync(_insert_call, name="save_notification.insert", retries=3, base_delay=0.8)

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


def fetch_tenant_mcp(tenant_id: str) -> Dict[str, Any]:
    """tenants 테이블에서 MCP 설정을 조회."""
    client = get_db_client()
    resp = client.table("tenants").select("*").eq("id", tenant_id).single().execute()
    return (resp.data or {}).get("mcp") if resp and resp.data else {}


def fetch_events_by_todo_id(todo_id: str) -> List[Dict[str, Any]]:
    client = get_db_client()
    resp = (
        client.table("events").select("*").eq("todo_id", todo_id).order("timestamp", desc=False).execute()
    )
    return resp.data or []


def fetch_workitem_by_id(id: str) -> Optional[Dict[str, Any]]:
    client = get_db_client()
    resp = client.table("todolist").select("*").eq("id", id).single().execute()
    return resp.data if resp and resp.data else None


def fetch_mcp_python_code(proc_def_id: str, activity_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    client = get_db_client()
    resp = (
        client.table("mcp_python_code")
        .select("*")
        .eq("proc_def_id", proc_def_id)
        .eq("activity_id", activity_id)
        .eq("tenant_id", tenant_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def upsert_mcp_python_code(*, code: str, parameters: dict, proc_def_id: str, activity_id: str, tenant_id: str) -> str:
    client = get_db_client()
    existing = fetch_mcp_python_code(proc_def_id, activity_id, tenant_id)
    data: Dict[str, Any] = {
        "code": code,
        "parameters": parameters,
        "proc_def_id": proc_def_id,
        "activity_id": activity_id,
        "tenant_id": tenant_id,
    }
    if existing:
        data["id"] = existing["id"]
    resp = client.table("mcp_python_code").upsert(data).execute()
    rows = resp.data or []
    if rows:
        return rows[0]["id"]
    raise RuntimeError("Failed to upsert mcp_python_code")

def fetch_form_by_id(id: str) -> Optional[Dict[str, Any]]:
    client = get_db_client()
    resp = client.table("form_def").select("*").eq("id", id).single().execute()
    return resp.data if resp and resp.data else None

