from __future__ import annotations
from contextvars import ContextVar
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

# ---- 필수 컨텍스트: 6개만 관리 ----
proc_inst_id_var: ContextVar[Optional[str]] = ContextVar("proc_inst_id", default=None)
task_id_var: ContextVar[Optional[str]] = ContextVar("task_id", default=None)
crew_type_var: ContextVar[Optional[str]] = ContextVar("crew_type", default=None)
users_email_var: ContextVar[Optional[str]] = ContextVar("users_email", default=None)
form_def_id_var: ContextVar[Optional[str]] = ContextVar("form_def_id", default=None)
form_key_var: ContextVar[Optional[str]] = ContextVar("form_key", default=None)

def set_context(
    *,
    proc_inst_id: Optional[str] = None,
    task_id: Optional[str] = None,
    crew_type: Optional[str] = None,
    users_email: Optional[str] = None,
    form_def_id: Optional[str] = None,
    form_key: Optional[str] = None,
) -> None:
    if proc_inst_id is not None:
        proc_inst_id_var.set(proc_inst_id)
        logger.info("🔧 컨텍스트 설정: proc_inst_id=%s", proc_inst_id)
    if task_id is not None:
        task_id_var.set(task_id)
        logger.info("📋 컨텍스트 설정: task_id=%s", task_id)
    if crew_type is not None:
        crew_type_var.set(crew_type)
        logger.info("👥 컨텍스트 설정: crew_type=%s", crew_type)
    if users_email is not None:
        users_email_var.set(users_email)
        logger.info("📧 컨텍스트 설정: users_email=%s", users_email)
    if form_def_id is not None:
        form_def_id_var.set(form_def_id)
        logger.info("📝 컨텍스트 설정: form_def_id=%s", form_def_id)
    if form_key is not None:
        form_key_var.set(form_key)
        logger.info("🔑 컨텍스트 설정: form_key=%s", form_key)

def reset_context() -> None:
    proc_inst_id_var.set(None)
    task_id_var.set(None)
    crew_type_var.set(None)
    users_email_var.set(None)
    form_def_id_var.set(None)
    form_key_var.set(None)
    logger.info("🔄 컨텍스트 리셋 완료")

def get_context_snapshot() -> Dict[str, Optional[str]]:
    return dict(
        proc_inst_id=proc_inst_id_var.get(),
        task_id=task_id_var.get(),
        crew_type=crew_type_var.get(),
        users_email=users_email_var.get(),
        form_def_id=form_def_id_var.get(),
        form_key=form_key_var.get(),
    )
