from __future__ import annotations
import os
import logging
from typing import Optional, List, Type
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from crewai.tools import BaseTool
from dotenv import load_dotenv
from mem0 import Memory
import requests

logger = logging.getLogger(__name__)

# ============================================================================
# 설정 및 초기화
# ============================================================================
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    # 상위로 전파 (필수 환경 누락은 하드 실패)
    raise ValueError("❌ DB 연결 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ============================================================================
# 스키마 정의
# ============================================================================
class KnowledgeQuerySchema(BaseModel):
    query: str = Field(..., description="검색할 지식 쿼리")

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, v):
        if isinstance(v, dict):
            if "description" in v:
                return v["description"]
            elif v:
                return str(list(v.values())[0])
            else:
                return ""
        elif isinstance(v, str):
            return v
        else:
            return str(v)

# ============================================================================
# 지식 검색 도구
# ============================================================================
class Mem0Tool(BaseTool):
    """Supabase 기반 mem0 지식 검색 도구 - 에이전트별"""
    name: str = "mem0"
    description: str = (
        "🧠 에이전트별 개인 지식 저장소 검색 도구\n\n"
        "🚨 필수 검색 순서: 작업 전 반드시 피드백부터 검색!\n\n"
        "저장된 정보:\n"
        "🔴 과거 동일한 작업에 대한 피드백 및 교훈 (최우선 검색 대상)\n"
        "🔴 과거 실패 사례 및 개선 방안\n"
        "• 객관적 정보 (사람명, 수치, 날짜, 사물 등)\n"
        "검색 목적:\n"
        "- 작업지시사항을 올바르게 수행하기 위해 필요한 정보(매개변수, 제약, 의존성)와\n"
        "  안전 수행을 위한 피드백/주의사항을 찾기 위함\n"
        "- 과거 실패 경험을 통한 실수 방지\n"
        "- 정확한 객관적 정보 조회\n\n"
        "사용 지침:\n"
        "- 현재 작업 맥락(사용자 요청, 시스템/도구 출력, 최근 단계)을 근거로 자연어의 완전한 문장으로 질의하세요.\n"
        "- 핵심 키워드 + 엔터티(고객명, 테이블명, 날짜 등) + 제약(환경/범위)을 조합하세요.\n"
        "- 동의어/영문 용어를 섞어 2~3개의 표현으로 재질의하여 누락을 줄이세요.\n"
        "- 필요한 경우 좁은 쿼리 → 넓은 쿼리 순서로 반복 검색하세요. (필요 시 기간/버전 범위 명시)\n"
        "- 동일 정보를 다른 표현으로 재질의하며, 최신/가장 관련 결과를 우선 검토하세요.\n\n"
        "⚡ 핵심: 어떤 작업이든 시작 전에, 해당 작업을 안전하게 수행하기 위한 피드백/주의사항과\n"
        "  필수 매개변수를 먼저 질의하여 확보하세요!"
    )
    args_schema: Type[KnowledgeQuerySchema] = KnowledgeQuerySchema
    _tenant_id: Optional[str] = PrivateAttr()
    _user_id: Optional[str] = PrivateAttr()
    _namespace: Optional[str] = PrivateAttr()
    _memory: Memory = PrivateAttr()

    def __init__(self, tenant_id: str = None, user_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._namespace = user_id
        self._memory = self._initialize_memory()
        logger.info("🧠 Mem0Tool 초기화 완료 | user_id=%s, namespace=%s", self._user_id, self._namespace)

    def _initialize_memory(self) -> Memory:
        """Memory 인스턴스 초기화 - 에이전트별"""
        config = {
            "vector_store": {
                "provider": "supabase",
                "config": {
                    "connection_string": CONNECTION_STRING,
                    "collection_name": "memories",
                    "index_method": "hnsw",
                    "index_measure": "cosine_distance",
                },
            }
        }
        return Memory.from_config(config_dict=config)

    def _run(self, query: str) -> str:
        """지식 검색 및 결과 반환 - 에이전트별 메모리에서"""
        if not query:
            return "검색할 쿼리를 입력해주세요."
        if not self._user_id:
            logger.error("❌ 개인지식 검색 실패 | user_id=%s", self._user_id)
            raise ValueError("mem0 requires user_id")

        try:
            logger.info("🔍 개인지식 검색 시작 | user_id=%s, query=%s", self._user_id, query)
            results = self._memory.search(query, user_id=self._user_id)
            hits = results.get("results", [])

            THRESHOLD = 0.5
            MIN_RESULTS = 5
            hits_sorted = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
            filtered_hits = [h for h in hits_sorted if h.get("score", 0) >= THRESHOLD]
            if len(filtered_hits) < MIN_RESULTS:
                filtered_hits = hits_sorted[:MIN_RESULTS]
            hits = filtered_hits

            logger.info("📊 검색 결과: %d개 (임계값: %.2f)", len(hits), THRESHOLD)
            if not hits:
                return f"'{query}'에 대한 개인 지식이 없습니다."

            return self._format_results(hits)

        except Exception as e:
            # 상세 로깅 후 상위로 전파
            logger.error("❌ 개인지식 검색 실패 | user_id=%s query=%s err=%s", self._user_id, query, str(e), exc_info=True)
            raise

    def _format_results(self, hits: List[dict]) -> str:
        items = []
        for idx, hit in enumerate(hits, start=1):
            memory_text = hit.get("memory", "")
            score = hit.get("score", 0)
            items.append(f"개인지식 {idx} (관련도: {score:.2f})\n{memory_text}")
        return "\n\n".join(items)


# ============================================================================
# 사내 문서 검색 (memento) 도구
# ============================================================================
class MementoQuerySchema(BaseModel):
    query: str = Field(..., description="검색 키워드 또는 질문")

class MementoTool(BaseTool):
    """사내 문서 검색을 수행하는 도구"""
    name: str = "memento"
    description: str = (
        "🔒 보안 민감한 사내 문서 검색 도구\n\n"
        "저장된 정보:\n"
        "• 보안 민감한 사내 기밀 문서\n"
        "• 대용량 사내 문서 및 정책 자료\n"
        "• 객관적이고 정확한 회사 내부 지식\n"
        "• 업무 프로세스, 규정, 기술 문서\n\n"
        "검색 목적:\n"
        "- 작업지시사항을 올바르게 수행하기 위한 회사 정책/규정/프로세스/매뉴얼 확보\n"
        "- 최신 버전의 표준과 가이드라인 확인\n\n"
        "사용 지침:\n"
        "- 현재 작업/요청과 직접 연결된 문맥을 담아 자연어의 완전한 문장으로 질의하세요.\n"
        "- 문서 제목/버전/담당조직/기간/환경(프로덕션·스테이징·모듈 등) 조건을 명확히 포함하세요.\n"
        "- 약어·정식명칭, 한·영 용어를 함께 사용해 2~3회 재질의하며 누락을 줄이세요.\n"
        "- 처음엔 좁게, 필요 시 점진적으로 범위를 넓혀 검색하세요.\n\n"
        "⚠️ 보안 민감 정보 포함 - 적절한 권한과 용도로만 사용"
    )
    args_schema: Type[MementoQuerySchema] = MementoQuerySchema
    _tenant_id: str = PrivateAttr()

    def __init__(self, tenant_id: str = "localhost", **kwargs):
        super().__init__(**kwargs)
        self._tenant_id = tenant_id
        logger.info("🔒 MementoTool 초기화 완료 | tenant_id=%s", self._tenant_id)

    def _run(self, query: str) -> str:
        try:
            logger.info("🔍 사내문서 검색 시작 | tenant_id=%s, query=%s", self._tenant_id, query)
            resp = requests.post(
                "http://memento.process-gpt.io/retrieve",
                json={"query": query, "options": {"tenant_id": self._tenant_id}},
                timeout=40,
            )
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("response", [])
            logger.info("📄 사내문서 검색 결과: %d개", len(docs))
            if not docs:
                return f"테넌트 '{self._tenant_id}'에서 '{query}' 검색 결과가 없습니다."

            results = []
            for doc in docs:
                meta = doc.get("metadata", {}) or {}
                fname = meta.get("file_name", "unknown")
                idx = meta.get("chunk_index", "unknown")
                content = doc.get("page_content", "")
                results.append(f"📄 파일: {fname} (청크 #{idx})\n내용: {content}\n---")

            return f"테넌트 '{self._tenant_id}'에서 '{query}' 검색 결과:\n\n" + "\n".join(results)

        except Exception as e:
            logger.error("❌ 사내문서 검색 실패 | tenant_id=%s query=%s err=%s", self._tenant_id, query, str(e), exc_info=True)
            # 상위로 전파
            raise
