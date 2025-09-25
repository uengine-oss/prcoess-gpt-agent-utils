import os
import base64
import logging
import traceback
from typing import Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import uuid

from openai import OpenAI
from supabase import create_client, Client

# ============================================================================
# 설정
# ============================================================================
load_dotenv()
logger = logging.getLogger(__name__)

# 고정/환경 기본값
DEFAULT_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")
DEFAULT_SIZE = os.getenv("IMAGE_DEFAULT_SIZE", "1024x1024")
DEFAULT_QUALITY = os.getenv("IMAGE_DEFAULT_QUALITY", "medium")
BUCKET_NAME = os.getenv("IMAGE_BUCKET", "task-image")

# ============================================================================
# 스키마
# ============================================================================
class ImageGenSchema(BaseModel):
    prompt: str = Field(..., description="생성할 이미지 설명")
    filename: Optional[str] = Field(None, description="저장 파일명(.png 권장). 없으면 자동 생성")
    size: str = Field(
        DEFAULT_SIZE,
        description="이미지 크기 (예: 1024x1024 | 1536x1024 | 1024x1536)"
    )
    quality: str = Field(
        DEFAULT_QUALITY,
        description="이미지 품질 (low | medium | high)"
    )

# ============================================================================
# Tool
# ============================================================================
class ImageGenTool(BaseTool):
    """🎨 GPT-Image 기반 이미지 생성 + Supabase Storage 업로드 (폴백 없음, 실패 시 예외 전파)"""
    name: str = "image_gen"
    description: str = (
        "이미지 생성 지시가 있거나, 보고서 및 슬라이드 생성시 내용에 어울리는 이미지를 생성해줍니다.\n"
        "OpenAI gpt-image-1로 이미지를 생성해 Supabase Storage에 업로드하고 공개 URL을 반환합니다.\n"
        "필수: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY\n"
        "반환값: Supabase Storage 공개 URL(문자열)"
    )
    args_schema: Type[ImageGenSchema] = ImageGenSchema

    _client: OpenAI = PrivateAttr()
    _supabase: Client = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)

        # ── OpenAI 클라이언트 ─────────────────────────────────────────────
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        base_url = os.getenv("OPENAI_BASE_URL")  # 없으면 SDK 기본값 사용
        self._client = OpenAI(api_key=api_key, base_url=base_url)

        # ── Supabase 클라이언트 ───────────────────────────────────────────
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_KEY", "")
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL 또는 SUPABASE_KEY 환경 변수가 설정되지 않았습니다.")
        try:
            self._supabase = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase 클라이언트 초기화 완료")
        except Exception as e:
            logger.exception("❌ Supabase 클라이언트 초기화 실패")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # 내부 유틸: 업로드 (실패 시 예외)
    # ─────────────────────────────────────────────────────────────────────
    def _upload_to_supabase(self, image_data: bytes, filename: str) -> str:
        try:
            # 선택적 리사이즈(512x512). 실패해도 원본 업로드는 계속 진행.
            try:
                from PIL import Image
                from io import BytesIO

                img = Image.open(BytesIO(image_data))
                img_resized = img.resize((512, 512), Image.LANCZOS)

                out = BytesIO()
                img_resized.save(out, format="PNG", optimize=True)
                image_data = out.getvalue()
                logger.info("🖼️ 이미지 리사이즈 완료: %s → 512x512", getattr(img, "size", None))
            except ImportError:
                logger.warning("Pillow 미설치: 원본 크기로 업로드합니다.")
            except Exception as re:
                logger.warning("이미지 리사이즈 실패(원본 업로드로 계속): %s", re)

            # 업로드 (오류 시 예외)
            res = self._supabase.storage.from_(BUCKET_NAME).upload(filename, image_data)
            # supabase-py는 성공 시 dict/Response 객체를 반환(버전별 상이); 실패 시 예외 또는 오류 응답
            # 오류 응답을 반환하는 경우도 있으니 간단 검증
            if res is None:
                raise RuntimeError("Supabase Storage 업로드 실패(res=None)")

            # 공개 URL
            public_url = self._supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
            if not public_url:
                raise RuntimeError("Supabase Storage 공개 URL 생성 실패")
            logger.info("✅ 업로드 완료: %s", public_url)
            return public_url

        except Exception:
            logger.exception("❌ Supabase Storage 업로드 중 오류")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # BaseTool 인터페이스: 실행 (실패 시 예외, 폴백 없음)
    # ─────────────────────────────────────────────────────────────────────
    def _run(
        self,
        prompt: str,
        filename: Optional[str] = None,
        size: str = DEFAULT_SIZE,
        quality: str = DEFAULT_QUALITY
    ) -> str:
        # 입력 검증
        if not prompt or not prompt.strip():
            raise ValueError("prompt가 비어 있습니다.")

        # 파일명 자동 생성
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique = uuid.uuid4().hex[:8]
            filename = f"generated_image_{timestamp}_{unique}.png"
        # 확장자 보정
        if not filename.lower().endswith(".png"):
            filename = f"{Path(filename).stem}.png"

        logger.info("[image_gen] size=%s, quality=%s, file=%s", size, quality, filename)

        try:
            # 이미지 생성 (오류 시 예외)
            resp = self._client.images.generate(
                model=DEFAULT_IMAGE_MODEL,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
                response_format="b64_json",  # 명시
            )

            if not getattr(resp, "data", None):
                raise RuntimeError("이미지 생성 응답이 비어 있습니다(data 없음).")
            b64 = resp.data[0].b64_json
            if not b64:
                raise RuntimeError("이미지 생성 응답에 b64_json이 없습니다.")

            image_bytes = base64.b64decode(b64)

            # 업로드 (오류 시 예외)
            public_url = self._upload_to_supabase(image_bytes, filename)

            # 반환: 공개 URL 문자열(마크다운 감싸지 않음)
            return str(public_url)

        except Exception:
            logger.exception("❌ [image_gen] 처리 중 오류")
            raise
