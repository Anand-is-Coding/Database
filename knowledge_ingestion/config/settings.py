"""Centralized application configuration loaded from environment variables / .env."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for all runtime configuration.

    Values are loaded from a `.env` file (see `.env.example`) and/or the
    process environment. Environment variables always take precedence over
    values defined in `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # AWS S3
    AWS_ACCESS_KEY_ID: str = Field(default="", description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: SecretStr = Field(default=SecretStr(""), description="AWS secret access key")
    AWS_REGION: str = Field(default="us-east-1", description="AWS region for the S3 bucket")
    S3_BUCKET_NAME: str = Field(default="", description="S3 bucket containing source PDFs")

    # Qdrant
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant endpoint URL")
    QDRANT_API_KEY: SecretStr = Field(default=SecretStr(""), description="Qdrant API key")
    QDRANT_COLLECTION: str = Field(
        default="yotutor_knowledge_base", description="Qdrant collection name"
    )

    # Embedding
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-m3", description="HuggingFace embedding model identifier"
    )

    # Parsing
    DOCLING_DO_OCR: bool = Field(
        default=True,
        description=(
            "Run OCR on every PDF page. OCR is one of the heaviest parts of "
            "Docling's pipeline (memory and time); set to false for digitally "
            "generated PDFs with a real text layer (most textbooks) - only "
            "scanned/image-only PDFs actually need it."
        ),
    )
    DOCLING_DO_TABLE_STRUCTURE: bool = Field(
        default=True,
        description="Detect and extract table structure. Leave true unless memory-constrained and tables aren't needed.",
    )
    DOCLING_NUM_THREADS: int = Field(
        default=2,
        description="Number of threads allocated for Docling's PyTorch models. Lower this to 1 or 2 to save memory."
    )
    DOCLING_BATCH_SIZE: int = Field(
        default=2,
        description="Batch size for layout and table extraction models. Lower this (e.g. to 1) to avoid std::bad_alloc."
    )
    DOCLING_IMAGE_SCALE: float = Field(
        default=1.0,
        description=(
            "Scale for rendered page images, passed straight to Docling's "
            "PdfPipelineOptions.images_scale (Docling's own built-in default "
            "is 1.0 - keep this at 1.0 unless memory-constrained). Memory "
            "scales roughly quadratically with this value (it scales both "
            "width and height), so raising it is expensive and lowering it "
            "below 1.0 (e.g. 0.75 or 0.5) is one of the most effective single "
            "levers against std::bad_alloc crashes on low-RAM machines, at "
            "the cost of coarser layout/table detection on dense pages."
        ),
    )

    # Generation (Q&A)
    GEMINI_API_KEY: SecretStr = Field(
        default=SecretStr(""), description="Google Gemini API key, used to generate answers from retrieved chunks"
    )
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini model used for answer generation")

    # Voice (ElevenLabs speech-to-text + text-to-speech)
    ELEVENLABS_API_KEY: SecretStr = Field(
        default=SecretStr(""), description="ElevenLabs API key, used for speech-to-text and text-to-speech"
    )
    ELEVENLABS_DEFAULT_VOICE_ID: str = Field(
        default="",
        description="Fallback ElevenLabs voice_id used when no subject is selected or no per-subject voice is configured",
    )
    ELEVENLABS_VOICE_MAP: str = Field(
        default="",
        description=(
            "Per-subject ElevenLabs voice_id mapping, e.g. "
            "'Economics=voice_id_1;Mathematics=voice_id_2'. Subject names must match "
            "Qdrant collection names exactly. Parsed by voice/text_to_speech.py."
        ),
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Root log level")


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide `Settings` instance."""
    return Settings()


settings = get_settings()
