"""Configuration and output models for the embedding stage.

`EmbeddedChunk` is the final artifact the ingestion side of this repository
produces per chunk — vector plus a full Qdrant-ready payload — consumed by
the vector-store stage (`vectorstore/qdrant_client.py`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from config.settings import settings


class EmbeddingConfig(BaseModel):
    """Tunable knobs for `EmbeddingService`.

    `model_name` defaults from `Settings.EMBEDDING_MODEL`, so swapping the
    embedding model is a `.env` change, not a code change.
    """

    model_name: str = Field(default_factory=lambda: settings.EMBEDDING_MODEL)
    batch_size: int = Field(default=32, gt=0)
    normalize_embeddings: bool = True
    device: str | None = None
    max_retries: int = Field(default=3, ge=1)
    encode_timeout_seconds: float | None = Field(default=None, gt=0)


class EmbeddedChunk(BaseModel):
    """A chunk's text plus its embedding vector and full payload for Qdrant."""

    vector: list[float]
    chunk_id: str
    document_id: str
    subject: str | None = None
    class_name: str | None = None
    chapter: str | None = None
    section: str | None = None
    page_number: int | None = None
    source_pdf: str
    chunk_number: int
    total_chunks: int
    token_count: int
    character_count: int
    original_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str
    embedding_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("vector")
    @classmethod
    def _vector_non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("vector must not be empty")
        return v
