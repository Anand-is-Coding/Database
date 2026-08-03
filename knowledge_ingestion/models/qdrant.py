"""Configuration and small result models for the Qdrant vector-store stage."""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.settings import settings


class QdrantConfig(BaseModel):
    """Tunable knobs for `VectorStore`.

    `url`/`api_key` default from `Settings`, so pointing at a different
    Qdrant instance (or adding an API key) is a `.env` change, not a code
    change. `fallback_collection` is only used for the rare chunk with no
    inferred `subject` — normal operation never touches it, since the
    collection name is otherwise always the chunk's own subject.
    """

    url: str = Field(default_factory=lambda: settings.QDRANT_URL)
    api_key: str | None = Field(
        default_factory=lambda: settings.QDRANT_API_KEY.get_secret_value() or None
    )
    fallback_collection: str | None = Field(default_factory=lambda: settings.QDRANT_COLLECTION or None)
    batch_size: int = Field(default=100, gt=0)
    max_retries: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    vector_size: int | None = None
    indexed_payload_fields: list[str] = Field(
        default_factory=lambda: ["document_id", "class", "chapter"]
    )


class UploadResult(BaseModel):
    """Summary of one `upsert_vectors` call, possibly spanning collections."""

    collection_name: str
    requested: int
    uploaded: int
    skipped_invalid: int = 0
    batches: int = 0

    @property
    def success(self) -> bool:
        return self.uploaded == self.requested


class CollectionStats(BaseModel):
    """A snapshot of one collection's size/health, from `collection_stats()`."""

    name: str
    points_count: int
    indexed_vectors_count: int | None = None
    segments_count: int
    status: str
