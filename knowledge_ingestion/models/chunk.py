"""Configuration and output models for the chunking stage.

`Chunk`/`ChunkMetadata` are what downstream stages (embedding, Qdrant)
consume — `ChunkMetadata` is designed to become a Qdrant payload as-is, so
every field here is preserved output, not scratch state.
"""

from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ContentKind(str, Enum):
    """What kind of educational content a chunk represents.

    Drives which chunks are treated as atomic (never split) by
    `LlamaChunker` and is preserved in `ChunkMetadata` for later use by
    retrieval (e.g. boosting definitions/theorems for a tutor's answer).
    """

    TEXT = "text"
    DEFINITION = "definition"
    EXAMPLE = "example"
    THEOREM = "theorem"
    NOTE = "note"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"


class ChunkingConfig(BaseModel):
    """Tunable knobs for `LlamaChunker`. Size fields are measured in tokens."""

    target_chunk_size_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    min_chunk_size_tokens: int = Field(default=20, ge=0)
    include_heading_context: bool = True

    @model_validator(mode="after")
    def _overlap_smaller_than_target(self) -> "ChunkingConfig":
        if self.chunk_overlap_tokens >= self.target_chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than target_chunk_size_tokens"
            )
        return self


class ChunkMetadata(BaseModel):
    """Traces a chunk back to its source document and position within it."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    subject: str | None = None
    class_name: str | None = None
    chapter: str | None = None
    section: str | None = None
    page_number: int | None = None
    page_numbers: list[int] = Field(default_factory=list)
    source_pdf: str
    parser_version: str
    chunk_number: int
    total_chunks: int = 0
    character_count: int
    estimated_token_count: int
    content_kind: ContentKind = ContentKind.TEXT
    image_references: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A single semantically meaningful unit of text, ready for embedding."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
