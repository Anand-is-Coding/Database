"""Input/output models for the retrieval stage.

`TeachingContext` is the tutor's current teaching state, not a generic
search query - it's what lets the retriever narrow a search far more than
a bare question string ever could. `RetrievedChunk` is the final artifact
this module produces, one per matched chunk.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TeachingContext(BaseModel):
    """What the AI Teacher currently knows about the lesson in progress.

    Every field except `student_question` is optional - an unknown
    `subject` means "search every collection"; an unknown `chapter` means
    "search the whole subject" (see `Retriever`).
    """

    subject: str | None = None
    class_name: str | None = None
    chapter: str | None = None
    current_topic: str | None = None
    current_page: int | None = None
    board: str | None = None
    language: str | None = None
    student_question: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "The student's raw question. Bounded to a generous but finite "
            "length as basic input-validation hygiene - this model has no "
            "visibility into whatever API/auth layer sits in front of it, "
            "so it shouldn't assume upstream validation already happened."
        ),
    )


class RetrievalConfig(BaseModel):
    """Tunable knobs for `Retriever`.

    `section`/`page_min`/`page_max` are here (not on `TeachingContext`)
    because they're explicit, opt-in search refinements - unlike
    `current_page` (informational teaching state), a page *range* filter
    should never be applied unless a caller deliberately asks for it.
    """

    top_k: int = Field(default=5, gt=0)
    score_threshold: float | None = None
    section: str | None = None
    page_min: int | None = None
    page_max: int | None = None
    search_params: dict[str, Any] | None = None
    max_retries: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=10.0, gt=0)


class RetrievedChunk(BaseModel):
    """A single matched chunk, ranked and ready to hand to a future LLM/tutor stage."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    subject: str | None = None
    class_name: str | None = None
    chapter: str | None = None
    section: str | None = None
    page_number: int | None = None
    source_pdf: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
