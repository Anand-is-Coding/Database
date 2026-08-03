"""Input/output models for the generation stage.

`Answer` is the final artifact of the Q&A flow -- the LLM's natural-language
answer plus the exact chunks it was grounded in, so a caller can show
citations instead of trusting the answer blindly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.settings import settings
from models.retrieval import RetrievedChunk


class GenerationConfig(BaseModel):
    """Tunable knobs for `AnswerGenerator`.

    `model_name` defaults from `Settings.GEMINI_MODEL`, so swapping the
    generation model is a `.env` change, not a code change.
    """

    model_name: str = Field(default_factory=lambda: settings.GEMINI_MODEL)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    max_retries: int = Field(default=3, ge=1)


class Answer(BaseModel):
    """A generated answer, grounded in the chunks it cites."""

    text: str
    question: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    model_name: str
