"""Generation module -- turns retrieved chunks and a student's question into
a grounded natural-language answer using Gemini.

Single responsibility: `list[RetrievedChunk]` + question -> `Answer`. This
module knows nothing about Qdrant, embeddings, or retrieval -- it only
builds a grounding prompt from whatever chunks it's handed and asks an LLM
to answer strictly from them.
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from config.logging import get_logger
from config.settings import settings
from generation.exceptions import AnswerGenerationError, EmptyContextError
from models.generation import Answer, GenerationConfig
from models.retrieval import RetrievedChunk
from utils.retry import retry_with_backoff

logger = get_logger(__name__)

_TRANSIENT_KEYWORDS = (
    "timeout",
    "connection",
    "network",
    "temporarily unavailable",
    "unavailable",
    "resource_exhausted",
    "429",
    "500",
    "502",
    "503",
    "504",
)

_SYSTEM_INSTRUCTION = (
    "You are a patient AI tutor for school students. Answer the student's "
    "question using ONLY the textbook excerpts provided as context. If the "
    "excerpts don't contain enough information to answer, say so plainly "
    "instead of guessing or using outside knowledge. Keep answers clear and "
    "age-appropriate."
)


def _is_transient_generation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in _TRANSIENT_KEYWORDS)


class AnswerGenerator:
    """Generates a grounded answer from retrieved chunks using Gemini.

    The `genai.Client` is created lazily (mirrors `EmbeddingService`'s lazy
    model load) and can be injected via the `client` constructor argument so
    tests never need a real API key or network access.
    """

    def __init__(self, config: GenerationConfig | None = None, client: Any | None = None) -> None:
        self.config = config or GenerationConfig()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            api_key = settings.GEMINI_API_KEY.get_secret_value()
            if not api_key:
                raise AnswerGenerationError(
                    "GEMINI_API_KEY is not set. Add it to your .env file to use the Q&A generator."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        """The main flow: build a grounding prompt from `chunks` and ask
        Gemini to answer `question` strictly from it.
        """
        if not question or not question.strip():
            raise AnswerGenerationError("question must not be empty.")
        if not chunks:
            raise EmptyContextError("No chunks were retrieved to ground an answer in.")

        prompt = self.build_prompt(question, chunks)
        text = self._generate_with_retry(prompt)

        logger.info("Answer generated: %d char(s), grounded in %d chunk(s).", len(text), len(chunks))
        return Answer(
            text=text.strip(),
            question=question,
            sources=chunks,
            model_name=self.config.model_name,
        )

    @staticmethod
    def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        excerpts = []
        for i, chunk in enumerate(chunks, start=1):
            location = " / ".join(
                part
                for part in (
                    chunk.subject,
                    chunk.chapter,
                    f"page {chunk.page_number}" if chunk.page_number else None,
                )
                if part
            )
            header = f"[Excerpt {i}" + (f" - {location}]" if location else "]")
            excerpts.append(f"{header}\n{chunk.text}")

        context = "\n\n".join(excerpts)
        return f"Context:\n{context}\n\nStudent's question: {question}"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_with_retry(self, prompt: str) -> str:
        def _do_generate() -> str:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                ),
            )
            text = getattr(response, "text", None)
            if not text:
                raise AnswerGenerationError("Gemini returned an empty response.")
            return text

        try:
            return retry_with_backoff(
                _do_generate,
                should_retry=_is_transient_generation_error,
                max_attempts=self.config.max_retries,
            )
        except AnswerGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend raises various types
            raise AnswerGenerationError(f"Failed to generate an answer: {exc}") from exc
