"""Unit tests for AnswerGenerator.

The Gemini SDK client is faked (no real API key/network needed) so these
tests are fast and hermetic, matching the rest of the suite.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from config.settings import settings
from generation.exceptions import AnswerGenerationError, EmptyContextError
from generation.gemini_generator import AnswerGenerator
from models.generation import GenerationConfig
from models.retrieval import RetrievedChunk


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.models = _FakeModels(responses)


def _chunk(text="GDP is the total value of goods and services produced.", subject="Economics", chapter="Ch1", page=12):
    return RetrievedChunk(
        chunk_id="c1", document_id="d1", text=text, score=0.9, subject=subject, chapter=chapter, page_number=page
    )


def test_generate_returns_answer_grounded_in_chunks():
    client = _FakeClient(["GDP measures total economic output."])
    generator = AnswerGenerator(client=client)
    chunks = [_chunk()]

    answer = generator.generate("What is GDP?", chunks)

    assert answer.text == "GDP measures total economic output."
    assert answer.question == "What is GDP?"
    assert answer.sources == chunks
    assert client.models.calls[0]["model"] == generator.config.model_name


def test_prompt_includes_chunk_text_and_location():
    prompt = AnswerGenerator.build_prompt("What is GDP?", [_chunk()])

    assert "GDP is the total value" in prompt
    assert "Economics / Ch1 / page 12" in prompt
    assert "Student's question: What is GDP?" in prompt


def test_empty_chunks_raises_empty_context_error():
    generator = AnswerGenerator(client=_FakeClient([]))

    with pytest.raises(EmptyContextError):
        generator.generate("What is GDP?", [])


def test_empty_question_raises_answer_generation_error():
    generator = AnswerGenerator(client=_FakeClient([]))

    with pytest.raises(AnswerGenerationError):
        generator.generate("   ", [_chunk()])


def test_transient_error_is_retried_then_succeeds():
    client = _FakeClient([ConnectionError("network blip"), "Recovered answer."])
    generator = AnswerGenerator(config=GenerationConfig(max_retries=2), client=client)

    answer = generator.generate("What is GDP?", [_chunk()])

    assert answer.text == "Recovered answer."
    assert len(client.models.calls) == 2


def test_non_transient_error_raises_without_exhausting_retries():
    client = _FakeClient([ValueError("invalid request")])
    generator = AnswerGenerator(config=GenerationConfig(max_retries=3), client=client)

    with pytest.raises(AnswerGenerationError):
        generator.generate("What is GDP?", [_chunk()])

    assert len(client.models.calls) == 1


def test_empty_response_text_raises_answer_generation_error():
    client = _FakeClient([""])
    generator = AnswerGenerator(config=GenerationConfig(max_retries=1), client=client)

    with pytest.raises(AnswerGenerationError):
        generator.generate("What is GDP?", [_chunk()])


def test_missing_api_key_raises_on_client_access(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr(""))
    generator = AnswerGenerator()

    with pytest.raises(AnswerGenerationError):
        generator.generate("What is GDP?", [_chunk()])
