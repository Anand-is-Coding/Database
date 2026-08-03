"""Unit tests for SpeechToTextService.

The ElevenLabs SDK client is faked (no real API key/network needed) so
these tests are fast and hermetic, matching the rest of the suite.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from config.settings import settings
from voice.exceptions import EmptyTranscriptError, TranscriptionError
from voice.speech_to_text import SpeechToTextService


class _FakeTranscriptResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeSpeechToText:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def convert(self, *, model_id, file):
        self.calls.append({"model_id": model_id, "file": file})
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return _FakeTranscriptResponse(result)


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.speech_to_text = _FakeSpeechToText(responses)


def test_transcribe_returns_stripped_text():
    client = _FakeClient(["  What is GDP?  "])
    service = SpeechToTextService(client=client)

    text = service.transcribe(b"fake-wav-bytes")

    assert text == "What is GDP?"
    assert client.speech_to_text.calls[0]["file"] == b"fake-wav-bytes"


def test_empty_audio_raises_transcription_error():
    service = SpeechToTextService(client=_FakeClient([]))

    with pytest.raises(TranscriptionError):
        service.transcribe(b"")


def test_silent_audio_raises_empty_transcript_error():
    client = _FakeClient([""])
    service = SpeechToTextService(client=client)

    with pytest.raises(EmptyTranscriptError):
        service.transcribe(b"fake-wav-bytes")


def test_transient_error_is_retried_then_succeeds():
    client = _FakeClient([ConnectionError("network blip"), "Recovered transcript."])
    service = SpeechToTextService(max_retries=2, client=client)

    text = service.transcribe(b"fake-wav-bytes")

    assert text == "Recovered transcript."
    assert len(client.speech_to_text.calls) == 2


def test_non_transient_error_raises_without_exhausting_retries():
    client = _FakeClient([ValueError("invalid audio format")])
    service = SpeechToTextService(max_retries=3, client=client)

    with pytest.raises(TranscriptionError):
        service.transcribe(b"fake-wav-bytes")

    assert len(client.speech_to_text.calls) == 1


def test_missing_api_key_raises_on_client_access(monkeypatch):
    monkeypatch.setattr(settings, "ELEVENLABS_API_KEY", SecretStr(""))
    service = SpeechToTextService()

    with pytest.raises(TranscriptionError):
        service.transcribe(b"fake-wav-bytes")
