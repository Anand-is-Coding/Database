"""Unit tests for TextToSpeechService and `parse_voice_map`.

The ElevenLabs SDK client and the audio player are both faked (no real API
key or audio hardware needed) so these tests are fast and hermetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import SecretStr

from config.settings import settings
from voice.exceptions import PlaybackError, SpeechSynthesisError, UnknownVoiceSubjectError
from voice.text_to_speech import TextToSpeechService, parse_voice_map


class _FakeTextToSpeech:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def convert(self, voice_id, *, text, model_id, output_format):
        self.calls.append(
            {"voice_id": voice_id, "text": text, "model_id": model_id, "output_format": output_format}
        )
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return iter(result)


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.text_to_speech = _FakeTextToSpeech(responses)


class _FakePlayer:
    def __init__(self) -> None:
        self.played: list[tuple[np.ndarray, int]] = []

    def play(self, samples, samplerate):
        self.played.append((samples, samplerate))

    def wait(self):
        pass


class _FailingPlayer:
    def play(self, samples, samplerate):
        raise RuntimeError("no audio device")

    def wait(self):
        pass


def test_parse_voice_map_splits_on_semicolon_and_equals():
    raw = "Economics=voice_1;Mathematics=voice_2;Chemistry11,12=voice_3"

    result = parse_voice_map(raw)

    assert result == {"Economics": "voice_1", "Mathematics": "voice_2", "Chemistry11,12": "voice_3"}


def test_parse_voice_map_ignores_blank_and_malformed_pairs():
    raw = "Economics=voice_1;;  ;NoVoiceId=;=OrphanVoice"

    result = parse_voice_map(raw)

    assert result == {"Economics": "voice_1"}


def test_resolve_voice_id_prefers_subject_specific_voice():
    service = TextToSpeechService(voice_map={"Economics": "voice_econ"}, default_voice_id="voice_default")

    assert service.resolve_voice_id("Economics") == "voice_econ"


def test_resolve_voice_id_falls_back_to_default_for_unmapped_subject():
    service = TextToSpeechService(voice_map={"Economics": "voice_econ"}, default_voice_id="voice_default")

    assert service.resolve_voice_id("Mathematics") == "voice_default"


def test_resolve_voice_id_falls_back_to_default_when_subject_is_none():
    service = TextToSpeechService(voice_map={"Economics": "voice_econ"}, default_voice_id="voice_default")

    assert service.resolve_voice_id(None) == "voice_default"


def test_resolve_voice_id_raises_when_no_default_and_no_match():
    service = TextToSpeechService(voice_map={"Economics": "voice_econ"}, default_voice_id="")

    with pytest.raises(UnknownVoiceSubjectError):
        service.resolve_voice_id("Mathematics")


def test_synthesize_uses_resolved_voice_and_returns_audio_bytes():
    client = _FakeClient([[b"chunk1", b"chunk2"]])
    service = TextToSpeechService(voice_map={"Economics": "voice_econ"}, default_voice_id="voice_default", client=client)

    audio = service.synthesize("GDP is total output.", "Economics")

    assert audio == b"chunk1chunk2"
    assert client.text_to_speech.calls[0]["voice_id"] == "voice_econ"
    assert client.text_to_speech.calls[0]["text"] == "GDP is total output."


def test_synthesize_empty_text_raises():
    service = TextToSpeechService(default_voice_id="voice_default", client=_FakeClient([]))

    with pytest.raises(SpeechSynthesisError):
        service.synthesize("   ")


def test_synthesize_transient_error_is_retried_then_succeeds():
    client = _FakeClient([ConnectionError("network blip"), [b"chunk1"]])
    service = TextToSpeechService(default_voice_id="voice_default", max_retries=2, client=client)

    audio = service.synthesize("Recovered.")

    assert audio == b"chunk1"
    assert len(client.text_to_speech.calls) == 2


def test_play_sends_int16_samples_to_player():
    player = _FakePlayer()
    service = TextToSpeechService(default_voice_id="voice_default", player=player)
    pcm_bytes = np.array([1, -1, 100], dtype=np.int16).tobytes()

    service.play(pcm_bytes, sample_rate=24000)

    assert len(player.played) == 1
    samples, samplerate = player.played[0]
    assert samplerate == 24000
    assert list(samples) == [1, -1, 100]


def test_play_wraps_player_failure_in_playback_error():
    service = TextToSpeechService(default_voice_id="voice_default", player=_FailingPlayer())

    with pytest.raises(PlaybackError):
        service.play(np.array([1], dtype=np.int16).tobytes())


def test_speak_synthesizes_then_plays():
    client = _FakeClient([[b"chunk1"]])
    player = _FakePlayer()
    service = TextToSpeechService(default_voice_id="voice_default", client=client, player=player)

    service.speak("Hello.")

    assert len(player.played) == 1


def test_missing_api_key_raises_on_client_access(monkeypatch):
    monkeypatch.setattr(settings, "ELEVENLABS_API_KEY", SecretStr(""))
    service = TextToSpeechService(default_voice_id="voice_default")

    with pytest.raises(SpeechSynthesisError):
        service.synthesize("Hello.")
