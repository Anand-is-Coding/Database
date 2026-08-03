"""Speech-to-text module -- transcribes recorded audio into text using
ElevenLabs Scribe.

Single responsibility: WAV bytes -> str. Knows nothing about the
microphone, retrieval, or generation.
"""

from __future__ import annotations

from typing import Any

from elevenlabs.client import ElevenLabs

from config.logging import get_logger
from config.settings import settings
from utils.retry import retry_with_backoff
from voice.exceptions import EmptyTranscriptError, TranscriptionError

logger = get_logger(__name__)

_TRANSIENT_KEYWORDS = (
    "timeout",
    "connection",
    "network",
    "temporarily unavailable",
    "unavailable",
    "429",
    "500",
    "502",
    "503",
    "504",
)


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in _TRANSIENT_KEYWORDS)


class SpeechToTextService:
    """Transcribes audio into text using ElevenLabs' speech-to-text API.

    The `ElevenLabs` client is created lazily and can be injected via the
    `client` constructor argument so tests never need a real API key.
    """

    def __init__(self, model_id: str = "scribe_v1", max_retries: int = 3, client: Any | None = None) -> None:
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            api_key = settings.ELEVENLABS_API_KEY.get_secret_value()
            if not api_key:
                raise TranscriptionError(
                    "ELEVENLABS_API_KEY is not set. Add it to your .env file to use voice input."
                )
            self._client = ElevenLabs(api_key=api_key)
        return self._client

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe a WAV file's bytes into text. Raises `EmptyTranscriptError`
        if the audio contained no recognizable speech (e.g. silence).
        """
        if not audio_bytes:
            raise TranscriptionError("No audio bytes to transcribe.")

        def _do_transcribe() -> str:
            response = self.client.speech_to_text.convert(model_id=self.model_id, file=audio_bytes)
            return getattr(response, "text", "") or ""

        try:
            text = retry_with_backoff(
                _do_transcribe, should_retry=_is_transient_error, max_attempts=self.max_retries
            )
        except TranscriptionError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend raises various types
            raise TranscriptionError(f"Failed to transcribe audio: {exc}") from exc

        text = text.strip()
        if not text:
            raise EmptyTranscriptError("Transcription produced no text (silence or inaudible audio?).")

        logger.info("Transcript received: %r", text)
        return text
