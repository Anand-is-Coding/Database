"""Text-to-speech module -- synthesizes an answer into subject-specific
voice audio using ElevenLabs, and plays it back.

Single responsibility: (text, subject) -> spoken audio. Knows nothing
about retrieval or generation -- it only turns a string into sound.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs

from config.logging import get_logger
from config.settings import settings
from utils.retry import retry_with_backoff
from voice.exceptions import PlaybackError, SpeechSynthesisError, UnknownVoiceSubjectError

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

# Raw PCM (no container/codec) so playback never needs an MP3/Opus decoder.
_OUTPUT_FORMAT = "pcm_24000"
_OUTPUT_SAMPLE_RATE = 24000


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in _TRANSIENT_KEYWORDS)


def parse_voice_map(raw: str) -> dict[str, str]:
    """Parse `Settings.ELEVENLABS_VOICE_MAP` into a `{subject: voice_id}` dict.

    Format: `"Subject=voice_id;Subject Two=voice_id_2"`. `;` separates
    pairs and `=` separates the subject from its voice_id - not `,`, since
    real Qdrant collection names here already contain commas (e.g.
    `"Chemistry11,12"`).
    """
    voice_map: dict[str, str] = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        subject, _, voice_id = pair.partition("=")
        subject, voice_id = subject.strip(), voice_id.strip()
        if subject and voice_id:
            voice_map[subject] = voice_id
    return voice_map


class TextToSpeechService:
    """Synthesizes text into speech using a subject-specific ElevenLabs voice.

    The `ElevenLabs` client is created lazily and can be injected via the
    `client` constructor argument; playback goes through `sounddevice`,
    injectable via `player` - both so tests never need a real API key or
    audio hardware.
    """

    def __init__(
        self,
        voice_map: dict[str, str] | None = None,
        default_voice_id: str | None = None,
        model_id: str = "eleven_multilingual_v2",
        max_retries: int = 3,
        client: Any | None = None,
        player: Any = sd,
    ) -> None:
        self.voice_map = voice_map if voice_map is not None else parse_voice_map(settings.ELEVENLABS_VOICE_MAP)
        self.default_voice_id = (
            default_voice_id if default_voice_id is not None else settings.ELEVENLABS_DEFAULT_VOICE_ID
        )
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = client
        self._player = player

    @property
    def client(self) -> Any:
        if self._client is None:
            api_key = settings.ELEVENLABS_API_KEY.get_secret_value()
            if not api_key:
                raise SpeechSynthesisError(
                    "ELEVENLABS_API_KEY is not set. Add it to your .env file to use voice output."
                )
            self._client = ElevenLabs(api_key=api_key)
        return self._client

    def resolve_voice_id(self, subject: str | None) -> str:
        """Pick the voice_id for `subject`, falling back to the default voice
        when `subject` is unscoped or has no entry in `voice_map`.
        """
        voice_id = self.voice_map.get(subject) if subject else None
        voice_id = voice_id or self.default_voice_id
        if not voice_id:
            raise UnknownVoiceSubjectError(
                f"No ElevenLabs voice_id configured for subject {subject!r} and "
                "ELEVENLABS_DEFAULT_VOICE_ID is not set."
            )
        return voice_id

    def synthesize(self, text: str, subject: str | None = None) -> bytes:
        """Convert `text` to raw 16-bit PCM audio (24kHz mono) using the voice
        configured for `subject` (or the default voice if unscoped/unmapped).
        """
        if not text or not text.strip():
            raise SpeechSynthesisError("text must not be empty.")

        voice_id = self.resolve_voice_id(subject)

        def _do_synthesize() -> bytes:
            chunks = self.client.text_to_speech.convert(
                voice_id, text=text, model_id=self.model_id, output_format=_OUTPUT_FORMAT
            )
            return b"".join(chunks)

        try:
            audio_bytes = retry_with_backoff(
                _do_synthesize, should_retry=_is_transient_error, max_attempts=self.max_retries
            )
        except Exception as exc:  # noqa: BLE001 - backend raises various types
            raise SpeechSynthesisError(f"Failed to synthesize speech: {exc}") from exc

        if not audio_bytes:
            raise SpeechSynthesisError("ElevenLabs returned no audio.")

        logger.info("Audio synthesized: %d byte(s), voice_id=%s", len(audio_bytes), voice_id)
        return audio_bytes

    def play(self, pcm_bytes: bytes, *, sample_rate: int = _OUTPUT_SAMPLE_RATE) -> None:
        """Play raw 16-bit PCM audio through the default output device."""
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            self._player.play(samples, samplerate=sample_rate)
            self._player.wait()
        except Exception as exc:  # noqa: BLE001 - PortAudio raises various types
            raise PlaybackError(f"Failed to play audio: {exc}") from exc

    def speak(self, text: str, subject: str | None = None) -> None:
        """Convenience: synthesize `text` and play it immediately."""
        self.play(self.synthesize(text, subject))
