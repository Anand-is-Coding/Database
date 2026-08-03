"""Microphone capture -- records from the default input device until the
user presses Enter again, returning a WAV file ready for
`SpeechToTextService`.

Single responsibility: (nothing) -> WAV bytes. Knows nothing about
transcription, retrieval, or generation -- it only turns "how long did the
user talk" into an in-memory WAV file.
"""

from __future__ import annotations

import io
import threading
import wave

import numpy as np
import sounddevice as sd

from config.logging import get_logger
from voice.exceptions import RecordingError

logger = get_logger(__name__)


def record_until_enter(*, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Record from the default microphone until the user presses Enter.

    Returns a complete, in-memory WAV file (16-bit PCM) -- the format
    ElevenLabs' speech-to-text endpoint accepts directly as `file=`.
    """
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def _on_audio(indata: np.ndarray, frame_count: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            logger.warning("Audio input status: %s", status)
        frames.append(indata.copy())

    def _wait_for_enter() -> None:
        input()
        stop_event.set()

    listener = threading.Thread(target=_wait_for_enter, daemon=True)
    listener.start()

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="int16", callback=_on_audio):
            while not stop_event.is_set():
                sd.sleep(100)
    except Exception as exc:  # noqa: BLE001 - PortAudio raises various types
        raise RecordingError(f"Failed to record from the microphone: {exc}") from exc

    if not frames:
        raise RecordingError("No audio was captured.")

    audio = np.concatenate(frames, axis=0)
    logger.info("Recording captured: %.1fs", len(audio) / sample_rate)
    return to_wav_bytes(audio, sample_rate=sample_rate, channels=channels)


def to_wav_bytes(audio: np.ndarray, *, sample_rate: int, channels: int) -> bytes:
    """Encode 16-bit PCM samples as an in-memory WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()
