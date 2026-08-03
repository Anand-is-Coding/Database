"""Custom exceptions for the voice stage.

Kept local to `voice/` so callers only need to handle a small, well-defined
set of errors instead of sounddevice/ElevenLabs internals leaking through.
"""

from __future__ import annotations


class VoiceError(Exception):
    """Base exception for all voice I/O failures."""


class RecordingError(VoiceError):
    """Microphone recording failed."""


class TranscriptionError(VoiceError):
    """Speech-to-text failed to produce a transcript."""


class EmptyTranscriptError(VoiceError):
    """Speech-to-text returned no usable text (e.g. silence)."""


class SpeechSynthesisError(VoiceError):
    """Text-to-speech failed to produce audio."""


class UnknownVoiceSubjectError(VoiceError):
    """No voice_id is configured for the requested subject and no default is set."""


class PlaybackError(VoiceError):
    """Audio playback failed."""
