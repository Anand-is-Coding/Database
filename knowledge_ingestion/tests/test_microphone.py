"""Unit tests for the pure WAV-encoding part of `voice/microphone.py`.

Actual microphone capture needs real audio hardware and isn't unit-tested
here - `to_wav_bytes` is the deterministic, hardware-free part.
"""

from __future__ import annotations

import wave
from io import BytesIO

import numpy as np

from voice.microphone import to_wav_bytes


def test_to_wav_bytes_round_trips_samples():
    samples = np.array([0, 100, -100, 32767, -32768], dtype=np.int16)

    wav_bytes = to_wav_bytes(samples, sample_rate=16000, channels=1)

    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        frames = wav_file.readframes(wav_file.getnframes())
        round_tripped = np.frombuffer(frames, dtype=np.int16)

    assert list(round_tripped) == list(samples)
