"""Audio utility unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from oasis_voice.audio import (
    decode_wav,
    encode_wav,
    float32_to_pcm16,
    frame_iterator,
    pcm16_to_float32,
    resample_pcm16,
    split_sentences,
)


def test_split_sentences_basic():
    out = split_sentences("Hello world. This is a test! Is it? Yes.")
    assert out == ["Hello world.", "This is a test!", "Is it?", "Yes."]


def test_split_sentences_empty_or_no_terminator():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("no terminator") == ["no terminator"]


def test_pcm_roundtrip():
    src = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    pcm = float32_to_pcm16(src)
    back = pcm16_to_float32(pcm)
    assert back.dtype == np.float32
    assert np.allclose(src, back, atol=1e-3)


def test_wav_roundtrip():
    sr = 16000
    samples = np.sin(np.linspace(0, 2 * np.pi, sr, dtype=np.float32)) * 0.5
    pcm = float32_to_pcm16(samples)
    wav = encode_wav(pcm, sample_rate=sr)
    back, back_sr, channels = decode_wav(wav)
    assert back_sr == sr
    assert channels == 1
    assert len(back) == len(pcm)


def test_resample_passthrough():
    pcm = b"\x01\x00" * 100
    assert resample_pcm16(pcm, 16000, 16000) == pcm


def test_resample_changes_length():
    pcm = float32_to_pcm16(np.zeros(16000, dtype=np.float32))
    out = resample_pcm16(pcm, 16000, 8000)
    # 8kHz at int16 mono = 2 bytes/sample; 1s of audio → 16000 bytes.
    assert len(out) == pytest.approx(16000, rel=0.01)


def test_frame_iterator_yields_fixed_size():
    pcm = b"\x00\x00" * 16000  # 1s at 16kHz int16 mono
    frames = list(frame_iterator(pcm, sample_rate=16000, frame_ms=20))
    # 50 frames of 20ms at 16kHz = 320 samples * 2 bytes = 640 bytes each
    assert len(frames) == 50
    assert all(len(f) == 640 for f in frames)
