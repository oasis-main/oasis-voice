"""
Moonshine STT backend tests.

We mock the upstream `moonshine-onnx` model so we exercise the buffering,
resampling, silence-detection, and partial-emission logic without needing
ONNX runtime or weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from oasis_voice.audio import float32_to_pcm16
from oasis_voice.stt.moonshine import MoonshineBackend


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[int] = []  # sample counts per generate() call

    def generate(self, batch: np.ndarray) -> Any:
        # batch is shape (1, T); we encode "len(T) // 1000" as a token list.
        n = batch.shape[1]
        self.calls.append(n)
        return [[n]]  # single row, one fake token


class FakeTokenizer:
    def decode_batch(self, tokens):
        # Render the token (sample count) deterministically.
        return [f"text@{tokens[0][0]}"]


@pytest.fixture
def fake_backend(monkeypatch):
    backend = MoonshineBackend(silence_ms=200, max_chunk_ms=1000)

    async def fake_warmup(self):
        self._model = FakeModel()
        self._tokenizer = FakeTokenizer()
        self._loaded = True

    monkeypatch.setattr(MoonshineBackend, "warmup", fake_warmup)
    return backend


def _silence_pcm(seconds: float, sr: int = 16000) -> bytes:
    return float32_to_pcm16(np.zeros(int(sr * seconds), dtype=np.float32))


def _tone_pcm(seconds: float, sr: int = 16000, freq: int = 440) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return float32_to_pcm16(np.sin(2 * np.pi * freq * t) * 0.5)


@pytest.mark.asyncio
async def test_transcribe_sync(fake_backend):
    await fake_backend.warmup()
    pcm = _tone_pcm(0.5)  # 0.5s of audio
    chunk = await fake_backend.transcribe(pcm, sample_rate=16000)
    assert chunk.is_final is True
    assert chunk.text.startswith("text@")
    assert chunk.language == "en"
    assert chunk.end_ms == pytest.approx(500, abs=20)


@pytest.mark.asyncio
async def test_transcribe_resamples(fake_backend):
    """8kHz input should be resampled to 16kHz before decode."""
    await fake_backend.warmup()
    pcm = _tone_pcm(1.0, sr=8000)
    chunk = await fake_backend.transcribe(pcm, sample_rate=8000)
    # FakeModel records the sample count it saw; should be ~16k for 1s @ 16k
    assert fake_backend._model.calls[-1] == pytest.approx(16000, rel=0.01)
    assert chunk.is_final is True


@pytest.mark.asyncio
async def test_stream_emits_partial_then_final(fake_backend):
    await fake_backend.warmup()

    async def frames():
        # 0.5s of speech, then 0.4s of silence (>200ms threshold) → triggers a partial.
        # Then more speech, then close.
        yield _tone_pcm(0.5)
        yield _silence_pcm(0.4)
        yield _tone_pcm(0.3)

    chunks = []
    async for c in fake_backend.stream(frames(), sample_rate=16000):
        chunks.append(c)

    # We expect at least one partial and exactly one terminal.
    finals = [c for c in chunks if c.is_final]
    partials = [c for c in chunks if not c.is_final]
    assert len(finals) == 1
    assert len(partials) >= 1
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_stream_min_audio_skipped(fake_backend):
    """Frames below MIN_AUDIO_SAMPLES shouldn't reach the decoder."""
    await fake_backend.warmup()

    async def frames():
        # 50ms — well under 100ms minimum.
        yield _tone_pcm(0.05)

    chunks = [c async for c in fake_backend.stream(frames(), sample_rate=16000)]
    # Final chunk emitted on close, but model.generate should NOT have been
    # called with the tiny sub-threshold buffer.
    assert chunks[-1].is_final
    assert fake_backend._model.calls == []


@pytest.mark.asyncio
async def test_warmup_required(fake_backend):
    backend = MoonshineBackend()  # not warmed up
    with pytest.raises(RuntimeError, match="warmup"):
        await backend.transcribe(b"\x00\x00" * 16000, sample_rate=16000)
