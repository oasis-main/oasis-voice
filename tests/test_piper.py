"""
Piper TTS backend tests.

We mock out the upstream `piper-tts` package so the orchestration logic
(sentence splitting, streaming, voice resolution, sample-rate plumbing)
can be exercised without actually installing piper or downloading voices.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from oasis_voice.tts.base import VoiceRef
from oasis_voice.tts.piper import PiperBackend, resolve_voice_path


# ────────────── voice path resolver ──────────────


def test_resolve_voice_path_finds_override(tmp_path, monkeypatch):
    voice = tmp_path / "en_US-lessac-high.onnx"
    voice.write_bytes(b"fake")
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(tmp_path))
    assert resolve_voice_path("en_US-lessac-high") == voice
    assert resolve_voice_path("piper:en_US-lessac-high") == voice


def test_resolve_voice_path_missing_lists_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    with pytest.raises(FileNotFoundError, match="Piper voice 'nope' not found"):
        resolve_voice_path("nope")


# ────────────── synth orchestration ──────────────


class FakeChunk:
    def __init__(self, payload: bytes) -> None:
        self.audio_int16_bytes = payload
        self.sample_rate = 22050


class FakePiperVoice:
    """Mimics the public surface of `piper.PiperVoice` we depend on."""

    config = SimpleNamespace(sample_rate=22050)

    def __init__(self, transcripts_seen: list[str]) -> None:
        self._seen = transcripts_seen

    def synthesize(self, text, **_kwargs):
        self._seen.append(text)
        # 0.05s of "audio" per sentence, deterministically derived from text.
        yield FakeChunk(bytes([len(text) % 256]) * 1000)


@pytest.fixture
def fake_piper(monkeypatch):
    """Patch PiperBackend so warmup() doesn't need the real package."""
    transcripts: list[str] = []

    async def fake_warmup(self):
        self._PiperVoice = lambda *a, **kw: None  # not actually called
        self._SynthesisConfig = None
        self._voices["en_US-lessac-high"] = FakePiperVoice(transcripts)

    monkeypatch.setattr(PiperBackend, "warmup", fake_warmup)
    monkeypatch.setattr(
        PiperBackend,
        "_ensure_voice",
        lambda self, voice_id: self._voices["en_US-lessac-high"],
    )
    return transcripts


@pytest.mark.asyncio
async def test_speak_concatenates_sentences(fake_piper):
    backend = PiperBackend()
    await backend.warmup()
    chunk = await backend.speak(
        "Hello world. This is line two.",
        VoiceRef(voice_id="piper:en_US-lessac-high", kind="preset"),
    )
    assert chunk.is_final is True
    assert chunk.sample_rate == 22050
    assert chunk.channels == 1
    # Two sentences * 1000 bytes per fake render
    assert len(chunk.pcm) == 2000
    assert fake_piper == ["Hello world.", "This is line two."]


@pytest.mark.asyncio
async def test_stream_emits_per_sentence(fake_piper):
    backend = PiperBackend()
    await backend.warmup()
    chunks = []
    async for c in backend.stream(
        "First. Second. Third.",
        VoiceRef(voice_id="piper:en_US-lessac-high", kind="preset"),
    ):
        chunks.append(c)
    assert len(chunks) == 3
    assert [c.is_final for c in chunks] == [False, False, True]
    assert all(c.sample_rate == 22050 for c in chunks)


@pytest.mark.asyncio
async def test_stream_empty_text_terminates(fake_piper):
    backend = PiperBackend()
    await backend.warmup()
    chunks = [c async for c in backend.stream(
        "   ",
        VoiceRef(voice_id="piper:en_US-lessac-high", kind="preset"),
    )]
    assert len(chunks) == 1
    assert chunks[0].is_final is True
    assert chunks[0].pcm == b""


@pytest.mark.asyncio
async def test_clone_voice_rejected(fake_piper):
    backend = PiperBackend()
    await backend.warmup()
    with pytest.raises(ValueError, match="preset"):
        await backend.speak("hi", VoiceRef(voice_id="clone:abc", kind="clone"))
