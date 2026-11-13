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


# ────────────── voice enumeration (CLAW-107 Phase 1) ──────────────
#
# GET /v1/voices used to be a stub returning empty lists, so a bot could not
# discover what voices existed and therefore could not choose one. These cover
# the directory walk that replaced it.


def _write_voice(directory: Path, name: str, speakers: dict[str, int] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    onnx = directory / f"{name}.onnx"
    onnx.write_bytes(b"fake")
    if speakers is not None:
        import json as _json
        (directory / f"{name}.onnx.json").write_text(_json.dumps({"speaker_id_map": speakers}))
    return onnx


def test_list_presets_walks_search_paths(tmp_path, monkeypatch):
    _write_voice(tmp_path, "en_GB-alan-medium")
    _write_voice(tmp_path, "en_GB-aru-medium")
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    presets = PiperBackend().list_presets()

    assert [p.voice_id for p in presets] == [
        "piper:en_GB-alan-medium",
        "piper:en_GB-aru-medium",
    ]
    assert all(p.speakers == () for p in presets)


def test_list_presets_reports_multi_speaker_names(tmp_path, monkeypatch):
    _write_voice(tmp_path, "en_GB-vctk-medium", speakers={"p236": 0, "p239": 1, "p244": 2})
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    (preset,) = PiperBackend().list_presets()

    assert preset.voice_id == "piper:en_GB-vctk-medium"
    assert preset.speakers == ("p236", "p239", "p244")


def test_list_presets_single_entry_map_reports_no_choice(tmp_path, monkeypatch):
    # A one-speaker map is a single-speaker model. Reporting one option would
    # imply a choice the caller does not actually have.
    _write_voice(tmp_path, "en_GB-alan-medium", speakers={"only": 0})
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    (preset,) = PiperBackend().list_presets()
    assert preset.speakers == ()


def test_list_presets_higher_priority_dir_shadows(tmp_path, monkeypatch):
    # Same voice installed twice. resolve_voice_path returns the override copy,
    # so enumeration must report it once, not twice.
    override, xdg = tmp_path / "override", tmp_path / "xdg"
    _write_voice(override, "en_GB-alan-medium", speakers={"a": 0, "b": 1})
    _write_voice(xdg / "oasis-voice" / "piper", "en_GB-alan-medium")
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(override))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))

    presets = PiperBackend().list_presets()

    assert len(presets) == 1
    assert presets[0].speakers == ("a", "b")  # the override copy won


def test_list_presets_survives_missing_and_unreadable_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(tmp_path / "does-not-exist"))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert PiperBackend().list_presets() == []


def test_list_presets_ignores_corrupt_sidecar(tmp_path, monkeypatch):
    _write_voice(tmp_path, "en_GB-alan-medium")
    (tmp_path / "en_GB-alan-medium.onnx.json").write_text("{not json")
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    (preset,) = PiperBackend().list_presets()
    assert preset.speakers == ()  # degrades, does not raise


def test_list_presets_does_not_require_warmup(tmp_path, monkeypatch):
    # THE contract for this method. Listing must not load a model: on the lite
    # tier warmup downloads a voice (~60s cold), and a caller deciding WHICH
    # voice to use must not pay that just to see the options.
    _write_voice(tmp_path, "en_GB-alan-medium")
    monkeypatch.setenv("VOICE_PIPER_VOICE_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    backend = PiperBackend()
    assert backend._PiperVoice is None  # warmup() never called
    assert len(backend.list_presets()) == 1
    assert backend._PiperVoice is None  # and still not called
    assert backend._voices == {}        # no model was loaded
