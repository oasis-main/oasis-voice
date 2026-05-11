"""
End-to-end-ish tests for the FastAPI surface.

We monkeypatch the loader so warmup is skipped and the backends are simple
fakes; this verifies the route plumbing (multipart upload, WAV decode, JSON
response shape) without requiring real models.
"""

from __future__ import annotations

import io
import os
from collections.abc import AsyncIterator

os.environ.setdefault("VOICE_SKIP_WARMUP", "1")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oasis_voice.audio import encode_wav, float32_to_pcm16
from oasis_voice.stt.base import STTBackend, TranscriptionChunk
from oasis_voice.tts.base import AudioChunk, TTSBackend, VoiceRef


class StubSTT(STTBackend):
    async def warmup(self) -> None:
        return None

    async def transcribe(self, audio_bytes, *, sample_rate):
        return TranscriptionChunk(
            text="stub-text",
            is_final=True,
            start_ms=0,
            end_ms=len(audio_bytes) // (sample_rate * 2 // 1000),
            confidence=None,
            language="en",
        )

    async def stream(self, frames, *, sample_rate) -> AsyncIterator[TranscriptionChunk]:
        async for _ in frames:
            pass
        yield TranscriptionChunk(
            text="stub-final", is_final=True, start_ms=0, end_ms=0,
            confidence=None, language="en",
        )


class StubTTS(TTSBackend):
    supports_streaming = True
    supports_cloning = False

    async def warmup(self) -> None:
        return None

    async def speak(self, text, voice):
        pcm = float32_to_pcm16(np.zeros(1600, dtype=np.float32))  # 0.1s @ 16k
        return AudioChunk(pcm=pcm, sample_rate=16000, channels=1, is_final=True)

    async def stream(self, text, voice) -> AsyncIterator[AudioChunk]:
        yield AudioChunk(
            pcm=float32_to_pcm16(np.zeros(1600, dtype=np.float32)),
            sample_rate=16000,
            channels=1,
            is_final=True,
        )


@pytest.fixture
def client(monkeypatch):
    # Patch the loader BEFORE importing main.
    from oasis_voice import loader

    monkeypatch.setattr(loader, "make_stt", lambda _: StubSTT())
    monkeypatch.setattr(loader, "make_tts", lambda _: StubTTS())

    # Re-import main fresh so it picks up the patched loader.
    import importlib

    from oasis_voice import main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["stt_backend"] == "moonshine"
    assert body["tts_backend"] == "piper"


def test_list_tiers(client):
    r = client.get("/v1/tiers")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] == {"stt": "lite", "tts": "lite"}
    names = {t["name"] for t in body["tiers"]}
    assert "lite" in names


def test_stt_transcribe(client):
    pcm = float32_to_pcm16(np.zeros(16000, dtype=np.float32))
    wav = encode_wav(pcm, sample_rate=16000)
    r = client.post(
        "/v1/stt/transcribe",
        files={"audio": ("clip.wav", io.BytesIO(wav), "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["text"] == "stub-text"


def test_tts_speak(client):
    r = client.post("/v1/tts/speak", json={"text": "hello world"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_tts_speak_rejects_unknown_format(client):
    r = client.post(
        "/v1/tts/speak?format=mp3",
        json={"text": "hi"},
    )
    assert r.status_code == 400
    assert "unsupported format" in r.json()["detail"]


def test_tts_speak_opus_calls_encoder(client, monkeypatch):
    """
    With format=opus the endpoint should route through encode_opus.
    We monkeypatch the encoder so this works on hosts without ffmpeg
    (the real-ffmpeg path is covered in test_audio.py's
    test_encode_opus_roundtrip_via_real_ffmpeg).
    """
    import oasis_voice.main as voice_main

    called = {"hit": False, "args": None}

    def fake_encode(pcm, sample_rate, channels=1):
        called["hit"] = True
        called["args"] = (len(pcm), sample_rate, channels)
        return b"OggS" + b"\x00" * 64

    monkeypatch.setattr(voice_main, "encode_opus", fake_encode)

    r = client.post("/v1/tts/speak?format=opus", json={"text": "hi"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/ogg"
    assert r.content.startswith(b"OggS")
    assert called["hit"], "encode_opus was not called"


def test_voice_clone_unsupported(client):
    pcm = float32_to_pcm16(np.zeros(16000, dtype=np.float32))
    wav = encode_wav(pcm, sample_rate=16000)
    r = client.post(
        "/v1/voice/clone",
        files={"audio": ("ref.wav", io.BytesIO(wav), "audio/wav")},
        data={"voice_id": "test"},
    )
    assert r.status_code == 501
