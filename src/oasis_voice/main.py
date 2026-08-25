"""
FastAPI entrypoint.

The route surface is what the openclaw voice-call provider speaks. We
instantiate exactly the active STT and TTS backends at startup — anything
else stays advertised on /v1/tiers but unloaded.

Run:
    uvicorn oasis_voice.main:app --host 0.0.0.0 --port 8731
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from .audio import (
    AudioDecodeError,
    decode_audio_any,
    decode_wav,
    encode_opus,
    encode_wav,
    frame_iterator,
)
from .config import resolve_active
from .loader import make_stt, make_tts
from .stt.base import STTBackend
from .tiers import all_tiers
from .tts.base import TTSBackend, VoiceRef


log = logging.getLogger("oasis_voice")
ACTIVE = resolve_active()

# Set during lifespan startup.
_stt: STTBackend | None = None
_tts: TTSBackend | None = None

# Lazy-warmup state. When VOICE_SKIP_WARMUP=1 we defer the model load to the
# first request and gate it on these flags+locks so concurrent requests don't
# race the model into existence twice.
_stt_warmed = False
_tts_warmed = False
_stt_warmup_lock: asyncio.Lock | None = None
_tts_warmup_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _stt, _tts, _stt_warmed, _tts_warmed
    global _stt_warmup_lock, _tts_warmup_lock
    skip_warmup = os.environ.get("VOICE_SKIP_WARMUP") == "1"
    _stt = make_stt(ACTIVE.stt.backend)
    _tts = make_tts(ACTIVE.tts.backend)
    _stt_warmup_lock = asyncio.Lock()
    _tts_warmup_lock = asyncio.Lock()
    if not skip_warmup:
        await asyncio.gather(_stt.warmup(), _tts.warmup())
        _stt_warmed = True
        _tts_warmed = True
        log.info(
            "warmup complete: stt=%s tts=%s", ACTIVE.stt.backend, ACTIVE.tts.backend
        )
    else:
        log.warning("VOICE_SKIP_WARMUP=1 — backends will load on first request")
    yield


async def _ensure_stt_warmed() -> None:
    global _stt_warmed
    if _stt_warmed or _stt is None or _stt_warmup_lock is None:
        return
    async with _stt_warmup_lock:
        if _stt_warmed:
            return
        log.info("lazy-loading STT backend on first request: %s", ACTIVE.stt.backend)
        await _stt.warmup()
        _stt_warmed = True


async def _ensure_tts_warmed() -> None:
    global _tts_warmed
    if _tts_warmed or _tts is None or _tts_warmup_lock is None:
        return
    async with _tts_warmup_lock:
        if _tts_warmed:
            return
        log.info("lazy-loading TTS backend on first request: %s", ACTIVE.tts.backend)
        await _tts.warmup()
        _tts_warmed = True


app = FastAPI(title="oasis-voice", version="0.1.0", lifespan=lifespan)


# ─────────────────────────── meta ───────────────────────────


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "stt_tier": ACTIVE.stt.name,
        "stt_backend": ACTIVE.stt.backend,
        "stt_instantiated": _stt is not None,
        "stt_loaded": _stt_warmed,
        "tts_tier": ACTIVE.tts.name,
        "tts_backend": ACTIVE.tts.backend,
        "tts_instantiated": _tts is not None,
        "tts_loaded": _tts_warmed,
        "has_gpu": ACTIVE.has_gpu,
    }


@app.get("/v1/tiers")
def list_tiers() -> dict[str, Any]:
    return {
        "active": {"stt": ACTIVE.stt.name, "tts": ACTIVE.tts.name},
        "tiers": [
            {
                "name": t.name,
                "modality": t.modality,
                "backend": t.backend,
                "license": t.license,
                "requires_gpu": t.requires_gpu,
                "vram_gb": t.vram_gb,
                "streaming": t.streaming,
                "languages": list(t.languages),
                "notes": t.notes,
            }
            for t in all_tiers().values()
        ],
    }


# ─────────────────────────── STT ───────────────────────────


async def _require_stt() -> STTBackend:
    if _stt is None:
        raise HTTPException(503, "STT backend not instantiated yet")
    await _ensure_stt_warmed()
    return _stt


@app.post("/v1/stt/transcribe")
async def stt_transcribe(audio: UploadFile = File(...)) -> dict[str, Any]:
    backend = await _require_stt()
    raw = await audio.read()
    try:
        pcm, sr, _ch = decode_audio_any(
            raw, mime=audio.content_type, file_name=audio.filename
        )
    except AudioDecodeError as e:
        raise HTTPException(415, f"unsupported audio format: {e}") from e
    except Exception as e:
        raise HTTPException(415, f"audio decode failed: {e}") from e
    chunk = await backend.transcribe(pcm, sample_rate=sr)
    return {
        "text": chunk.text,
        "is_final": chunk.is_final,
        "start_ms": chunk.start_ms,
        "end_ms": chunk.end_ms,
        "confidence": chunk.confidence,
        "language": chunk.language,
    }


@app.websocket("/v1/stt/stream")
async def stt_stream(ws: WebSocket) -> None:
    await ws.accept()
    backend = await _require_stt()
    sample_rate = int(ws.query_params.get("sample_rate", "16000"))

    async def frames():
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                data = msg.get("bytes")
                if data:
                    yield data
                # Text messages reserved for control; ignored for now.
        except WebSocketDisconnect:
            return

    try:
        async for chunk in backend.stream(frames(), sample_rate=sample_rate):
            await ws.send_json(
                {
                    "text": chunk.text,
                    "is_final": chunk.is_final,
                    "start_ms": chunk.start_ms,
                    "end_ms": chunk.end_ms,
                    "confidence": chunk.confidence,
                    "language": chunk.language,
                }
            )
            if chunk.is_final:
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


# ─────────────────────────── TTS ───────────────────────────


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    voice: str | None = None  # e.g. "piper:en_US-lessac-high"; backend default if None


async def _require_tts() -> TTSBackend:
    if _tts is None:
        raise HTTPException(503, "TTS backend not instantiated yet")
    await _ensure_tts_warmed()
    return _tts


def _voice_ref(name: str | None) -> VoiceRef:
    if not name:
        return VoiceRef(voice_id="", kind="preset")
    kind = "clone" if name.startswith("clone:") else "preset"
    speaker: str | None = None
    voice_id = name
    if kind == "preset" and "#" in name:
        voice_id, speaker = name.rsplit("#", 1)
        speaker = speaker.strip() or None
    return VoiceRef(voice_id=voice_id, kind=kind, speaker=speaker)


@app.post("/v1/tts/speak")
async def tts_speak(req: SpeakRequest, format: str = "wav") -> Response:
    """
    Synthesize `req.text` to audio. Default output is WAV; pass
    `?format=opus` to receive Opus-in-OGG (used by channels that
    treat opus specifically as a "voice note" — Telegram sendVoice,
    iMessage audio messages, etc.).
    """
    backend = await _require_tts()
    chunk = await backend.speak(req.text, _voice_ref(req.voice))
    fmt = format.lower().strip()
    if fmt == "wav":
        wav = encode_wav(chunk.pcm, chunk.sample_rate, channels=chunk.channels)
        return Response(content=wav, media_type="audio/wav")
    if fmt in ("opus", "ogg", "ogg-opus"):
        try:
            opus = encode_opus(chunk.pcm, chunk.sample_rate, channels=chunk.channels)
        except AudioDecodeError as e:
            raise HTTPException(500, f"opus encode failed: {e}") from e
        return Response(content=opus, media_type="audio/ogg")
    raise HTTPException(400, f"unsupported format='{format}'; supported: wav, opus")


@app.websocket("/v1/tts/stream")
async def tts_stream(ws: WebSocket) -> None:
    await ws.accept()
    backend = await _require_tts()
    try:
        spec = await ws.receive_json()
    except Exception as e:
        await ws.close(code=1003, reason=f"bad request: {e}")
        return

    text = (spec.get("text") or "").strip()
    voice_name = spec.get("voice")
    if not text:
        await ws.close(code=1003, reason="missing 'text'")
        return

    voice = _voice_ref(voice_name)
    try:
        first = True
        async for chunk in backend.stream(text, voice):
            if first:
                await ws.send_json(
                    {
                        "type": "header",
                        "sample_rate": chunk.sample_rate,
                        "channels": chunk.channels,
                        "encoding": "pcm_s16le",
                    }
                )
                first = False
            for frame in frame_iterator(chunk.pcm, chunk.sample_rate):
                await ws.send_bytes(frame)
            await ws.send_json({"type": "boundary", "is_final": chunk.is_final})
    except WebSocketDisconnect:
        return
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


# ─────────────────────────── voice registry ───────────────────────────


@app.post("/v1/voice/clone")
async def voice_clone(
    audio: UploadFile = File(...),
    voice_id: str = Form(...),
) -> dict[str, Any]:
    backend = await _require_tts()
    if not backend.supports_cloning:
        raise HTTPException(
            501,
            f"active TTS backend '{ACTIVE.tts.backend}' does not support cloning. "
            f"Switch to a clone-light or clone-pro tier.",
        )
    raw = await audio.read()
    pcm, sr, _ = decode_wav(raw)
    ref = await backend.clone(reference_pcm=pcm, sample_rate=sr, voice_id=voice_id)
    return {"voice_id": ref.voice_id, "kind": ref.kind}


@app.get("/v1/voices")
async def list_voices() -> dict[str, Any]:
    """
    Enumerate the voices a caller may select.

    Deliberately does NOT go through _require_tts(): that warms the backend,
    and warmup on the lite tier downloads a voice on first hit (~60s cold).
    Listing is a filesystem walk, so it must stay cheap enough to call before
    deciding which voice to use. A backend that has not been instantiated yet
    still 503s, because "which voices exist" has no answer without one.
    """
    if _tts is None:
        raise HTTPException(503, "TTS backend not instantiated yet")
    presets = [
        {"voice_id": v.voice_id, "speakers": list(v.speakers)}
        for v in _tts.list_presets()
    ]
    cloned = [
        {"voice_id": v.voice_id, "speakers": list(v.speakers)}
        for v in _tts.list_cloned()
    ]
    return {
        "presets": presets,
        "cloned": cloned,
        "backend": ACTIVE.tts.backend,
        "supports_cloning": _tts.supports_cloning,
    }
