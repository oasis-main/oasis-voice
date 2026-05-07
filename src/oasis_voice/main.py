"""
FastAPI entrypoint — sketch.

This file declares the route surface the openclaw voice-call provider talks
to. Concrete handlers are stubs that return 501 until the corresponding
backend module is implemented.

Run (once deps are installed):
  uvicorn oasis_voice.main:app --host 0.0.0.0 --port 8731
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from .config import resolve_active
from .tiers import all_tiers

app = FastAPI(title="oasis-voice-server", version="0.0.1-sketch")
ACTIVE = resolve_active()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "stt_tier": ACTIVE.stt.name,
        "stt_backend": ACTIVE.stt.backend,
        "stt_loaded": False,                   # backends not wired yet
        "tts_tier": ACTIVE.tts.name,
        "tts_backend": ACTIVE.tts.backend,
        "tts_loaded": False,
        "has_gpu": ACTIVE.has_gpu,
    }


@app.get("/v1/tiers")
def list_tiers() -> dict[str, Any]:
    """Advertise every tier the build supports — not just the loaded ones."""
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


# All real handlers below are stubs. Implementing them is the next milestone.

@app.post("/v1/stt/transcribe")
async def stt_transcribe() -> Any:
    raise HTTPException(501, "not yet implemented — see stt/{moonshine,whisper,...}.py")


@app.websocket("/v1/stt/stream")
async def stt_stream(*_: Any, **__: Any) -> Any:
    raise HTTPException(501, "not yet implemented")


@app.post("/v1/tts/speak")
async def tts_speak() -> Any:
    raise HTTPException(501, "not yet implemented — see tts/{piper,kokoro,...}.py")


@app.websocket("/v1/tts/stream")
async def tts_stream(*_: Any, **__: Any) -> Any:
    raise HTTPException(501, "not yet implemented")


@app.post("/v1/voice/clone")
async def voice_clone() -> Any:
    raise HTTPException(501, "not yet implemented — clone-light/clone-pro tiers only")


@app.get("/v1/voices")
async def list_voices() -> Any:
    return {"presets": [], "cloned": []}
