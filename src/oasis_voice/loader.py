"""
Backend factory.

Each tier names a backend module under stt/ or tts/. This file is the only
place that knows how to map `backend="piper"` to a concrete class. Lazy
imports here keep `main.py` free of optional-deps coupling.
"""

from __future__ import annotations

from .stt.base import STTBackend
from .tts.base import TTSBackend


def make_stt(backend_name: str) -> STTBackend:
    if backend_name == "moonshine":
        from .stt.moonshine import MoonshineBackend
        return MoonshineBackend()
    raise NotImplementedError(
        f"STT backend '{backend_name}' is declared in tiers.py but no "
        f"adapter is wired up yet. Add one under src/oasis_voice/stt/."
    )


def make_tts(backend_name: str) -> TTSBackend:
    if backend_name == "piper":
        from .tts.piper import PiperBackend
        return PiperBackend()
    raise NotImplementedError(
        f"TTS backend '{backend_name}' is declared in tiers.py but no "
        f"adapter is wired up yet. Add one under src/oasis_voice/tts/."
    )
