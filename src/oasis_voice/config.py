"""
Boot-time tier resolution.

Reads VOICE_STT_TIER and VOICE_TTS_TIER from the env, looks them up in the
registry, and verifies the host can actually run them (GPU presence, VRAM
budget). Loud, early failure beats a silent fallback at first request.

This is the one place "the cloud MVP" and "Nimbus" deployments differ.
Same code, same config keys, just different env values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .tiers import STT_TIERS, TTS_TIERS, TierSpec


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActiveTiers:
    stt: TierSpec
    tts: TierSpec
    has_gpu: bool


def detect_gpu() -> tuple[bool, int]:
    """Returns (present, vram_gb). VRAM is the smallest visible device."""
    try:
        import torch
    except ImportError:
        return (False, 0)
    if not torch.cuda.is_available():
        return (False, 0)
    vram_gb = min(
        torch.cuda.get_device_properties(i).total_memory // (1024**3)
        for i in range(torch.cuda.device_count())
    )
    return (True, int(vram_gb))


def resolve_active() -> ActiveTiers:
    stt_name = os.environ.get("VOICE_STT_TIER", "lite")
    tts_name = os.environ.get("VOICE_TTS_TIER", "lite")

    if stt_name not in STT_TIERS:
        raise ConfigError(f"VOICE_STT_TIER='{stt_name}' not in {sorted(STT_TIERS)}")
    if tts_name not in TTS_TIERS:
        raise ConfigError(f"VOICE_TTS_TIER='{tts_name}' not in {sorted(TTS_TIERS)}")

    stt = STT_TIERS[stt_name]
    tts = TTS_TIERS[tts_name]

    has_gpu, vram_gb = detect_gpu()
    needed_gpu_gb = max(
        stt.vram_gb if stt.requires_gpu else 0,
        tts.vram_gb if tts.requires_gpu else 0,
    )
    if needed_gpu_gb > 0 and not has_gpu:
        raise ConfigError(
            f"Tier(s) require GPU but none detected: stt={stt_name}({stt.vram_gb}GB), "
            f"tts={tts_name}({tts.vram_gb}GB)"
        )
    if needed_gpu_gb > vram_gb:
        raise ConfigError(
            f"Tier(s) need {needed_gpu_gb}GB VRAM, only {vram_gb}GB available."
        )

    return ActiveTiers(stt=stt, tts=tts, has_gpu=has_gpu)
