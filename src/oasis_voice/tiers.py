"""
Declarative registry of STT and TTS tiers.

A tier is a (name, backend, weights, hardware) tuple. Backends are loaded
lazily — only the tier(s) selected by VOICE_STT_TIER / VOICE_TTS_TIER are
instantiated at startup. Other tiers are advertised on /v1/tiers but not
loaded.

Adding a new tier:
  1. Add an entry below with a real sha256 from the upstream weights.
  2. Add the matching backend module under stt/ or tts/.
  3. Add a smoke fixture under tests/fixtures/<tier>.

Removing a tier (e.g. the upstream license changes to non-commercial):
  - Delete the entry; CI will fail any test that references it. Do not leave
    a non-permissive model in place behind a "feature flag" — the README
    license posture is a hard constraint, not a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModalityT = Literal["stt", "tts"]
LicenseT = Literal["MIT", "Apache-2.0", "CC-BY-4.0"]
# Note: no "CC-BY-NC", "CPML", "GPL-3.0", or model-specific "research only"
# entries. The Literal narrows it at type-check time too.


@dataclass(frozen=True)
class TierSpec:
    name: str
    modality: ModalityT
    backend: str                  # module under stt/ or tts/, e.g. "moonshine"
    weights: str                  # HF repo id, "<org>/<repo>"
    revision: str                 # commit sha or release tag — never "main"
    sha256: dict[str, str]        # filename → expected hash for download verifier
    requires_gpu: bool
    vram_gb: int                  # 0 if CPU is enough; informational
    license: LicenseT
    streaming: bool               # true streaming (not just chunked sync)
    languages: tuple[str, ...]    # ISO-639-1 codes
    notes: str = ""


# ─────────────────────────────────── STT ───────────────────────────────────

STT_TIERS: dict[str, TierSpec] = {
    "lite": TierSpec(
        name="lite",
        modality="stt",
        backend="moonshine",
        weights="UsefulSensors/moonshine-base",
        revision="TBD",                    # pin before first deploy
        sha256={},                         # populate via scripts/download-weights.py
        requires_gpu=False,
        vram_gb=0,
        license="MIT",
        streaming=True,                    # Moonshine is designed for streaming
        languages=("en",),
        notes="27M params, edge/Pi-friendly. Cloud MVP default.",
    ),
    "default": TierSpec(
        name="default",
        modality="stt",
        backend="whisper",
        weights="distil-whisper/distil-small.en",
        revision="TBD",
        sha256={},
        requires_gpu=False,
        vram_gb=0,
        license="MIT",
        streaming=False,                   # chunked, not true streaming
        languages=("en",),
        notes="Distil-Whisper small for async/bulk transcription.",
    ),
    "stream-light": TierSpec(
        name="stream-light",
        modality="stt",
        backend="sherpa_zipformer",
        weights="csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        revision="TBD",
        sha256={},
        requires_gpu=False,
        vram_gb=0,
        license="Apache-2.0",
        streaming=True,
        languages=("en",),
        notes="True streaming on commodity CPU. Same upstream as sherpa-onnx-tts.",
    ),
    "stream-pro": TierSpec(
        name="stream-pro",
        modality="stt",
        backend="voxtral_realtime",
        weights="mistralai/Voxtral-Mini-4B-Realtime-2602",
        revision="TBD",
        sha256={},
        requires_gpu=True,
        vram_gb=16,
        license="Apache-2.0",
        streaming=True,
        languages=("ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt", "zh", "ja", "ko", "ru"),
        notes="vLLM only. WebSocket streaming, 480ms typical delay.",
    ),
    "accuracy": TierSpec(
        name="accuracy",
        modality="stt",
        backend="parakeet",
        weights="nvidia/parakeet-tdt-0.6b-v2",
        revision="TBD",
        sha256={},
        requires_gpu=True,
        vram_gb=4,
        license="CC-BY-4.0",
        streaming=False,
        languages=("en",),
        notes="Tops Open ASR Leaderboard for size; English-only.",
    ),
}


# ─────────────────────────────────── TTS ───────────────────────────────────

TTS_TIERS: dict[str, TierSpec] = {
    "lite": TierSpec(
        name="lite",
        modality="tts",
        backend="piper",
        weights="rhasspy/piper-voices/en_US-lessac-high",
        revision="TBD",
        sha256={},
        requires_gpu=False,
        vram_gb=0,
        license="MIT",
        streaming=True,                    # Piper streams sentence-by-sentence
        languages=("en",),                 # voice-specific; pick at install
        notes="Anchor for cloud MVP. Already exercised via sherpa-onnx-tts skill.",
    ),
    "default": TierSpec(
        name="default",
        modality="tts",
        backend="kokoro",
        weights="hexgrad/Kokoro-82M",
        revision="TBD",
        sha256={},
        requires_gpu=False,
        vram_gb=0,
        license="Apache-2.0",
        streaming=True,
        languages=("en", "es", "fr", "hi", "it", "ja", "pt", "zh"),
        notes="82M params; sub-200ms TTFA on CPU. Default natural voice.",
    ),
    "natural": TierSpec(
        name="natural",
        modality="tts",
        backend="melotts",
        weights="myshell-ai/MeloTTS-English",
        revision="TBD",
        sha256={},
        requires_gpu=True,
        vram_gb=2,
        license="MIT",
        streaming=False,
        languages=("en", "es", "fr", "ja", "ko", "zh"),
        notes="MyShell-maintained; multi-lingual; expressive.",
    ),
    "style": TierSpec(
        name="style",
        modality="tts",
        backend="styletts2",
        weights="yl4579/StyleTTS2-LibriTTS",
        revision="TBD",
        sha256={},
        requires_gpu=True,
        vram_gb=4,
        license="MIT",                     # gruut variant only — see notes
        streaming=False,
        languages=("en",),
        notes="Use the MIT/gruut variant only. Upstream model agreement also "
              "requires disclosure-of-synthesis to listeners — phone-call recording "
              "disclosure flow must be wired up before exposing this tier publicly.",
    ),
    "clone-light": TierSpec(
        name="clone-light",
        modality="tts",
        backend="openvoice_v2",
        weights="myshell-ai/OpenVoiceV2",
        revision="TBD",
        sha256={},
        requires_gpu=True,
        vram_gb=4,
        license="MIT",
        streaming=False,
        languages=("en", "es", "fr", "ja", "ko", "zh"),
        notes="Tone-color transfer; pairs with MeloTTS for the speaking voice.",
    ),
    "clone-pro": TierSpec(
        name="clone-pro",
        modality="tts",
        backend="cosyvoice2",
        weights="FunAudioLLM/CosyVoice2-0.5B",
        revision="TBD",
        sha256={},
        requires_gpu=True,
        vram_gb=16,
        license="Apache-2.0",
        streaming=True,
        languages=("en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt"),
        notes="Zero-shot cloning from 10s reference. vllm-omni runtime.",
    ),
}


def all_tiers() -> dict[str, TierSpec]:
    """Used by /v1/tiers — flat view for the HTTP API."""
    return {**STT_TIERS, **TTS_TIERS}


def resolve(modality: ModalityT, name: str) -> TierSpec:
    table = STT_TIERS if modality == "stt" else TTS_TIERS
    if name not in table:
        raise KeyError(
            f"Unknown {modality} tier '{name}'. Available: {sorted(table)}"
        )
    return table[name]
