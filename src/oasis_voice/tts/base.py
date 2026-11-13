"""
TTS backend interface.

Same shape as STT: a sync entry point and a streaming entry point. The TTFA
(time-to-first-audio) story is the whole product, so streaming is first-class.

Voice handling: every backend declares whether it supports preset voices,
voice cloning from a reference, or both. The /v1/voices endpoint composes
these into a single registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import AsyncIterator
from typing import Literal


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes                     # raw signed-16 little-endian PCM
    sample_rate: int               # Hz
    channels: int                  # 1 or 2
    is_final: bool                 # True for terminal chunk only


@dataclass(frozen=True)
class VoiceRef:
    """
    Identifies a voice. Either a preset name (`"piper:en_US-lessac-high"`)
    or a cloned voice id minted by /v1/voice/clone (`"clone:abc123"`).

    For multi-speaker models, append ``#speaker`` to the voice_id
    (e.g. ``"piper:en_GB-vctk-medium#p236"``). The backend resolves the
    speaker name to an integer id via the model's speaker_id_map.
    """
    voice_id: str
    kind: Literal["preset", "clone"]
    speaker: str | None = None


@dataclass(frozen=True)
class VoicePreset:
    """
    One installed, selectable voice, as reported by GET /v1/voices.

    `speakers` is empty for a single-speaker model. For a multi-speaker model
    it lists the selectable speaker names; a caller addresses one by appending
    ``#speaker`` to the voice_id, the same form `_voice_ref` parses.
    """
    voice_id: str                    # fully qualified, e.g. "piper:en_GB-alan-medium"
    speakers: tuple[str, ...] = ()


class TTSBackend(ABC):
    supports_cloning: bool = False
    supports_streaming: bool = False

    def list_presets(self) -> list[VoicePreset]:
        """
        Enumerate the preset voices this backend can serve RIGHT NOW.

        Contract: this must be cheap and must NOT require warmup(). A caller
        listing voices should not pay a cold model download. Piper resolves
        voices from disk at request time, so its implementation is a directory
        walk. A backend that genuinely cannot enumerate returns [].
        """
        return []

    def list_cloned(self) -> list[VoicePreset]:
        """
        Enumerate cloned voices minted through /v1/voice/clone. Same cheapness
        contract as list_presets. Backends without cloning return [].
        """
        return []

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    async def speak(self, text: str, voice: VoiceRef) -> AudioChunk:
        """Sync: full text → single AudioChunk(is_final=True)."""

    @abstractmethod
    async def stream(
        self,
        text: str,
        voice: VoiceRef,
    ) -> AsyncIterator[AudioChunk]:
        """
        Streaming: text in, audio frames out as fast as the model produces them.
        Non-streaming backends emit one chunk and finish — caller code is
        identical either way.
        """

    async def clone(self, *, reference_pcm: bytes, sample_rate: int, voice_id: str) -> VoiceRef:
        """
        Mint a cloned-voice reference from ~10s of speaker audio. Only
        backends with supports_cloning=True implement this; the default
        raises NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support cloning")
