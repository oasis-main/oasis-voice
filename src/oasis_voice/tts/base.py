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
    """
    voice_id: str
    kind: Literal["preset", "clone"]


class TTSBackend(ABC):
    supports_cloning: bool = False
    supports_streaming: bool = False

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
