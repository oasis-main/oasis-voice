"""
STT backend interface.

Every concrete backend (moonshine, whisper, sherpa_zipformer, voxtral_realtime,
parakeet) implements this. Streaming and chunked-sync backends both go through
the same iterator shape — non-streaming backends just yield once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import AsyncIterator


@dataclass(frozen=True)
class TranscriptionChunk:
    text: str
    is_final: bool                 # True for the terminal chunk only
    start_ms: int                  # offset from session start
    end_ms: int
    confidence: float | None       # 0..1; None if backend doesn't provide
    language: str                  # ISO-639-1


class STTBackend(ABC):
    @abstractmethod
    async def warmup(self) -> None:
        """Load weights, JIT-compile, allocate buffers. Called once at boot."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, *, sample_rate: int) -> TranscriptionChunk:
        """Sync: full audio → single TranscriptionChunk(is_final=True)."""

    @abstractmethod
    async def stream(
        self,
        frames: AsyncIterator[bytes],
        *,
        sample_rate: int,
    ) -> AsyncIterator[TranscriptionChunk]:
        """
        Streaming: audio frames in, partial+final chunks out.

        Non-streaming backends (whisper, parakeet) implement this by buffering
        until a silence threshold or a configured max-chunk-ms, then calling
        their own transcribe(). Caller sees the same shape either way.
        """
