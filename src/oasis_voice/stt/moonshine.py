"""
Moonshine STT backend — `tier=lite`.

Moonshine is MIT-licensed, ~27M params, runs comfortably on a single CPU
core, and was designed for streaming. Useful Sensors ships an ONNX build
on Hugging Face under `UsefulSensors/moonshine-base`. This wrapper uses
the upstream `moonshine-onnx` Python package; we don't fork or vendor any
weights — they're fetched separately via download_weights.py.

Streaming model: Moonshine is fundamentally an offline-on-a-window model.
We get true streaming by maintaining a rolling buffer plus a VAD-ish
silence detector; on every silence boundary or every `max_chunk_ms`, we
re-decode the active window and emit a partial. The terminal `is_final`
chunk is emitted when the upstream client closes the WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from .base import STTBackend, TranscriptionChunk


log = logging.getLogger(__name__)


# Moonshine was trained on 16kHz mono audio. Anything else gets resampled
# upstream (in audio.py) before reaching this backend.
TARGET_SAMPLE_RATE = 16_000

# Minimum audio length the Moonshine encoder is willing to look at. Padding
# below this just produces empty output; we skip the call entirely.
MIN_AUDIO_SAMPLES = 1_600  # 0.1s at 16kHz

# Default streaming knobs. Tunable via env without code changes.
DEFAULT_MAX_CHUNK_MS = int(os.environ.get("VOICE_STT_MAX_CHUNK_MS", "4000"))
DEFAULT_SILENCE_MS = int(os.environ.get("VOICE_STT_SILENCE_MS", "600"))
DEFAULT_SILENCE_RMS = float(os.environ.get("VOICE_STT_SILENCE_RMS", "0.005"))


class MoonshineBackend(STTBackend):
    """
    Wraps `moonshine-onnx`. Imports are deferred to warmup() so the rest of
    the server (and the test suite) can import this module without the
    upstream package present.
    """

    def __init__(
        self,
        *,
        model_name: str = "moonshine/base",
        max_chunk_ms: int = DEFAULT_MAX_CHUNK_MS,
        silence_ms: int = DEFAULT_SILENCE_MS,
        silence_rms: float = DEFAULT_SILENCE_RMS,
    ) -> None:
        self._model_name = os.environ.get("VOICE_MOONSHINE_MODEL", model_name)
        self._max_chunk_samples = max_chunk_ms * TARGET_SAMPLE_RATE // 1000
        self._silence_samples = silence_ms * TARGET_SAMPLE_RATE // 1000
        self._silence_rms = silence_rms
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    # ───────────────────────── lifecycle ─────────────────────────

    async def warmup(self) -> None:
        try:
            from moonshine_onnx import (  # type: ignore[import-not-found]
                MoonshineOnnxModel,
                load_tokenizer,
            )
        except ImportError as e:  # pragma: no cover — install-time failure
            raise RuntimeError(
                "moonshine-onnx is not installed. Install the lite-stt extra:\n"
                "    pip install -e '.[lite-stt]'"
            ) from e

        def _load() -> tuple[Any, Any]:
            return MoonshineOnnxModel(model_name=self._model_name), load_tokenizer()

        self._model, self._tokenizer = await asyncio.to_thread(_load)
        # One throwaway pass primes any JIT / ORT session warmups.
        primer = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)  # 1s of silence
        await asyncio.to_thread(self._decode, primer)
        self._loaded = True
        log.info("moonshine backend ready, model=%s", self._model_name)

    # ─────────────────────── public surface ──────────────────────

    async def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int
    ) -> TranscriptionChunk:
        self._require_loaded()
        audio = self._prepare_audio(audio_bytes, sample_rate)
        duration_ms = int(len(audio) * 1000 / TARGET_SAMPLE_RATE)
        text = await asyncio.to_thread(self._decode, audio)
        return TranscriptionChunk(
            text=text,
            is_final=True,
            start_ms=0,
            end_ms=duration_ms,
            confidence=None,
            language="en",
        )

    async def stream(
        self,
        frames: AsyncIterator[bytes],
        *,
        sample_rate: int,
    ) -> AsyncIterator[TranscriptionChunk]:
        self._require_loaded()
        buffer = np.zeros(0, dtype=np.float32)
        last_emitted_text = ""
        session_start = time.monotonic()
        silence_run = 0
        emitted_samples = 0

        def _flush_partial(samples: np.ndarray) -> TranscriptionChunk:
            text = self._decode(samples)
            now_ms = int((time.monotonic() - session_start) * 1000)
            return TranscriptionChunk(
                text=text,
                is_final=False,
                start_ms=int(emitted_samples * 1000 / TARGET_SAMPLE_RATE),
                end_ms=now_ms,
                confidence=None,
                language="en",
            )

        async for raw in frames:
            if not raw:
                continue
            chunk = self._prepare_audio(raw, sample_rate)
            buffer = np.concatenate([buffer, chunk])

            silence_run = self._update_silence_run(silence_run, chunk)

            should_emit = False
            if len(buffer) >= MIN_AUDIO_SAMPLES and (
                silence_run >= self._silence_samples
                or len(buffer) >= self._max_chunk_samples
            ):
                should_emit = True

            if should_emit:
                partial = await asyncio.to_thread(_flush_partial, buffer)
                if partial.text and partial.text != last_emitted_text:
                    last_emitted_text = partial.text
                    yield partial
                # Reset the rolling window on a silence boundary; on a
                # max-chunk boundary, also reset — Moonshine's window is
                # the unit of work, so we simply move on.
                emitted_samples += len(buffer)
                buffer = np.zeros(0, dtype=np.float32)
                silence_run = 0

        # Drain any audio still buffered on stream close.
        final_text = ""
        final_end_ms = int((time.monotonic() - session_start) * 1000)
        if len(buffer) >= MIN_AUDIO_SAMPLES:
            final_text = await asyncio.to_thread(self._decode, buffer)
        yield TranscriptionChunk(
            text=final_text or last_emitted_text,
            is_final=True,
            start_ms=int(emitted_samples * 1000 / TARGET_SAMPLE_RATE),
            end_ms=final_end_ms,
            confidence=None,
            language="en",
        )

    # ───────────────────── internal helpers ──────────────────────

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("MoonshineBackend.warmup() must be called first")

    def _prepare_audio(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        """Bytes (signed-16 LE) at any rate → float32 mono at 16kHz."""
        if not audio_bytes:
            return np.zeros(0, dtype=np.float32)
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != TARGET_SAMPLE_RATE:
            n_dst = int(round(len(arr) * TARGET_SAMPLE_RATE / sample_rate))
            if n_dst <= 0:
                return np.zeros(0, dtype=np.float32)
            x_src = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
            x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
            arr = np.interp(x_dst, x_src, arr).astype(np.float32)
        return arr

    def _update_silence_run(self, prev: int, chunk: np.ndarray) -> int:
        if len(chunk) == 0:
            return prev
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms < self._silence_rms:
            return prev + len(chunk)
        return 0

    def _decode(self, audio: np.ndarray) -> str:
        if len(audio) < MIN_AUDIO_SAMPLES:
            return ""
        # moonshine-onnx expects shape (1, T), float32, 16kHz.
        batch = audio[np.newaxis, :].astype(np.float32)
        tokens = self._model.generate(batch)
        decoded = self._tokenizer.decode_batch(tokens)
        if not decoded:
            return ""
        return str(decoded[0]).strip()
