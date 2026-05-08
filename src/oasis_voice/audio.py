"""
Audio plumbing — format conversion, framing, sentence splitting.

Kept tiny and dependency-light. soundfile + numpy do the heavy lifting; we
deliberately avoid pulling in librosa/torchaudio because the cpu image
budget doesn't have room.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator

import numpy as np
import soundfile as sf


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """
    Cheap sentence splitter for TTS streaming. Good enough for English prose;
    backends that want smarter segmentation can override.
    """
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Bytes (signed 16-bit LE) → float32 in [-1, 1]."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


def float32_to_pcm16(arr: np.ndarray) -> bytes:
    """float32 in [-1, 1] → signed 16-bit LE bytes. Clipped, not dithered."""
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def encode_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw PCM in a WAV container so /v1/tts/speak can return audio/wav."""
    buf = io.BytesIO()
    arr = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        arr = arr.reshape(-1, channels)
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def decode_wav(data: bytes) -> tuple[bytes, int, int]:
    """
    WAV/FLAC/OGG (anything libsndfile reads) → (pcm16_bytes, sample_rate, channels).
    Always normalises to int16 PCM at the file's native rate.
    """
    arr, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=False)
    channels = 1 if arr.ndim == 1 else arr.shape[1]
    return arr.tobytes(), int(sr), channels


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """
    Linear resample int16 → int16. Naive but ample for telephony (8k↔16k).
    For higher-quality conversions, defer to the backend's own resampler.
    """
    if src_rate == dst_rate or not pcm:
        return pcm
    src = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_dst = int(round(len(src) * dst_rate / src_rate))
    if n_dst <= 0:
        return b""
    x_src = np.linspace(0.0, 1.0, num=len(src), endpoint=False)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    dst = np.interp(x_dst, x_src, src)
    return dst.astype(np.int16).tobytes()


def frame_iterator(
    pcm: bytes, sample_rate: int, frame_ms: int = 20
) -> Iterator[bytes]:
    """Slice a PCM buffer into fixed-duration frames for WS streaming."""
    bytes_per_frame = (sample_rate * frame_ms // 1000) * 2  # int16 mono
    if bytes_per_frame <= 0:
        yield pcm
        return
    for i in range(0, len(pcm), bytes_per_frame):
        chunk = pcm[i : i + bytes_per_frame]
        if chunk:
            yield chunk
