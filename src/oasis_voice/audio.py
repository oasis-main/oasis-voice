"""
Audio plumbing — format conversion, framing, sentence splitting.

Kept tiny and dependency-light. soundfile + numpy do the heavy lifting for
PCM formats libsndfile understands natively (WAV, FLAC, Vorbis-in-OGG); we
deliberately avoid pulling in librosa/torchaudio because the cpu image
budget doesn't have room.

For codecs libsndfile doesn't ship (notably **Opus**, which Telegram voice
messages and most modern web audio use), we shell out to ffmpeg — which is
already an apt dep of the runtime image (Dockerfile.cpu line 14). The
ffmpeg path is a fallback, not the default: callers should pass
`mime`/`fileName` hints when known so we can route WAV through the fast
libsndfile path without spawning a subprocess.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
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


def encode_opus(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """
    Wrap raw PCM in an Opus-in-OGG container suitable for telegram
    `sendVoice` (and iMessage audio messages, browser MediaSource, etc.).

    Telegram only treats `.ogg/opus` audio as a true "voice note" — WAV
    is uploaded as a generic audio file. Doing the WAV→opus transcode
    server-side keeps the Node-side SpeechProvider plugin lean and
    consistent with the inbound path (`decode_audio_any` also routes
    non-WAV through ffmpeg).

    Bitrate: 24kbps mono is Telegram's preferred voice-note bitrate.
    For higher fidelity (e.g. music TTS), callers can encode WAV
    locally and bypass this helper.
    """
    if not _ffmpeg_available():
        raise AudioDecodeError(
            "ffmpeg not found on PATH — needed to encode opus output."
        )
    # Build a minimal in-memory WAV from the raw PCM so ffmpeg sees a
    # parseable container. We could also pipe raw `-f s16le` but the
    # WAV header is essentially free and lets ffmpeg sniff sample-rate
    # changes if a backend overrides chunk.sample_rate mid-stream.
    wav = encode_wav(pcm, sample_rate=sample_rate, channels=channels)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-b:a",
        "24k",
        "-ar",
        "48000",  # Opus is fixed at 48 kHz internally; resample once here.
        "-ac",
        str(channels),
        "-f",
        "ogg",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=wav,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioDecodeError(
            f"ffmpeg timed out encoding {len(pcm)} bytes of PCM"
        ) from e
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg opus encode exit {proc.returncode}: {stderr[:240]}")
    if not proc.stdout:
        raise AudioDecodeError("ffmpeg produced empty opus output")
    return proc.stdout


def decode_wav(data: bytes) -> tuple[bytes, int, int]:
    """
    WAV/FLAC/Vorbis-OGG (anything libsndfile reads natively) → (pcm16_bytes, sample_rate, channels).
    Always normalises to int16 PCM at the file's native rate.

    Raises whatever soundfile raises for unreadable input — callers that
    accept arbitrary user-uploaded audio should use `decode_audio_any`
    instead, which falls back to ffmpeg for Opus / AAC / WebM / etc.
    """
    arr, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=False)
    channels = 1 if arr.ndim == 1 else arr.shape[1]
    return arr.tobytes(), int(sr), channels


class AudioDecodeError(RuntimeError):
    """Raised when neither libsndfile nor ffmpeg can decode the input."""


# Container/codec MIMEs that libsndfile handles natively. Anything else
# routes through ffmpeg. Keep this list tight — when in doubt, let ffmpeg
# handle it (correct but slower).
_LIBSNDFILE_MIMES = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/flac",
        "audio/x-flac",
    }
)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_decode_to_pcm16(data: bytes, target_sr: int) -> tuple[bytes, int, int]:
    """
    Pipe `data` through ffmpeg → raw signed-16-bit-LE mono PCM at target_sr.

    ffmpeg auto-detects the container/codec; we do NOT pass `-f <container>`
    on the input. That lets the same call handle .ogg/opus from Telegram,
    .m4a from iMessage, .webm from a browser MediaRecorder, etc.
    """
    if not _ffmpeg_available():
        raise AudioDecodeError(
            "ffmpeg not found on PATH — needed to decode non-WAV/FLAC audio."
        )
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=data,
            capture_output=True,
            check=False,
            # Cap at 30s — large enough for a 5-minute Telegram voice note
            # at any reasonable bitrate, small enough that a malicious
            # zip-bomb-shaped audio file can't hang the server.
            timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioDecodeError(f"ffmpeg timed out on {len(data)}-byte input") from e
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg exit {proc.returncode}: {stderr[:240]}")
    pcm = proc.stdout
    if not pcm:
        raise AudioDecodeError("ffmpeg produced empty output (no audio decoded)")
    return pcm, target_sr, 1


def decode_audio_any(
    data: bytes,
    *,
    mime: str | None = None,
    file_name: str | None = None,
    target_sr: int = 16000,
) -> tuple[bytes, int, int]:
    """
    Decode any audio format to int16 PCM. Used by `/v1/stt/transcribe` so
    callers (Telegram voice notes, iMessage clips, web uploads) don't have
    to transcode client-side.

    Routing:
      1. If `mime` is a libsndfile-native type → try `decode_wav` first.
         Falls through to ffmpeg if libsndfile errors (defensive against
         mis-labelled MIMEs).
      2. Otherwise → ffmpeg, transcoded to PCM_16LE mono at target_sr.

    `file_name` is currently advisory only (logged on failure for triage);
    we don't sniff extensions because the MIME hint from the caller is
    more reliable in practice.
    """
    mime_norm = (mime or "").split(";")[0].strip().lower() or None
    if mime_norm in _LIBSNDFILE_MIMES:
        try:
            return decode_wav(data)
        except Exception:
            # Mis-labelled MIME; let ffmpeg try.
            pass
    return _ffmpeg_decode_to_pcm16(data, target_sr=target_sr)


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
