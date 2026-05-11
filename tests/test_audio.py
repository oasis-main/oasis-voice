"""Audio utility unit tests."""

from __future__ import annotations

import shutil
import subprocess
from unittest import mock

import numpy as np
import pytest

from oasis_voice.audio import (
    AudioDecodeError,
    decode_audio_any,
    decode_wav,
    encode_wav,
    float32_to_pcm16,
    frame_iterator,
    pcm16_to_float32,
    resample_pcm16,
    split_sentences,
)


HAVE_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG,
    reason="ffmpeg not on PATH (installed in oasis-voice CPU image; skipped on bare hosts)",
)


def test_split_sentences_basic():
    out = split_sentences("Hello world. This is a test! Is it? Yes.")
    assert out == ["Hello world.", "This is a test!", "Is it?", "Yes."]


def test_split_sentences_empty_or_no_terminator():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("no terminator") == ["no terminator"]


def test_pcm_roundtrip():
    src = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    pcm = float32_to_pcm16(src)
    back = pcm16_to_float32(pcm)
    assert back.dtype == np.float32
    assert np.allclose(src, back, atol=1e-3)


def test_wav_roundtrip():
    sr = 16000
    samples = np.sin(np.linspace(0, 2 * np.pi, sr, dtype=np.float32)) * 0.5
    pcm = float32_to_pcm16(samples)
    wav = encode_wav(pcm, sample_rate=sr)
    back, back_sr, channels = decode_wav(wav)
    assert back_sr == sr
    assert channels == 1
    assert len(back) == len(pcm)


def test_resample_passthrough():
    pcm = b"\x01\x00" * 100
    assert resample_pcm16(pcm, 16000, 16000) == pcm


def test_resample_changes_length():
    pcm = float32_to_pcm16(np.zeros(16000, dtype=np.float32))
    out = resample_pcm16(pcm, 16000, 8000)
    # 8kHz at int16 mono = 2 bytes/sample; 1s of audio → 16000 bytes.
    assert len(out) == pytest.approx(16000, rel=0.01)


def test_frame_iterator_yields_fixed_size():
    pcm = b"\x00\x00" * 16000  # 1s at 16kHz int16 mono
    frames = list(frame_iterator(pcm, sample_rate=16000, frame_ms=20))
    # 50 frames of 20ms at 16kHz = 320 samples * 2 bytes = 640 bytes each
    assert len(frames) == 50
    assert all(len(f) == 640 for f in frames)


# ─────────────────────── decode_audio_any ────────────────────────────────


def test_decode_audio_any_wav_takes_libsndfile_path():
    """audio/wav with valid bytes → libsndfile (no ffmpeg spawned)."""
    sr = 16000
    samples = np.sin(np.linspace(0, 2 * np.pi, sr, dtype=np.float32)) * 0.5
    pcm_in = float32_to_pcm16(samples)
    wav = encode_wav(pcm_in, sample_rate=sr)
    # Mock the ffmpeg path so we'd notice if it accidentally got called.
    with mock.patch("oasis_voice.audio._ffmpeg_decode_to_pcm16") as mocked:
        pcm, out_sr, ch = decode_audio_any(wav, mime="audio/wav")
        mocked.assert_not_called()
    assert out_sr == sr
    assert ch == 1
    assert len(pcm) == len(pcm_in)


def test_decode_audio_any_mislabelled_mime_falls_through_to_ffmpeg():
    """audio/wav header lies — libsndfile fails, ffmpeg gets a chance."""
    bogus = b"NOT-A-WAV" + b"\x00" * 100
    with mock.patch(
        "oasis_voice.audio._ffmpeg_decode_to_pcm16",
        return_value=(b"\x00\x00" * 8, 16000, 1),
    ) as mocked:
        pcm, sr, ch = decode_audio_any(bogus, mime="audio/wav")
        mocked.assert_called_once()
    assert sr == 16000
    assert ch == 1


def test_decode_audio_any_unknown_mime_routes_to_ffmpeg():
    """audio/opus (libsndfile can't read) → ffmpeg path."""
    bogus = b"\x00" * 32
    with mock.patch(
        "oasis_voice.audio._ffmpeg_decode_to_pcm16",
        return_value=(b"\x01\x00" * 16, 16000, 1),
    ) as mocked:
        decode_audio_any(bogus, mime="audio/opus")
        mocked.assert_called_once()
        # Verify we ask ffmpeg to produce 16kHz mono PCM, not whatever
        # the input rate happens to be. The whole point of having ffmpeg
        # in the path is that it normalises for us.
        call_kwargs = mocked.call_args.kwargs
        assert call_kwargs.get("target_sr") == 16000


def test_decode_audio_any_no_mime_routes_to_ffmpeg():
    """No mime hint → don't guess; let ffmpeg sniff."""
    bogus = b"\x00" * 32
    with mock.patch(
        "oasis_voice.audio._ffmpeg_decode_to_pcm16",
        return_value=(b"\x01\x00" * 4, 16000, 1),
    ) as mocked:
        decode_audio_any(bogus, mime=None)
        mocked.assert_called_once()


def test_decode_audio_any_raises_when_ffmpeg_missing(monkeypatch):
    """Bare host without ffmpeg AND non-WAV input → AudioDecodeError."""
    monkeypatch.setattr("oasis_voice.audio.shutil.which", lambda _: None)
    with pytest.raises(AudioDecodeError, match="ffmpeg not found"):
        decode_audio_any(b"opus-bytes", mime="audio/opus")


def test_decode_audio_any_propagates_ffmpeg_failure(monkeypatch):
    """ffmpeg exits non-zero → AudioDecodeError with stderr context."""
    monkeypatch.setattr("oasis_voice.audio.shutil.which", lambda _: "/usr/bin/ffmpeg")

    fake_result = subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=1, stdout=b"", stderr=b"Invalid data found"
    )
    monkeypatch.setattr(
        "oasis_voice.audio.subprocess.run",
        lambda *_a, **_kw: fake_result,
    )
    with pytest.raises(AudioDecodeError, match="ffmpeg exit 1.*Invalid data found"):
        decode_audio_any(b"garbage", mime="audio/opus")


def test_decode_audio_any_handles_ffmpeg_timeout(monkeypatch):
    """Zip-bomb-shaped input → TimeoutExpired surfaces as AudioDecodeError."""
    monkeypatch.setattr("oasis_voice.audio.shutil.which", lambda _: "/usr/bin/ffmpeg")

    def _raise_timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

    monkeypatch.setattr("oasis_voice.audio.subprocess.run", _raise_timeout)
    with pytest.raises(AudioDecodeError, match="timed out"):
        decode_audio_any(b"x" * 1000, mime="audio/opus")


@needs_ffmpeg
def test_decode_audio_any_opus_roundtrip_via_real_ffmpeg(tmp_path):
    """
    End-to-end with a REAL ffmpeg: synthesize 0.5s of silence, encode to
    Opus-in-OGG via ffmpeg, decode via decode_audio_any, verify the result
    is PCM16 at 16kHz. This is the test that runs inside the oasis-voice
    CPU image (ffmpeg in apt layer) and is skipped on bare hosts.
    """
    # Generate WAV first, then transcode to opus, then feed to decode_audio_any.
    sr = 16000
    silence = float32_to_pcm16(np.zeros(sr // 2, dtype=np.float32))
    wav_bytes = encode_wav(silence, sample_rate=sr)
    opus_path = tmp_path / "silence.opus"
    res = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            "16k",
            str(opus_path),
        ],
        input=wav_bytes,
        capture_output=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr.decode("utf-8", errors="replace")
    opus_bytes = opus_path.read_bytes()
    assert len(opus_bytes) > 0

    pcm, out_sr, ch = decode_audio_any(opus_bytes, mime="audio/opus")
    assert out_sr == 16000
    assert ch == 1
    # 0.5s at 16kHz int16 mono → 16000 bytes. ffmpeg framing rounds a bit,
    # so just assert we got most of it.
    assert len(pcm) >= 14000
