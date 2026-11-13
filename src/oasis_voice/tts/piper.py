"""
Piper TTS backend — `tier=lite`.

Piper is MIT-licensed, runs on CPU, and streams sentence-by-sentence with
sub-200ms TTFA on commodity hardware. Voices are .onnx + .json pairs from
`rhasspy/piper-voices` on Hugging Face.

Resolution order for the model file, given a VoiceRef like
`piper:en_US-lessac-high`:

  1. $VOICE_PIPER_VOICE_DIR/<voice>.onnx              (operator override)
  2. $XDG_DATA_HOME/oasis-voice/piper/<voice>.onnx    (download_weights default)
  3. ~/.local/share/oasis-voice/piper/<voice>.onnx    (XDG fallback)

The default voice (when the caller passes the bare tier name `piper:default`
or omits `voice`) is read from the active TierSpec — `en_US-lessac-high`
out of the box, overrideable via $VOICE_PIPER_DEFAULT_VOICE.

Streaming model: Piper synthesises one sentence at a time and we yield the
PCM as soon as each sentence is rendered. That's the meaningful TTFA win —
not chunking inside a single sentence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..audio import split_sentences
from .base import AudioChunk, TTSBackend, VoicePreset, VoiceRef


log = logging.getLogger(__name__)


def _voice_search_paths() -> list[Path]:
    """Directories in priority order where Piper voices may live."""
    out: list[Path] = []
    override = os.environ.get("VOICE_PIPER_VOICE_DIR")
    if override:
        out.append(Path(override).expanduser())
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        out.append(Path(xdg).expanduser() / "oasis-voice" / "piper")
    out.append(Path.home() / ".local" / "share" / "oasis-voice" / "piper")
    return out


def resolve_voice_path(voice_id: str) -> Path:
    """
    Map `piper:en_US-lessac-high` → /…/en_US-lessac-high.onnx. Raises
    FileNotFoundError with a clear remediation hint if no candidate exists.
    """
    name = voice_id.split(":", 1)[-1]
    candidates: list[Path] = []
    for d in _voice_search_paths():
        candidates.append(d / f"{name}.onnx")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Piper voice '{name}' not found. Searched: "
        f"{[str(p) for p in candidates]}. "
        f"Run `python -m oasis_voice.scripts.download_weights piper:{name}` first."
    )


class PiperBackend(TTSBackend):
    """
    Wrapper around the upstream `piper-tts` Python package. Imports are
    deferred to warmup() so the rest of the server (and the test suite) can
    boot without piper installed.
    """

    supports_cloning = False
    supports_streaming = True

    def __init__(
        self,
        *,
        default_voice: str | None = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        self._default_voice = (
            default_voice
            or os.environ.get("VOICE_PIPER_DEFAULT_VOICE")
            or "en_US-lessac-high"
        )
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w = noise_w
        self._voices: dict[str, Any] = {}
        self._speaker_maps: dict[str, dict[str, int]] = {}
        self._PiperVoice: Any = None
        self._SynthesisConfig: Any = None

    # ───────────────────────── lifecycle ─────────────────────────

    async def warmup(self) -> None:
        try:
            from piper import PiperVoice  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover — install-time failure
            raise RuntimeError(
                "piper-tts is not installed. Install the lite-tts extra:\n"
                "    pip install -e '.[lite-tts]'"
            ) from e
        self._PiperVoice = PiperVoice
        try:
            from piper import SynthesisConfig  # type: ignore[import-not-found]
            self._SynthesisConfig = SynthesisConfig
        except ImportError:
            self._SynthesisConfig = None  # older piper-tts releases
        # Pre-load the default voice so the first request isn't slow.
        await asyncio.to_thread(self._ensure_voice, self._default_voice)
        log.info("piper backend ready, default voice=%s", self._default_voice)

    # ─────────────────────── public surface ──────────────────────

    async def speak(self, text: str, voice: VoiceRef) -> AudioChunk:
        if voice.kind != "preset":
            raise ValueError("PiperBackend only supports preset voices")
        chunks: list[AudioChunk] = []
        async for chunk in self.stream(text, voice):
            chunks.append(chunk)
        if not chunks:
            return AudioChunk(pcm=b"", sample_rate=22050, channels=1, is_final=True)
        sample_rate = chunks[0].sample_rate
        pcm = b"".join(c.pcm for c in chunks)
        return AudioChunk(pcm=pcm, sample_rate=sample_rate, channels=1, is_final=True)

    async def stream(
        self, text: str, voice: VoiceRef
    ) -> AsyncIterator[AudioChunk]:
        if voice.kind != "preset":
            raise ValueError("PiperBackend only supports preset voices")
        sentences = split_sentences(text)
        if not sentences:
            yield AudioChunk(pcm=b"", sample_rate=22050, channels=1, is_final=True)
            return

        voice_id = voice.voice_id or self._default_voice
        piper_voice = await asyncio.to_thread(self._ensure_voice, voice_id)
        sample_rate = self._sample_rate_of(piper_voice)
        speaker_id = self._resolve_speaker_id(voice_id, voice.speaker)

        for idx, sentence in enumerate(sentences):
            is_last = idx == len(sentences) - 1
            pcm = await asyncio.to_thread(
                self._synthesize_one, piper_voice, sentence, speaker_id
            )
            yield AudioChunk(
                pcm=pcm,
                sample_rate=sample_rate,
                channels=1,
                is_final=is_last,
            )

    def list_presets(self) -> list[VoicePreset]:
        """
        Walk the voice search paths and report every installed .onnx.

        Filesystem only — deliberately does NOT call warmup() or load a model,
        because listing must not cost a cold voice download (see the contract on
        TTSBackend.list_presets). Speaker names come from the JSON sidecar that
        already sits next to every Piper model, so a multi-speaker voice reports
        its selectable speakers without instantiating anything.

        Earlier search paths win, matching resolve_voice_path's precedence, so a
        voice shadowed by a higher-priority directory is reported once.
        """
        seen: dict[str, VoicePreset] = {}
        for d in _voice_search_paths():
            try:
                if not d.is_dir():
                    continue
                entries = sorted(d.glob("*.onnx"))
            except OSError as e:  # unreadable mount, bad symlink
                log.warning("cannot list piper voices in %s: %s", d, e)
                continue
            for onnx in entries:
                name = onnx.stem
                if name in seen:
                    continue  # higher-priority directory already provided it
                seen[name] = VoicePreset(
                    voice_id=f"piper:{name}",
                    speakers=self._peek_speakers(onnx),
                )
        return [seen[k] for k in sorted(seen)]

    def _peek_speakers(self, model_path: Path) -> tuple[str, ...]:
        """Read speaker names from the model's JSON sidecar without loading it."""
        json_path = Path(str(model_path) + ".json")
        try:
            with open(json_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            return ()
        sid_map = cfg.get("speaker_id_map")
        if not isinstance(sid_map, dict) or len(sid_map) <= 1:
            # A single-entry map is a single-speaker model; report no choice
            # rather than one meaningless option.
            return ()
        return tuple(sorted(sid_map))

    # ───────────────────── internal helpers ──────────────────────

    def _ensure_voice(self, voice_id: str) -> Any:
        if self._PiperVoice is None:
            raise RuntimeError("PiperBackend.warmup() must be called first")
        name = voice_id.split(":", 1)[-1]
        if name in self._voices:
            return self._voices[name]
        path = resolve_voice_path(name)
        log.info("loading piper voice from %s", path)
        loaded = self._PiperVoice.load(str(path))
        self._voices[name] = loaded
        self._load_speaker_map(name, path)
        return loaded

    def _load_speaker_map(self, name: str, model_path: Path) -> None:
        """Load speaker_id_map from the model's JSON config sidecar."""
        if name in self._speaker_maps:
            return
        json_path = Path(str(model_path) + ".json")
        if not json_path.exists():
            return
        try:
            with open(json_path) as f:
                cfg = json.load(f)
            sid_map = cfg.get("speaker_id_map")
            if isinstance(sid_map, dict) and sid_map:
                self._speaker_maps[name] = sid_map
                log.info("loaded %d speakers for %s", len(sid_map), name)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("failed to read speaker map from %s: %s", json_path, e)

    def _resolve_speaker_id(self, voice_id: str, speaker: str | None) -> int | None:
        if speaker is None:
            return None
        name = voice_id.split(":", 1)[-1]
        sid_map = self._speaker_maps.get(name)
        if sid_map and speaker in sid_map:
            return sid_map[speaker]
        try:
            return int(speaker)
        except ValueError:
            available = list((sid_map or {}).keys())[:10]
            raise ValueError(
                f"Unknown speaker '{speaker}' for voice '{name}'. "
                f"Available: {available}"
            )

    def _synthesize_one(self, piper_voice: Any, sentence: str, speaker_id: int | None = None) -> bytes:
        """
        Render one sentence to int16 PCM bytes. Piper exposes both a
        chunk-iterator API (modern) and a `synthesize_stream_raw` (older);
        we accommodate both so the install-time API drift doesn't break us.
        """
        # Modern API: voice.synthesize(text) → Iterable[AudioChunk-like]
        if hasattr(piper_voice, "synthesize"):
            kwargs: dict[str, Any] = {}
            if self._SynthesisConfig is not None:
                syn_kwargs: dict[str, Any] = {
                    "length_scale": self._length_scale,
                    "noise_scale": self._noise_scale,
                    "noise_w_scale": self._noise_w,
                }
                if speaker_id is not None:
                    syn_kwargs["speaker_id"] = speaker_id
                kwargs["syn_config"] = self._SynthesisConfig(**syn_kwargs)
            buf = bytearray()
            for chunk in piper_voice.synthesize(sentence, **kwargs):
                # Newer chunks have `.audio_int16_bytes`; older return raw bytes.
                if hasattr(chunk, "audio_int16_bytes"):
                    buf.extend(chunk.audio_int16_bytes)
                elif isinstance(chunk, (bytes, bytearray)):
                    buf.extend(chunk)
                else:  # pragma: no cover — unknown variant
                    raise TypeError(
                        f"unexpected piper chunk type: {type(chunk).__name__}"
                    )
            return bytes(buf)
        # Legacy API: voice.synthesize_stream_raw(text) → Iterable[bytes]
        if hasattr(piper_voice, "synthesize_stream_raw"):
            return b"".join(piper_voice.synthesize_stream_raw(sentence))
        raise RuntimeError(
            "Loaded piper voice exposes neither synthesize() nor "
            "synthesize_stream_raw(); piper-tts API has changed."
        )

    @property
    def speaker_maps(self) -> dict[str, dict[str, int]]:
        return self._speaker_maps

    @staticmethod
    def _sample_rate_of(piper_voice: Any) -> int:
        cfg = getattr(piper_voice, "config", None)
        sr = getattr(cfg, "sample_rate", None) if cfg is not None else None
        return int(sr) if sr else 22050
