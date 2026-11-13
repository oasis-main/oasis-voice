"""
Per-tier weight fetcher with sha256 pinning.

Usage:

    # Download every file for a single tier
    python -m oasis_voice.scripts.download_weights lite-tts
    python -m oasis_voice.scripts.download_weights lite-stt

    # Download a specific Piper voice
    python -m oasis_voice.scripts.download_weights piper:en_US-lessac-high

    # Recompute and print sha256 for a fetched file (helper for pinning new
    # weights). Does NOT mutate `tiers.py`; you do that by hand.
    python -m oasis_voice.scripts.download_weights --hash <path>

The contract is deliberately loud: any file whose sha256 doesn't match the
pinned value is deleted on disk and the script exits non-zero. This is the
same defense the sherpa-onnx-tts audit asked for, applied uniformly.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..tiers import STT_TIERS, TTS_TIERS, TierSpec


HF_BASE = "https://huggingface.co"


@dataclass(frozen=True)
class FetchTarget:
    url: str
    dest: Path
    sha256: str | None  # None means "not yet pinned" — we'll fetch and print


# ─────────────────────────── Piper voice catalog ───────────────────────────
#
# Piper voices live in the rhasspy/piper-voices repo on HF, structured as:
#   <lang>/<voice>/<quality>/<voice>-<quality>.onnx
#   <lang>/<voice>/<quality>/<voice>-<quality>.onnx.json
# Mapping the catalog name (`en_US-lessac-high`) to the path requires a
# small lookup table; we keep only what we actually ship.

PIPER_VOICE_CATALOG: dict[str, str] = {
    "en_US-lessac-high":   "en/en_US/lessac/high/en_US-lessac-high",
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium",
    "en_US-amy-medium":    "en/en_US/amy/medium/en_US-amy-medium",
    "en_GB-alan-medium":   "en/en_GB/alan/medium/en_GB-alan-medium",
    "en_GB-aru-medium":    "en/en_GB/aru/medium/en_GB-aru-medium",
    "en_GB-vctk-medium":   "en/en_GB/vctk/medium/en_GB-vctk-medium",
}


def data_root() -> Path:
    """Where downloaded weights live. Honors XDG and the docker bind-mount."""
    override = os.environ.get("VOICE_WEIGHTS_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "oasis-voice"
    return Path.home() / ".local" / "share" / "oasis-voice"


# ─────────────────────────── target builders ───────────────────────────


def _hf_resolve(repo: str, revision: str, filename: str) -> str:
    rev = revision or "main"
    return f"{HF_BASE}/{repo}/resolve/{rev}/{filename}"


def _piper_targets(voice: str, spec: TierSpec) -> list[FetchTarget]:
    if voice not in PIPER_VOICE_CATALOG:
        raise SystemExit(
            f"Unknown Piper voice '{voice}'. Known: {sorted(PIPER_VOICE_CATALOG)}.\n"
            f"Add it to PIPER_VOICE_CATALOG in scripts/download_weights.py."
        )
    base = PIPER_VOICE_CATALOG[voice]
    out_dir = data_root() / "piper"
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = "rhasspy/piper-voices"
    rev = spec.revision if spec.revision and spec.revision != "TBD" else "main"
    return [
        FetchTarget(
            url=_hf_resolve(repo, rev, f"{base}.onnx"),
            dest=out_dir / f"{voice}.onnx",
            sha256=spec.sha256.get(f"{voice}.onnx"),
        ),
        FetchTarget(
            url=_hf_resolve(repo, rev, f"{base}.onnx.json"),
            dest=out_dir / f"{voice}.onnx.json",
            sha256=spec.sha256.get(f"{voice}.onnx.json"),
        ),
    ]


def _moonshine_targets(spec: TierSpec) -> list[FetchTarget]:
    """
    moonshine-onnx fetches its own weights from `UsefulSensors/moonshine-base`
    on first model load, into ~/.cache/huggingface/hub. This script's job is
    to pre-warm that cache and verify hashes before serving traffic.

    The model directory contains four ONNX files (preprocess / encoder /
    uncached_decoder / cached_decoder) plus the tokenizer.json.
    """
    out_dir = data_root() / "moonshine" / spec.weights.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    rev = spec.revision if spec.revision and spec.revision != "TBD" else "main"
    files = [
        "preprocess.onnx",
        "encode.onnx",
        "uncached_decode.onnx",
        "cached_decode.onnx",
        "tokenizer.json",
    ]
    return [
        FetchTarget(
            url=_hf_resolve(spec.weights, rev, name),
            dest=out_dir / name,
            sha256=spec.sha256.get(name),
        )
        for name in files
    ]


def build_targets(selector: str) -> tuple[str, list[FetchTarget]]:
    """
    `selector` is one of:
      - `lite-tts` / `lite-stt`        — entire tier
      - `piper:<voice>`                — one Piper voice
      - `moonshine`                    — Moonshine Base STT
    """
    if selector.startswith("piper:"):
        return "piper", _piper_targets(selector.split(":", 1)[1], TTS_TIERS["lite"])
    if selector in ("lite-tts", "piper"):
        default_voice = os.environ.get(
            "VOICE_PIPER_DEFAULT_VOICE", "en_US-lessac-high"
        )
        return "piper", _piper_targets(default_voice, TTS_TIERS["lite"])
    if selector in ("lite-stt", "moonshine"):
        return "moonshine", _moonshine_targets(STT_TIERS["lite"])
    raise SystemExit(
        f"Unknown selector '{selector}'. Try: lite-tts, lite-stt, "
        f"piper:<voice>, or moonshine."
    )


# ───────────────────────────── core fetch ─────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_one(target: FetchTarget) -> str:
    """Returns the actual on-disk sha256 (so caller can verify or pin)."""
    if target.dest.exists() and target.sha256 and sha256_file(target.dest) == target.sha256:
        print(f"  ✓ {target.dest.name} (cached, hash matches)")
        return target.sha256
    print(f"  ↓ {target.url}")
    tmp = target.dest.with_suffix(target.dest.suffix + ".part")
    with urllib.request.urlopen(target.url) as resp, tmp.open("wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
    actual = sha256_file(tmp)
    if target.sha256 and actual != target.sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"sha256 mismatch for {target.dest.name}:\n"
            f"  expected: {target.sha256}\n"
            f"  actual:   {actual}\n"
            f"  Refusing to install an unverified weight file."
        )
    tmp.replace(target.dest)
    if target.sha256:
        print(f"  ✓ {target.dest.name} (hash matches)")
    else:
        print(f"  ⚠ {target.dest.name} pinned-hash=NONE — record this:\n"
              f"      {target.dest.name}: {actual}")
    return actual


def fetch(selector: str) -> None:
    kind, targets = build_targets(selector)
    print(f"Fetching {kind} weights for selector={selector!r}")
    print(f"Destination: {data_root()}")
    for t in targets:
        download_one(t)


# ─────────────────────────── CLI ───────────────────────────


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="oasis-voice-download")
    p.add_argument(
        "selector",
        nargs="?",
        help="lite-tts, lite-stt, piper:<voice>, or moonshine",
    )
    p.add_argument(
        "--hash",
        type=Path,
        help="Print the sha256 of an existing file and exit (pinning helper)",
    )
    return p.parse_args(list(argv))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.hash is not None:
        if not args.hash.exists():
            print(f"file not found: {args.hash}", file=sys.stderr)
            return 2
        print(sha256_file(args.hash))
        return 0
    if not args.selector:
        print("error: selector required (e.g. lite-tts, lite-stt)", file=sys.stderr)
        return 2
    fetch(args.selector)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
