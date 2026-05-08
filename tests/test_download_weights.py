"""
Weight downloader tests — exercise sha256 verification and target shaping
without touching the network.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pytest

from oasis_voice.scripts import download_weights as dw


class FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._read = False

    def read(self, _n: int) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_WEIGHTS_DIR", str(tmp_path))
    return tmp_path


def test_build_targets_piper_voice(tmp_data_root):
    kind, targets = dw.build_targets("piper:en_US-lessac-high")
    assert kind == "piper"
    assert len(targets) == 2
    assert targets[0].dest.name == "en_US-lessac-high.onnx"
    assert targets[1].dest.name == "en_US-lessac-high.onnx.json"
    assert "rhasspy/piper-voices" in targets[0].url


def test_build_targets_unknown_voice():
    with pytest.raises(SystemExit, match="Unknown Piper voice"):
        dw.build_targets("piper:nonexistent")


def test_build_targets_lite_stt(tmp_data_root):
    kind, targets = dw.build_targets("lite-stt")
    assert kind == "moonshine"
    names = {t.dest.name for t in targets}
    assert "encode.onnx" in names
    assert "tokenizer.json" in names


def test_download_one_writes_and_verifies(tmp_path, monkeypatch):
    payload = b"hello weights"
    expected = hashlib.sha256(payload).hexdigest()

    def fake_urlopen(url, *a, **kw):
        return FakeResp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = tmp_path / "weights.bin"
    target = dw.FetchTarget(url="https://example/x", dest=dest, sha256=expected)
    actual = dw.download_one(target)
    assert actual == expected
    assert dest.read_bytes() == payload


def test_download_one_rejects_mismatch(tmp_path, monkeypatch):
    payload = b"corrupt payload"

    def fake_urlopen(url, *a, **kw):
        return FakeResp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = tmp_path / "weights.bin"
    target = dw.FetchTarget(
        url="https://example/x",
        dest=dest,
        sha256="0" * 64,  # deliberately wrong
    )
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        dw.download_one(target)
    # On rejection the partial must be cleaned up.
    assert not dest.exists()
    assert not dest.with_suffix(".bin.part").exists()


def test_download_one_unpinned_prints_hash(tmp_path, monkeypatch, capsys):
    payload = b"fresh weight to pin"

    def fake_urlopen(url, *a, **kw):
        return FakeResp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = tmp_path / "weights.bin"
    target = dw.FetchTarget(url="https://example/x", dest=dest, sha256=None)
    actual = dw.download_one(target)
    assert actual == hashlib.sha256(payload).hexdigest()
    out = capsys.readouterr().out
    assert "pinned-hash=NONE" in out
    assert actual in out


def test_main_hash_helper(tmp_path, capsys):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    rc = dw.main(["--hash", str(f)])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == hashlib.sha256(b"abc").hexdigest()
