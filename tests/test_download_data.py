"""download_data.py against mocked HTTP (no network)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import respx


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_data", "scripts/download_data.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _targz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


@respx.mock
def test_fetch_librispeech_records_manifest(tmp_path: Path) -> None:
    mod = _script()
    payload = _targz({"LibriSpeech/test-clean/1/2/1-2.trans.txt": b"1-2-0000 HELLO\n"})
    respx.get(mod.DATASETS["librispeech-test-clean"].url).mock(
        return_value=httpx.Response(200, content=payload)
    )
    manifest = tmp_path / "MANIFEST.txt"
    with httpx.Client() as client:
        digest = mod.fetch(
            mod.DATASETS["librispeech-test-clean"], tmp_path / "raw", client, manifest=manifest
        )
    assert digest == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / "raw" / "LibriSpeech" / "test-clean" / "1" / "2" / "1-2.trans.txt").exists()
    line = manifest.read_text().strip()
    assert line.startswith(digest) and "librispeech-test-clean" in line and "CC BY 4.0" in line


@respx.mock
def test_fetch_zip_and_manifest_dedup(tmp_path: Path) -> None:
    mod = _script()
    ds = mod.DATASETS["multiwoz-2.4"]
    respx.get(ds.url).mock(
        return_value=httpx.Response(200, content=_zip({"MultiWOZ2.4-main/README.md": b"hi"}))
    )
    manifest = tmp_path / "MANIFEST.txt"
    with httpx.Client() as client:
        mod.fetch(ds, tmp_path / "raw", client, manifest=manifest)
        mod.fetch(ds, tmp_path / "raw", client, force=True, manifest=manifest)
    assert len(manifest.read_text().strip().splitlines()) == 1


@respx.mock
def test_skip_existing_archive(tmp_path: Path) -> None:
    mod = _script()
    ds = mod.DATASETS["librispeech-test-other"]
    route = respx.get(ds.url).mock(
        return_value=httpx.Response(200, content=_targz({"x/a.txt": b"a"}))
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / ds.filename).write_bytes(_targz({"x/a.txt": b"a"}))
    with httpx.Client() as client:
        mod.fetch(ds, tmp_path / "raw", client, manifest=tmp_path / "m.txt")
    assert route.call_count == 0


@respx.mock
def test_http_error_returns_nonzero(tmp_path: Path) -> None:
    mod = _script()
    for ds in mod.DATASETS.values():
        respx.get(ds.url).mock(return_value=httpx.Response(403))
    assert mod.main(["--dataset", "librispeech-test-clean", "--data-dir", str(tmp_path)]) == 1


def test_unsafe_archive_paths_rejected(tmp_path: Path) -> None:
    mod = _script()
    evil = tmp_path / "evil.tar.gz"
    evil.write_bytes(_targz({"../escape.txt": b"x"}))
    with pytest.raises(ValueError, match="unsafe"):
        mod.extract(evil, "tar.gz", tmp_path / "out")
