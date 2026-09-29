"""Download the public datasets used by the evaluation and record their SHA-256.

Datasets (all optional; the offline eval set in ``data/eval/`` is committed):

* LibriSpeech ``test-clean`` and ``test-other`` (CC BY 4.0, openslr.org/12) for WER.
* MultiWOZ 2.4 (MIT, github.com/smartyfh/MultiWOZ2.4) for dialogue realism checks.

Usage::

    python scripts/download_data.py --dataset librispeech-test-clean
    python scripts/download_data.py --dataset all --data-dir data/raw

Every archive is streamed to ``data/raw/``, hashed, extracted, and the hash is
appended to ``data/MANIFEST.txt``. Nothing is downloaded when the archive already
exists with the expected size unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from loguru import logger

MANIFEST = Path("data/MANIFEST.txt")


@dataclass(frozen=True)
class Dataset:
    """A downloadable archive and how to unpack it."""

    name: str
    url: str
    filename: str
    license: str
    archive: str  # "tar.gz" | "zip"


DATASETS: dict[str, Dataset] = {
    "librispeech-test-clean": Dataset(
        "librispeech-test-clean",
        "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "test-clean.tar.gz",
        "CC BY 4.0",
        "tar.gz",
    ),
    "librispeech-test-other": Dataset(
        "librispeech-test-other",
        "https://www.openslr.org/resources/12/test-other.tar.gz",
        "test-other.tar.gz",
        "CC BY 4.0",
        "tar.gz",
    ),
    "multiwoz-2.4": Dataset(
        "multiwoz-2.4",
        "https://github.com/smartyfh/MultiWOZ2.4/archive/refs/heads/main.zip",
        "MultiWOZ2.4-main.zip",
        "MIT",
        "zip",
    ),
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, client: httpx.Client, timeout_s: float = 60.0) -> Path:
    """Stream ``url`` to ``dest`` (atomic rename on completion).

    Raises:
        httpx.HTTPError: On transport failures or non-2xx responses.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url, timeout=timeout_s, follow_redirects=True) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def extract(archive: Path, kind: str, target: Path) -> None:
    """Extract a tar.gz or zip archive under ``target`` (paths are sanitised)."""
    target.mkdir(parents=True, exist_ok=True)
    if kind == "tar.gz":
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError(f"unsafe path in archive: {member.name}")
            tar.extractall(target)
    elif kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"unsafe path in archive: {name}")
            zf.extractall(target)
    else:
        raise ValueError(f"unknown archive kind {kind}")


def record_manifest(
    dataset: Dataset, archive: Path, digest: str, manifest: Path = MANIFEST
) -> None:
    """Append one line ``sha256  filename  dataset  license  timestamp`` to the manifest."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{digest}  {archive.name}  {dataset.name}  {dataset.license}  {archive.stat().st_size}  {stamp}\n"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    kept = [ln for ln in existing.splitlines() if f"  {archive.name}  " not in ln]
    manifest.write_text("\n".join(kept + [line.rstrip("\n")]) + "\n", encoding="utf-8")


def fetch(
    dataset: Dataset,
    data_dir: Path,
    client: httpx.Client,
    force: bool = False,
    manifest: Path = MANIFEST,
) -> str:
    """Download, hash, extract and record one dataset; returns the SHA-256."""
    archive = data_dir / dataset.filename
    if archive.exists() and not force:
        logger.info("{} already present, skipping download", archive)
    else:
        logger.info("downloading {} -> {}", dataset.url, archive)
        download(dataset.url, archive, client)
    digest = sha256_of(archive)
    extract(archive, dataset.archive, data_dir)
    record_manifest(dataset, archive, digest, manifest)
    logger.info("{} sha256={}", dataset.name, digest)
    return digest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    failures = 0
    with httpx.Client() as client:
        for name in names:
            try:
                fetch(DATASETS[name], args.data_dir, client, args.force)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                failures += 1
                logger.error("{} failed: {}", name, exc)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
