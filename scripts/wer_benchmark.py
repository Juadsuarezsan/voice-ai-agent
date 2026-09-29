"""Word error rate benchmark.

Two modes:

* Aligned text files (runs anywhere, used by the tests and ``make wer``)::

      python scripts/wer_benchmark.py --references refs.txt --hypotheses hyps.txt

* LibriSpeech directory transcribed with local Whisper (needs ``[stt]`` extra,
  ffmpeg and the dataset from ``scripts/download_data.py``)::

      python scripts/wer_benchmark.py --librispeech-dir data/raw/LibriSpeech/test-clean \\
          --whisper-model tiny --limit 200 --out eval/runs/2026-10-01-wer-tiny-clean.json

The output JSON carries the metrics, the normalisation and the source, so a
number can only land in ``eval/RESULTS.md`` with its provenance.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.wer import compute_wer, read_lines  # noqa: E402


def load_librispeech(root: Path, limit: int | None = None) -> list[tuple[Path, str]]:
    """Collect ``(flac_path, reference)`` pairs from a LibriSpeech split directory."""
    pairs: list[tuple[Path, str]] = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in read_lines(trans):
            utt_id, _, text = line.partition(" ")
            flac = trans.parent / f"{utt_id}.flac"
            if flac.exists():
                pairs.append((flac, text))
            if limit and len(pairs) >= limit:
                return pairs
    return pairs


def transcribe_with_whisper(
    pairs: list[tuple[Path, str]], model_name: str
) -> tuple[list[str], list[str], float]:
    """Transcribe every clip with local Whisper; returns refs, hyps and seconds spent."""
    whisper: Any = importlib.import_module("whisper")
    model = whisper.load_model(model_name)
    refs: list[str] = []
    hyps: list[str] = []
    t0 = time.perf_counter()
    for i, (path, ref) in enumerate(pairs, start=1):
        out = model.transcribe(str(path), language="en", fp16=False)
        refs.append(ref)
        hyps.append(str(out.get("text", "")))
        if i % 25 == 0:
            logger.info("transcribed {}/{}", i, len(pairs))
    return refs, hyps, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--references", type=Path)
    parser.add_argument("--hypotheses", type=Path)
    parser.add_argument("--librispeech-dir", type=Path)
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    source: dict[str, Any]
    seconds = 0.0
    if args.librispeech_dir:
        pairs = load_librispeech(args.librispeech_dir, args.limit)
        if not pairs:
            logger.error("no *.trans.txt/*.flac pairs under {}", args.librispeech_dir)
            return 1
        refs, hyps, seconds = transcribe_with_whisper(pairs, args.whisper_model)
        source = {
            "dataset": str(args.librispeech_dir),
            "model": f"whisper-{args.whisper_model}",
            "n": len(pairs),
        }
    elif args.references and args.hypotheses:
        refs, hyps = read_lines(args.references), read_lines(args.hypotheses)
        source = {
            "references": str(args.references),
            "hypotheses": str(args.hypotheses),
            "fixture": True,
        }
    else:
        parser.error("pass --references/--hypotheses or --librispeech-dir")

    report = compute_wer(refs, hyps)
    payload = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "normalisation": "lowercase, strip punctuation, collapse spaces (jiwer)",
        "transcription_seconds": round(seconds, 2),
        "metrics": report.model_dump(),
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        logger.info("wrote {}", args.out)
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
