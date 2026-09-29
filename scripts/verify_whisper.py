"""Verify that the local Whisper STT path works on this machine.

Run after ``pip install -e ".[stt]"``::

    python scripts/verify_whisper.py --model tiny

Loads the model through :class:`src.stt.whisper_wrapper.LocalWhisperSTT`,
transcribes one second of silence (must come back empty: no hallucination) and
reports load and inference times. It is a smoke test, not a WER benchmark
(see ``scripts/wer_benchmark.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.vad import encode_wav_b64  # noqa: E402
from src.stt.whisper_wrapper import LocalWhisperSTT  # noqa: E402


async def run(model_name: str) -> int:
    """Load the model and transcribe silence; returns a process exit code."""
    t0 = time.perf_counter()
    try:
        stt = LocalWhisperSTT(model_name)
    except ImportError:
        logger.error('openai-whisper is not installed: pip install -e ".[stt]"')
        return 1
    logger.info("loaded whisper {} in {:.1f}s", model_name, time.perf_counter() - t0)
    silence = encode_wav_b64(np.zeros(16_000, dtype=np.float32), 16_000)
    t1 = time.perf_counter()
    result = await stt.transcribe(silence, None, "en")
    logger.info("silence -> {!r} in {:.2f}s", result.text, time.perf_counter() - t1)
    if result.text.strip():
        logger.error("whisper hallucinated on silence")
        return 1
    logger.info("OK: local STT functional; set WHISPER_BACKEND=local WHISPER_MODEL={}", model_name)
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.model)))


if __name__ == "__main__":
    main()
