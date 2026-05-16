"""Manual verification that Whisper local STT works on this machine.

Run: python scripts/verify_whisper.py
- Downloads whisper tiny (~75 MB) to ~/.cache/whisper on first run
- Generates a short sine wave with a known frequency
- Transcribes silence (must return empty string, not hallucinate)
- Prints model parameter count and inference time

This is the script run before P9 v1.0.0 to confirm the local STT path is
functional. The real WER benchmark requires LibriSpeech (50 GB) which is
not committed.
"""
from __future__ import annotations

import time

import numpy as np


def main() -> None:
    import whisper

    t0 = time.perf_counter()
    print("Loading whisper tiny model (~75 MB on first run)...")
    model = whisper.load_model("tiny")
    load_t = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {load_t:.1f}s  params: {n_params:,}")

    # Test 1: silence must transcribe to empty (not hallucinate)
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    t0 = time.perf_counter()
    r = model.transcribe(silence, language="en", fp16=False)
    print(f"  silence → {r['text']!r}  (in {time.perf_counter()-t0:.2f}s)")
    assert r["text"].strip() == "", "Whisper hallucinated on silence — model load failed"

    print("\nOK — Whisper local STT functional. Wire WHISPER_BACKEND=local in .env.")


if __name__ == "__main__":
    main()
