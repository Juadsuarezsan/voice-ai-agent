"""WER computation and the benchmark script on the hand-written fixture."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.eval.wer import compute_wer, normalize, read_lines

FIXTURE = Path("data/eval/wer_fixture")


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wer_benchmark", "scripts/wer_benchmark.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_normalize() -> None:
    assert normalize("Hello, World!  It's 7:30 PM.") == "hello world its 730 pm"
    assert normalize("") == ""


def test_perfect_match_is_zero() -> None:
    r = compute_wer(["table for two"], ["Table for two."])
    assert r.wer == 0.0 and r.hits == 3 and r.n_reference_words == 3


def test_counts_substitutions_deletions_insertions() -> None:
    r = compute_wer(["a b c d e f"], ["a x c d e f g"])
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 1)
    assert r.wer == pytest.approx(2 / 6, abs=1e-4)
    d = compute_wer(["a b c"], ["a c"])
    assert (d.substitutions, d.deletions, d.insertions) == (0, 1, 0)


def test_fixture_wer_matches_hand_count() -> None:
    refs = read_lines(FIXTURE / "references.txt")
    hyps = read_lines(FIXTURE / "hypotheses.txt")
    r = compute_wer(refs, hyps)
    assert r.n_utterances == 10 and r.n_reference_words == 102
    assert (r.substitutions, r.deletions, r.insertions) == (2, 2, 1)
    assert r.wer == pytest.approx(5 / 102, abs=1e-3)


def test_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        compute_wer(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        compute_wer([], [])


def test_script_text_mode_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "wer.json"
    rc = _script().main(
        [
            "--references",
            str(FIXTURE / "references.txt"),
            "--hypotheses",
            str(FIXTURE / "hypotheses.txt"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["source"]["fixture"] is True
    assert payload["metrics"]["wer"] == pytest.approx(0.049, abs=1e-3)
    assert "jiwer" in payload["normalisation"]


def test_script_librispeech_mode_with_fake_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = tmp_path / "test-clean" / "1" / "2"
    split.mkdir(parents=True)
    (split / "1-2.trans.txt").write_text("1-2-0000 HELLO THERE\n1-2-0001 TABLE FOR TWO\n")
    (split / "1-2-0000.flac").write_bytes(b"fLaC")
    (split / "1-2-0001.flac").write_bytes(b"fLaC")

    class _Model:
        def transcribe(self, path: str, **kwargs: Any) -> dict[str, str]:
            return {"text": "hello there" if "0000" in path else "table for three"}

    fake = ModuleType("whisper")
    fake.load_model = lambda name: _Model()  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake)
    mod = _script()
    pairs = mod.load_librispeech(tmp_path / "test-clean")
    assert [ref for _, ref in pairs] == ["HELLO THERE", "TABLE FOR TWO"]
    out = tmp_path / "ls.json"
    assert (
        mod.main(
            [
                "--librispeech-dir",
                str(tmp_path / "test-clean"),
                "--whisper-model",
                "tiny",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text())
    assert payload["source"]["model"] == "whisper-tiny" and payload["metrics"]["substitutions"] == 1
    assert payload["metrics"]["wer"] == pytest.approx(0.2)


def test_script_requires_arguments() -> None:
    with pytest.raises(SystemExit):
        _script().main([])
