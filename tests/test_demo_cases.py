"""demo/predictions.json is produced by the repository code and has >=5 cases of >=6 turns."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.agent.reasoner import StubReasoner
from src.api.schemas import Slot


def test_predictions_json_shape() -> None:
    data = json.loads(Path("demo/predictions.json").read_text(encoding="utf-8"))
    assert data["generator"] == "scripts/build_demo_cases.py"
    assert "sin LLM" in data["backend"]
    cases = data["cases"]
    assert len(cases) >= 5
    for case in cases:
        assert len(case["turns"]) >= 6, case["id"]
        assert case["turns"][-1]["finished"] is True
        assert case["final_action"] in {"book", "transfer_to_human"}
        assert all(t["audio_omitted"] is True for t in case["turns"])
        assert all("total_ms" in t["latency"] for t in case["turns"])
    assert sum(c["final_action"] == "book" for c in cases) >= 3


@pytest.mark.asyncio
async def test_build_demo_cases_regenerates_same_transcripts() -> None:
    spec = importlib.util.spec_from_file_location("build_demo_cases", "scripts/build_demo_cases.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    fresh = await mod.build()
    saved = json.loads(Path("demo/predictions.json").read_text(encoding="utf-8"))
    strip = lambda cases: [  # noqa: E731
        [(t["user"], t["agent"], t["action"], t["slots"]) for t in c["turns"]] for c in cases
    ]
    assert strip(fresh["cases"]) == strip(saved["cases"])


def test_name_extraction_regressions_from_demo() -> None:
    r = StubReasoner()
    assert r._extract_name("For Friday, please.") is None
    assert r._extract_name("Around noon please") is None
    assert r._extract_name("Actually, could it be Priya Patel instead?") == "Priya Patel"
    slots = [
        Slot(name=n, value=v, confirmed=True)
        for n, v in (("party_size", 3), ("date", "tonight"), ("time", "18:45"), ("name", "Carlos"))
    ]
    out = r.respond(
        transcript="Actually, could it be Priya Patel instead?",
        slots=slots,
        history=[],
        awaiting_confirmation=True,
    )
    assert next(s.value for s in out.slots if s.name == "name") == "Priya Patel"
    assert out.awaiting_confirmation is True
