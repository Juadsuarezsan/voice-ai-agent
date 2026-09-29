"""Evaluation harness: dataset, simulator, judges, baselines, ablation, report and runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from pytest_mock import MockerFixture

from eval import run as eval_run
from src.agent.loop import VoiceLoop
from src.agent.reasoner import ClaudeReasoner, StubReasoner
from src.config import Settings
from src.eval.ablation import AlwaysSpeechVAD, run_vad_ablation
from src.eval.baselines import BASELINES, StatelessClaudeReasoner, runnable_baselines
from src.eval.dataset import Conversation, Expected, load_conversations
from src.eval.judge import RUBRIC, ClaudeJudge, HeuristicJudge, JudgeError, rubric_text, summarize
from src.eval.report import render_results
from src.eval.simulate import ConversationOutcome, percentile, simulate

DATASET = Path("data/eval/conversations.jsonl")


def test_dataset_has_100_synthetic_conversations() -> None:
    convs = load_conversations(DATASET)
    assert len(convs) == 100
    assert all(c.synthetic for c in convs)
    assert len({c.id for c in convs}) == 100
    assert {c.scenario for c in convs} >= {
        "incremental",
        "one_shot",
        "correction",
        "transfer_human",
        "out_of_scope",
        "spanish",
    }
    booking = [c for c in convs if c.expected.final_action == "book"]
    assert all(set(c.expected.slots) == {"party_size", "date", "time", "name"} for c in booking)


def test_build_eval_set_is_deterministic() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("build_eval_set", "scripts/build_eval_set.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    a, b = mod.build(30), mod.build(30)
    assert a == b and len(a) == 30
    assert a[0]["synthetic"] is True and a[0]["id"] == "conv-001"


@pytest.mark.asyncio
async def test_simulate_scores_stub(settings: Settings) -> None:
    convs = [
        Conversation(
            id="t-book",
            scenario="incremental",
            turns=["Hi", "Two people", "Tomorrow", "8 pm", "Under Ana", "Yes"],
            expected=Expected(
                slots={"party_size": 2, "date": "tomorrow", "time": "20:00", "name": "Ana"},
                final_action="book",
            ),
        ),
        Conversation(
            id="t-human",
            scenario="transfer_human",
            turns=["Hi", "Transfer me to an operator"],
            expected=Expected(final_action="transfer_to_human"),
        ),
        Conversation(
            id="t-fail",
            scenario="spanish",
            turns=["Hola, quiero reservar."],
            expected=Expected(
                slots={"party_size": 2, "date": "tomorrow", "time": "20:00", "name": "Ana"},
                final_action="book",
            ),
        ),
    ]
    report = await simulate(VoiceLoop(settings), convs, "stub_heuristic")
    assert report.n == 3 and report.resolved == 2
    assert report.booking_n == 2 and report.booking_resolution_rate == 0.5
    assert report.transfer_n == 1 and report.transfer_accuracy == 1.0
    assert report.per_scenario["spanish"]["resolution_rate"] == 0.0
    assert report.latency_p95_ms >= 0 and report.total_cost_usd == 0.0
    ok = next(o for o in report.outcomes if o.id == "t-book")
    assert ok.resolved and ok.slots_exact and ok.turns_used == 6


def test_percentile_nearest_rank() -> None:
    assert percentile([], 0.95) == 0
    assert percentile([5, 1, 9, 3], 0.5) == 3
    assert percentile(list(range(1, 101)), 0.95) == 95


def _outcome(**overrides: Any) -> ConversationOutcome:
    base: dict[str, Any] = {
        "id": "c1",
        "scenario": "incremental",
        "expected_action": "book",
        "final_action": "book",
        "finished": True,
        "turns_used": 6,
        "slots": {"party_size": 2},
        "expected_slots": {"party_size": 2},
        "slots_exact": True,
        "resolved": True,
        "transcript": [{"user": "hi", "agent": "How many people?", "action": "ask_clarification"}],
    }
    base.update(overrides)
    return ConversationOutcome(**base)


def test_rubric_is_numbered_and_rendered() -> None:
    assert [r[0] for r in RUBRIC] == [1, 2, 3, 4, 5]
    text = rubric_text()
    assert text.startswith("1. task_completion") and "5. naturalness" in text


def test_heuristic_judge_scores_from_ground_truth() -> None:
    judge = HeuristicJudge()
    good = judge.score(_outcome(), scripted_turns=6)
    assert good.passed and good.efficiency == 1 and good.naturalness == 5
    transferred = judge.score(
        _outcome(final_action="transfer_to_human", slots_exact=False, resolved=False), 6
    )
    assert transferred.task_completion == 0 and transferred.no_unnecessary_transfer == 0
    verbose = _outcome(
        transcript=[
            {"user": "x", "agent": "One. Two. Three. Four sentences here.", "action": "respond"}
        ]
    )
    assert judge.score(verbose, 6).naturalness == 3
    markdown = _outcome(
        transcript=[{"user": "x", "agent": "- item one\n- item two", "action": "respond"}]
    )
    assert judge.score(markdown, 6).naturalness == 1
    slow = judge.score(_outcome(turns_used=9), scripted_turns=6)
    assert slow.efficiency == 0


def test_summarize_verdicts() -> None:
    judge = HeuristicJudge()
    verdicts = [
        judge.score(_outcome(), 6),
        judge.score(_outcome(final_action="respond", finished=False, resolved=False), 6),
    ]
    s = summarize(verdicts)
    assert s["n"] == 2 and s["pass_rate"] == 0.5 and s["task_completion"] == 0.5
    assert summarize([]) == {"n": 0, "pass_rate": 0.0}


def _judge_client(mocker: MockerFixture, text: str) -> Any:
    client = mocker.MagicMock(spec=anthropic.Anthropic)
    client.messages = mocker.MagicMock()
    client.messages.create = mocker.MagicMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    )
    return client


def test_claude_judge_parses_verdict(mocker: MockerFixture) -> None:
    reply = json.dumps(
        {
            "task_completion": 1,
            "slot_correctness": 1,
            "no_unnecessary_transfer": 1,
            "efficiency": 0,
            "naturalness": 4,
            "rationale": "fine",
        }
    )
    judge = ClaudeJudge(Settings(ANTHROPIC_API_KEY="k"), client=_judge_client(mocker, reply))
    verdict = judge.score(_outcome(), scripted_turns=6)
    assert verdict.judge == "claude" and verdict.passed and verdict.naturalness == 4
    kwargs = judge.client.messages.create.call_args.kwargs
    assert "1. task_completion" in kwargs["system"] and kwargs["temperature"] == 0.0


def test_claude_judge_rejects_bad_reply(mocker: MockerFixture) -> None:
    judge = ClaudeJudge(
        Settings(ANTHROPIC_API_KEY="k"), client=_judge_client(mocker, "no json here")
    )
    with pytest.raises(JudgeError, match="not JSON"):
        judge.score(_outcome(), 6)
    judge = ClaudeJudge(
        Settings(ANTHROPIC_API_KEY="k"), client=_judge_client(mocker, '{"naturalness": 9}')
    )
    with pytest.raises(JudgeError, match="invalid"):
        judge.score(_outcome(), 6)
    with pytest.raises(JudgeError, match="ANTHROPIC_API_KEY"):
        ClaudeJudge(Settings(ANTHROPIC_API_KEY=None))


def test_baselines_registry(mocker: MockerFixture) -> None:
    assert [b.name for b in BASELINES] == [
        "stub_heuristic",
        "claude_zero_shot_stateless",
        "claude_pipeline",
    ]
    assert [b.name for b in runnable_baselines(Settings(ANTHROPIC_API_KEY=None))] == [
        "stub_heuristic"
    ]
    mocker.patch("anthropic.Anthropic")
    with_key = Settings(ANTHROPIC_API_KEY="k")
    assert len(runnable_baselines(with_key)) == 3
    assert isinstance(BASELINES[0].build(with_key), StubReasoner)
    assert isinstance(BASELINES[1].build(with_key), StatelessClaudeReasoner)
    assert isinstance(BASELINES[2].build(with_key), ClaudeReasoner)


def test_stateless_baseline_drops_slot_state(mocker: MockerFixture) -> None:
    inner = mocker.MagicMock(spec=ClaudeReasoner)
    from src.agent.reasoner import ReasonResult

    inner.respond.return_value = ReasonResult(
        response_text="ok", action="respond", backend="claude"
    )
    from src.api.schemas import Slot

    out = StatelessClaudeReasoner(inner).respond(
        transcript="hi",
        slots=[Slot(name="date", value="x")],
        history=[],
        awaiting_confirmation=True,
    )
    assert out.backend == "claude_stateless"
    inner.respond.assert_called_once_with(
        transcript="hi", slots=[], history=[], awaiting_confirmation=False
    )


@pytest.mark.asyncio
async def test_vad_ablation_energy_vs_off(settings: Settings) -> None:
    results = {r.vad: r for r in await run_vad_ablation(settings, n_turns=6)}
    assert (
        results["energy"].silence_rejection_rate == 1.0
        and results["energy"].speech_pass_rate == 1.0
    )
    assert results["off"].silence_rejection_rate == 0.0 and results["off"].stt_calls_saved == 0
    import numpy as np

    assert AlwaysSpeechVAD().detect(np.zeros(16, dtype=np.float32), 16).speech_detected is True


@pytest.mark.asyncio
async def test_build_run_and_render(settings: Settings) -> None:
    convs = load_conversations(DATASET)[:8]
    run = await eval_run.build_run(settings, convs, "unit")
    assert run["llm_enabled"] is False
    assert run["baselines"]["stub_heuristic"]["n"] == 8
    assert "claude_pipeline" in run["skipped_baselines"]
    assert run["judge"]["heuristic"]["n"] == 8 and "claude" not in run["judge"]
    assert {r["vad"] for r in run["ablations"]["vad"]} == {"energy", "off"}
    assert set(run["latency_per_node_ms"]) >= {"stt", "llm", "tts"}
    md = render_results(run)
    assert "fallback determinista, sin LLM" in md
    assert "pendiente (requiere ANTHROPIC_API_KEY)" in md
    assert "| STT | WER sobre LibriSpeech test-clean | pendiente" in md
    assert "| TTS | MOS automated | pendiente" in md


def test_saved_run_matches_results_md() -> None:
    """RESULTS.md must be the rendering of the saved run (no hand-edited numbers)."""
    runs = sorted(Path("eval/runs").glob("*.json"))
    assert runs, "eval/runs must contain at least one saved run"
    run = json.loads(runs[-1].read_text(encoding="utf-8"))
    assert Path("eval/RESULTS.md").read_text(encoding="utf-8") == render_results(run)
    assert run["baselines"]["stub_heuristic"]["n"] == 100


def test_check_mode_matches_saved_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert eval_run.main(["--check"]) == 0
