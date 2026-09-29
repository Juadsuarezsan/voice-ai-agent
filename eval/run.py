"""Run the full evaluation and regenerate ``eval/RESULTS.md``.

Usage::

    python -m eval.run              # write eval/runs/<date>-<name>.json + eval/RESULTS.md
    python -m eval.run --check      # re-run and compare deterministic metrics with the latest run
    python -m eval.run --n 20       # subset for a quick smoke run

Without ``ANTHROPIC_API_KEY`` only the deterministic stub baseline, the heuristic
judge proxy and the VAD ablation run; every other cell is written as pending.
With a key, the Claude baselines and the LLM judge are added to the same run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.loop import VoiceLoop
from src.config import Settings
from src.eval.ablation import run_vad_ablation
from src.eval.baselines import BASELINES, runnable_baselines
from src.eval.dataset import Conversation, load_conversations
from src.eval.judge import ClaudeJudge, HeuristicJudge, Judge, summarize
from src.eval.report import render_results
from src.eval.simulate import SimulationReport, simulate
from src.observability import configure_logging

RUNS_DIR = Path("eval/runs")
RESULTS_MD = Path("eval/RESULTS.md")
DETERMINISTIC_KEYS = (
    "n",
    "resolved",
    "resolution_rate",
    "booking_resolution_rate",
    "transfer_accuracy",
    "slot_exact_match_rate",
)


def _judge_all(
    judge: Judge, report: SimulationReport, conversations: list[Conversation]
) -> dict[str, float]:
    turns = {c.id: len(c.turns) for c in conversations}
    verdicts = [judge.score(o, turns[o.id]) for o in report.outcomes]
    return summarize(verdicts)


async def build_run(
    settings: Settings, conversations: list[Conversation], name: str
) -> dict[str, Any]:
    """Execute every runnable baseline, judge and ablation and return the run dict."""
    baselines: dict[str, Any] = {}
    reports: dict[str, SimulationReport] = {}
    for baseline in runnable_baselines(settings):
        loop = VoiceLoop(settings, reasoner=baseline.build(settings))
        logger.info("simulating baseline={} n={}", baseline.name, len(conversations))
        report = await simulate(loop, conversations, baseline.name)
        reports[baseline.name] = report
        baselines[baseline.name] = report.model_dump(exclude={"outcomes"})
        loop.close()
    skipped = [b.name for b in BASELINES if b.name not in baselines]

    stub_report = reports["stub_heuristic"]
    judge: dict[str, Any] = {"heuristic": _judge_all(HeuristicJudge(), stub_report, conversations)}
    if settings.anthropic_api_key and "claude_pipeline" in reports:
        judge["claude"] = _judge_all(
            ClaudeJudge(settings), reports["claude_pipeline"], conversations
        )

    vad = [r.model_dump() for r in await run_vad_ablation(settings)]

    # Per-node latency from a persisted stub run over the first 20 conversations.
    node_lat: dict[str, list[int]] = {}
    loop_for_nodes = VoiceLoop(settings)
    await simulate(loop_for_nodes, conversations[:20], "stub_heuristic")
    for record in loop_for_nodes.conversation_logger.recent(1000):
        for k, v in record.latency.items():
            node_lat.setdefault(k, []).append(v)
    latency_per_node = {k: round(statistics.mean(v), 2) for k, v in sorted(node_lat.items())}

    failures = [
        {
            "id": o.id,
            "scenario": o.scenario,
            "expected_action": o.expected_action,
            "final_action": o.final_action,
            "slots": o.slots,
            "expected_slots": o.expected_slots,
            "turns_used": o.turns_used,
            "transcript": o.transcript,
        }
        for o in stub_report.outcomes
        if not o.resolved
    ]
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_file = RUNS_DIR / f"{generated_at[:10]}-{name}.json"
    return {
        "name": name,
        "generated_at": generated_at,
        "run_file": str(run_file),
        "model": settings.anthropic_model,
        "llm_enabled": bool(settings.anthropic_api_key),
        "backends": {
            "reasoner": (
                "claude" if settings.anthropic_api_key else "stub (fallback determinista, sin LLM)"
            ),
            "stt": settings.whisper_backend,
            "tts": settings.tts_backend,
            "vad": settings.vad_backend,
        },
        "dataset": {
            "path": "data/eval/conversations.jsonl",
            "n": len(conversations),
            "synthetic": True,
        },
        "baselines": baselines,
        "skipped_baselines": {b: "pendiente (requiere ANTHROPIC_API_KEY)" for b in skipped},
        "judge": judge,
        "ablations": {
            "vad": vad,
            "whisper": "pendiente (requiere extra `[stt]` con torch y LibriSpeech; openslr.org bloqueado en este entorno)",
        },
        "latency_per_node_ms": latency_per_node,
        "failures": failures,
        "outcomes": [o.model_dump() for o in stub_report.outcomes],
    }


def _latest_run() -> dict[str, Any] | None:
    files = sorted(RUNS_DIR.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def check(run: dict[str, Any]) -> int:
    """Compare deterministic metrics of ``run`` with the latest saved run."""
    latest = _latest_run()
    if latest is None:
        logger.error("no saved run in {}", RUNS_DIR)
        return 1
    mismatches = []
    for key in DETERMINISTIC_KEYS:
        a = latest["baselines"]["stub_heuristic"][key]
        b = run["baselines"]["stub_heuristic"][key]
        if a != b:
            mismatches.append((key, a, b))
    if mismatches:
        for key, a, b in mismatches:
            logger.error("metric {} changed: saved={} now={}", key, a, b)
        return 1
    sys.stdout.write(f"check ok: deterministic metrics match {latest['run_file']}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="use only the first N conversations")
    parser.add_argument("--name", default=None, help="run name (default: offline-stub or claude)")
    parser.add_argument(
        "--check", action="store_true", help="compare with the latest saved run; do not write"
    )
    args = parser.parse_args(argv)

    settings = Settings(LOG_LEVEL="WARNING")
    configure_logging(settings)
    conversations = load_conversations()
    if args.n:
        conversations = conversations[: args.n]
    name = args.name or ("claude" if settings.anthropic_api_key else "offline-stub")
    run = asyncio.run(build_run(settings, conversations, name))
    if args.check:
        return check(run)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    Path(run["run_file"]).write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    RESULTS_MD.write_text(render_results(run), encoding="utf-8")
    logger.warning("wrote {} and {}", run["run_file"], RESULTS_MD)
    stub = run["baselines"]["stub_heuristic"]
    sys.stdout.write(
        f"resolution_rate={stub['resolution_rate']} ({stub['resolved']}/{stub['n']}) "
        f"[fallback determinista, sin LLM] p95={stub['latency_p95_ms']}ms\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
