"""Drive scripted conversations through a :class:`VoiceLoop` and score them."""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from src.agent.loop import VoiceLoop
from src.api.schemas import TurnRequest
from src.eval.dataset import Conversation


class ConversationOutcome(BaseModel):
    """Result of simulating one conversation."""

    id: str
    scenario: str
    expected_action: str
    final_action: str
    finished: bool
    turns_used: int
    slots: dict[str, str | int | float | None] = Field(default_factory=dict)
    expected_slots: dict[str, str | int] = Field(default_factory=dict)
    slots_exact: bool
    resolved: bool
    latencies_ms: list[int] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    transcript: list[dict[str, str]] = Field(default_factory=list)
    backend: str = "stub"


class SimulationReport(BaseModel):
    """Aggregate metrics over a simulated eval set."""

    backend: str
    n: int
    resolved: int
    resolution_rate: float
    booking_n: int
    booking_resolution_rate: float
    transfer_n: int
    transfer_accuracy: float
    slot_exact_match_rate: float
    avg_turns_to_finish: float | None
    latency_mean_ms: float
    latency_p50_ms: int
    latency_p95_ms: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    per_scenario: dict[str, dict[str, float]] = Field(default_factory=dict)
    outcomes: list[ConversationOutcome] = Field(default_factory=list)


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile (``q`` in ``[0, 1]``)."""
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(len(ordered) * q)) - 1))
    return ordered[idx]


def _is_resolved(outcome_action: str, finished: bool, expected: str, slots_exact: bool) -> bool:
    if expected == "book":
        return finished and outcome_action == "book" and slots_exact
    return finished and outcome_action == expected


async def simulate_conversation(loop: VoiceLoop, conv: Conversation) -> ConversationOutcome:
    """Run one scripted conversation until it finishes or turns run out."""
    loop.reset(conv.id)
    latencies: list[int] = []
    transcript: list[dict[str, str]] = []
    tokens_in = tokens_out = 0
    cost = 0.0
    last: Any = None
    for utterance in conv.turns:
        last = await loop.turn(TurnRequest(session_id=conv.id, text=utterance))
        latencies.append(last.latency.total_ms)
        tokens_in += last.usage.input_tokens
        tokens_out += last.usage.output_tokens
        cost += last.usage.cost_usd
        transcript.append({"user": utterance, "agent": last.response_text, "action": last.action})
        if last.finished:
            break
    slots = {s.name: s.value for s in last.slots} if last else {}
    expected_slots = dict(conv.expected.slots)
    slots_exact = (
        {k: slots.get(k) for k in expected_slots} == expected_slots if expected_slots else True
    )
    final_action = last.action if last else "respond"
    finished = bool(last and last.finished)
    return ConversationOutcome(
        id=conv.id,
        scenario=conv.scenario,
        expected_action=conv.expected.final_action,
        final_action=final_action,
        finished=finished,
        turns_used=len(transcript),
        slots=slots,
        expected_slots=expected_slots,
        slots_exact=slots_exact,
        resolved=_is_resolved(final_action, finished, conv.expected.final_action, slots_exact),
        latencies_ms=latencies,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost_usd=round(cost, 8),
        transcript=transcript,
        backend=(last.backends.get("reasoner", "stub") if last else "stub"),
    )


async def simulate(
    loop: VoiceLoop, conversations: list[Conversation], backend: str
) -> SimulationReport:
    """Simulate every conversation and aggregate the metrics.

    Args:
        loop: Pipeline under test.
        conversations: Eval set.
        backend: Label for the reasoner backend (``stub``, ``claude``...).

    Returns:
        Aggregate report including every outcome.
    """
    outcomes = [await simulate_conversation(loop, c) for c in conversations]
    all_lat = [ms for o in outcomes for ms in o.latencies_ms]
    booking = [o for o in outcomes if o.expected_action == "book"]
    transfer = [o for o in outcomes if o.expected_action == "transfer_to_human"]
    finished_turns = [o.turns_used for o in outcomes if o.resolved]
    per_scenario: dict[str, dict[str, float]] = {}
    for scenario in sorted({o.scenario for o in outcomes}):
        group = [o for o in outcomes if o.scenario == scenario]
        per_scenario[scenario] = {
            "n": len(group),
            "resolved": sum(o.resolved for o in group),
            "resolution_rate": round(sum(o.resolved for o in group) / len(group), 4),
        }
    return SimulationReport(
        backend=backend,
        n=len(outcomes),
        resolved=sum(o.resolved for o in outcomes),
        resolution_rate=round(sum(o.resolved for o in outcomes) / max(1, len(outcomes)), 4),
        booking_n=len(booking),
        booking_resolution_rate=round(sum(o.resolved for o in booking) / max(1, len(booking)), 4),
        transfer_n=len(transfer),
        transfer_accuracy=round(sum(o.resolved for o in transfer) / max(1, len(transfer)), 4),
        slot_exact_match_rate=round(sum(o.slots_exact for o in booking) / max(1, len(booking)), 4),
        avg_turns_to_finish=round(statistics.mean(finished_turns), 2) if finished_turns else None,
        latency_mean_ms=round(statistics.mean(all_lat), 2) if all_lat else 0.0,
        latency_p50_ms=percentile(all_lat, 0.5),
        latency_p95_ms=percentile(all_lat, 0.95),
        total_input_tokens=sum(o.input_tokens for o in outcomes),
        total_output_tokens=sum(o.output_tokens for o in outcomes),
        total_cost_usd=round(sum(o.cost_usd for o in outcomes), 6),
        per_scenario=per_scenario,
        outcomes=outcomes,
    )
