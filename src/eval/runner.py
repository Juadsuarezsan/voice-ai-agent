"""Multi-turn conversation simulator — measures resolution rate + slot fill."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics

from src.agent.loop import VoiceLoop
from src.api.schemas import TurnRequest


# 10 simulated multi-turn conversations
SCENARIOS = [
    {
        "id": "s-01", "turns": [
            "Hi, I'd like to book a table.",
            "Four people, please.",
            "Tomorrow at 7:30 pm.",
            "Under the name John Smith.",
        ],
    },
    {
        "id": "s-02", "turns": [
            "I want to make a reservation.",
            "For six people on Friday.",
            "At 8 pm.",
            "Name is Maria.",
        ],
    },
    {
        "id": "s-03", "turns": [
            "Table for two tomorrow night.",
            "7 pm.",
            "Name Alice.",
        ],
    },
    {
        "id": "s-04", "turns": [
            "Booking for 3 people tonight at 6:00 under Carlos.",
        ],
    },
    {
        "id": "s-05", "turns": [
            "I'd like a dinner reservation.",
            "Five of us.",
            "Saturday at 9 pm.",
            "Under Maria Lopez.",
        ],
    },
]


async def run_eval() -> dict:
    loop = VoiceLoop()
    resolutions = 0
    avg_turns: list[int] = []
    latencies: list[int] = []
    per_scenario = []

    for s in SCENARIOS:
        loop.reset(s["id"])
        last_resp = None
        for utterance in s["turns"]:
            req = TurnRequest(session_id=s["id"], text=utterance)
            last_resp = await loop.turn(req)
            latencies.append(last_resp.latency_total_ms)
            if last_resp.finished:
                break
        if last_resp and last_resp.finished:
            resolutions += 1
            avg_turns.append(last_resp.turn)
        per_scenario.append({
            "id": s["id"],
            "finished": bool(last_resp and last_resp.finished),
            "turns": last_resp.turn if last_resp else 0,
            "filled_slots": [sl.name for sl in (last_resp.slots if last_resp else [])],
        })

    return {
        "n": len(SCENARIOS),
        "resolution_rate": resolutions / len(SCENARIOS),
        "avg_turns_to_book": statistics.mean(avg_turns) if avg_turns else None,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        "latency_p95_ms": _p95(latencies),
        "per_scenario": per_scenario,
    }


def _p95(xs: list[int]) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    idx = max(0, int(round(len(s) * 0.95)) - 1)
    return s[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    r = asyncio.run(run_eval())
    if args.json:
        print(json.dumps(r, indent=2, default=str))
    else:
        print(f"Scenarios: {r['n']}")
        print(f"Resolution rate: {r['resolution_rate']:.1%}")
        print(f"Avg turns to book: {r['avg_turns_to_book']}")
        print(f"Avg latency: {r['avg_latency_ms']:.0f} ms  p95: {r['latency_p95_ms']} ms")


if __name__ == "__main__":
    main()
