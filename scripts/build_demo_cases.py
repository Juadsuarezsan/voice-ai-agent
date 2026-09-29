"""Produce ``demo/predictions.json``: pre-baked conversations for the static demo.

The conversations are run through the real pipeline (``VoiceLoop`` + LangGraph)
with the deterministic stub backends, so every transcript, action, slot and
latency in the file is an actual output of this repository's code. The stub
audio (a placeholder tone) is omitted to keep the file small; the live demo
gets real audio from ``POST /api/turn``.

Run::

    python scripts/build_demo_cases.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.loop import VoiceLoop  # noqa: E402
from src.api.schemas import TurnRequest  # noqa: E402
from src.config import Settings  # noqa: E402
from src.observability import configure_logging  # noqa: E402

OUT = Path("demo/predictions.json")

CASES: list[dict[str, Any]] = [
    {
        "id": "demo-01",
        "title": "Reserva paso a paso (6 turnos)",
        "turns": [
            "Hi, I'd like to book a table.",
            "Four people, please.",
            "Tomorrow evening.",
            "7:30 pm.",
            "Under the name John Smith.",
            "Yes, that's right.",
        ],
    },
    {
        "id": "demo-02",
        "title": "Corrección durante la confirmación (7 turnos)",
        "turns": [
            "Hello, can I make a reservation?",
            "There will be two of us.",
            "For Friday, please.",
            "At 8 pm.",
            "My name is Maria Lopez.",
            "No, make it 6 people.",
            "Correct, go ahead.",
        ],
    },
    {
        "id": "demo-03",
        "title": "Todo en una frase y confirmación (6 turnos con relleno)",
        "turns": [
            "Good evening, I want to reserve a table please.",
            "um, let me think.",
            "Booking for 3 people tonight at 6:45 pm under Carlos.",
            "Hmm, one second.",
            "Actually, could it be Priya Patel instead?",
            "Yes please.",
        ],
    },
    {
        "id": "demo-04",
        "title": "Pide hablar con una persona en la confirmación (6 turnos)",
        "turns": [
            "Hi there, do you have any tables available?",
            "Party of 8.",
            "Saturday.",
            "9 pm.",
            "Under Daniel.",
            "Actually, I'd rather talk to a real person.",
        ],
    },
    {
        "id": "demo-05",
        "title": "Fecha ISO, mediodía y nombre compuesto (6 turnos)",
        "turns": [
            "Hello, I'd like to book a table for lunch.",
            "Five of us.",
            "2026-10-03.",
            "Noon.",
            "It's Sofia Rossi.",
            "Sounds good.",
        ],
    },
]


async def build() -> dict[str, Any]:
    """Run every case through the pipeline and collect the responses."""
    settings = Settings(LOG_LEVEL="WARNING", WHISPER_BACKEND="stub", TTS_BACKEND="stub")
    configure_logging(settings)
    loop = VoiceLoop(settings)
    cases: list[dict[str, Any]] = []
    for case in CASES:
        turns: list[dict[str, Any]] = []
        for utterance in case["turns"]:
            resp = await loop.turn(TurnRequest(session_id=case["id"], text=utterance))
            turns.append(
                {
                    "turn": resp.turn,
                    "user": utterance,
                    "agent": resp.response_text,
                    "action": resp.action,
                    "finished": resp.finished,
                    "slots": [s.model_dump() for s in resp.slots],
                    "latency": resp.latency.model_dump(),
                    "backends": resp.backends,
                    "audio_omitted": True,
                }
            )
            if resp.finished:
                break
        cases.append(
            {
                "id": case["id"],
                "title": case["title"],
                "turns": turns,
                "final_action": turns[-1]["action"],
            }
        )
    return {
        "project": "voice-ai-agent",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/build_demo_cases.py",
        "backend": "fallback determinista, sin LLM (StubReasoner + STT/TTS stub)",
        "model_when_enabled": settings.anthropic_model,
        "note": (
            "Transcripts, actions, slots and latencies are real outputs of the repository code "
            "with stub backends. They are not Claude/Whisper/ElevenLabs results. Placeholder "
            "audio omitted; the live demo fetches audio from POST /api/turn."
        ),
        "cases": cases,
    }


def main() -> None:
    """CLI entry point."""
    payload = asyncio.run(build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {OUT} with {len(payload['cases'])} cases\n")


if __name__ == "__main__":
    main()
