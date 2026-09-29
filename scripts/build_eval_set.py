"""Generate the synthetic conversation eval set (``data/eval/conversations.jsonl``).

Every utterance comes from the hand-written phrase banks below; the generator
only combines them with a fixed seed. Ground truth (expected slots and final
action) is derived from the sampled values, so it is correct by construction
and does not depend on any model. All records carry ``"synthetic": true``.

Run::

    python scripts/build_eval_set.py            # writes 100 conversations
    python scripts/build_eval_set.py --n 20     # smaller set for quick checks
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SEED = 20260516
OUT = Path("data/eval/conversations.jsonl")

# --- hand-written phrase banks -------------------------------------------------

NAMES = [
    "John Smith",
    "Maria Lopez",
    "Alice",
    "Carlos",
    "Priya Patel",
    "Daniel",
    "Sofia Rossi",
    "Ahmed",
    "Emily Chen",
    "Lucas",
]
PARTY = [1, 2, 3, 4, 5, 6, 8, 10]
DATES = ["tomorrow", "tonight", "friday", "saturday", "sunday", "2026-10-03", "october 12"]
TIMES = [
    ("7 pm", "19:00"),
    ("7:30 pm", "19:30"),
    ("8 pm", "20:00"),
    ("noon", "12:00"),
    ("6:45 pm", "18:45"),
    ("9 pm", "21:00"),
]

OPENERS = [
    "Hi, I'd like to book a table.",
    "Hello, can I make a reservation?",
    "Good evening, I want to reserve a table please.",
    "Hi there, do you have any tables available?",
    "um, hi, I'd like to book a table for dinner.",
]
PARTY_PHRASES = [
    "{n} people, please.",
    "There will be {n} of us.",
    "A table for {n}.",
    "uh, {n} guests.",
    "Party of {n}.",
]
DATE_PHRASES = ["{d}.", "For {d}, please.", "Let's say {d}.", "{d} would be great."]
TIME_PHRASES = ["At {t}.", "{t} please.", "Around {t}.", "Let's do {t}."]
NAME_PHRASES = ["Under the name {name}.", "My name is {name}.", "It's {name}.", "Under {name}."]
CONFIRM_PHRASES = ["Yes, that's right.", "Yes please.", "Correct, go ahead.", "Sounds good."]
ONE_SHOT_PHRASES = [
    "Booking for {n} people {d} at {t} under {name}.",
    "I need a table for {n} {d} at {t}, the name is {name}.",
]
CORRECTION_PARTY = ["No, make it {n} people.", "Actually, {n} of us."]
HUMAN_PHRASES = [
    "Can I speak to a human please?",
    "I'd rather talk to a real person.",
    "Transfer me to an operator.",
]
OUT_OF_SCOPE = [
    "I want to cancel my delivery order.",
    "I'd like to file a complaint about last night.",
    "Are you hiring for a job in the kitchen?",
]
SPANISH_OPENERS = ["Hola, quiero reservar una mesa.", "Buenas, me gustaría hacer una reserva."]
SPANISH_PARTY = ["Somos {n} personas.", "Para {n}, por favor."]
SPANISH_DATE = ["Para mañana.", "El viernes."]
SPANISH_TIME = ["A las {t}.", "Sobre las {t}."]
SPANISH_NAME = ["A nombre de {name}.", "Mi nombre es {name}."]
SPANISH_CONFIRM = ["Sí, correcto.", "Sí, perfecto."]


def _pick(rng: random.Random) -> dict[str, Any]:
    t_text, t_norm = rng.choice(TIMES)
    return {
        "n": rng.choice(PARTY),
        "d": rng.choice(DATES),
        "t": t_text,
        "t_norm": t_norm,
        "name": rng.choice(NAMES),
    }


def _expected(v: dict[str, Any], action: str) -> dict[str, Any]:
    slots = (
        {"party_size": v["n"], "date": v["d"], "time": v["t_norm"], "name": v["name"]}
        if action == "book"
        else {}
    )
    return {"slots": slots, "final_action": action}


def scenario_incremental(rng: random.Random) -> dict[str, Any]:
    v = _pick(rng)
    turns = [
        rng.choice(OPENERS),
        rng.choice(PARTY_PHRASES).format(**v),
        rng.choice(DATE_PHRASES).format(**v).capitalize(),
        rng.choice(TIME_PHRASES).format(**v),
        rng.choice(NAME_PHRASES).format(**v),
        rng.choice(CONFIRM_PHRASES),
    ]
    return {"scenario": "incremental", "turns": turns, "expected": _expected(v, "book")}


def scenario_one_shot(rng: random.Random) -> dict[str, Any]:
    v = _pick(rng)
    turns = [rng.choice(ONE_SHOT_PHRASES).format(**v), rng.choice(CONFIRM_PHRASES)]
    return {"scenario": "one_shot", "turns": turns, "expected": _expected(v, "book")}


def scenario_correction(rng: random.Random) -> dict[str, Any]:
    v = _pick(rng)
    wrong = rng.choice([p for p in PARTY if p != v["n"]])
    turns = [
        rng.choice(OPENERS),
        rng.choice(PARTY_PHRASES).format(n=wrong),
        rng.choice(DATE_PHRASES).format(**v).capitalize(),
        rng.choice(TIME_PHRASES).format(**v),
        rng.choice(NAME_PHRASES).format(**v),
        rng.choice(CORRECTION_PARTY).format(**v),
        rng.choice(CONFIRM_PHRASES),
    ]
    return {"scenario": "correction", "turns": turns, "expected": _expected(v, "book")}


def scenario_transfer(rng: random.Random) -> dict[str, Any]:
    v = _pick(rng)
    turns = [rng.choice(OPENERS), rng.choice(PARTY_PHRASES).format(**v), rng.choice(HUMAN_PHRASES)]
    return {
        "scenario": "transfer_human",
        "turns": turns,
        "expected": _expected(v, "transfer_to_human"),
    }


def scenario_out_of_scope(rng: random.Random) -> dict[str, Any]:
    turns = [rng.choice(OUT_OF_SCOPE)]
    return {
        "scenario": "out_of_scope",
        "turns": turns,
        "expected": _expected({}, "transfer_to_human"),
    }


def scenario_spanish(rng: random.Random) -> dict[str, Any]:
    v = _pick(rng)
    v["t"] = v["t_norm"]
    d = rng.choice(SPANISH_DATE)
    v["d"] = "tomorrow" if "mañana" in d else "friday"
    turns = [
        rng.choice(SPANISH_OPENERS),
        rng.choice(SPANISH_PARTY).format(**v),
        d,
        rng.choice(SPANISH_TIME).format(**v),
        rng.choice(SPANISH_NAME).format(**v),
        rng.choice(SPANISH_CONFIRM),
    ]
    return {"scenario": "spanish", "turns": turns, "expected": _expected(v, "book")}


MIX: list[tuple[Any, int]] = [
    (scenario_incremental, 40),
    (scenario_one_shot, 20),
    (scenario_correction, 12),
    (scenario_transfer, 10),
    (scenario_out_of_scope, 8),
    (scenario_spanish, 10),
]


def build(n: int = 100, seed: int = SEED) -> list[dict[str, Any]]:
    """Build ``n`` conversations following the scenario mix.

    Args:
        n: Number of conversations (the mix is scaled proportionally).
        seed: Random seed for reproducibility.

    Returns:
        List of JSON-serialisable conversation records.
    """
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    total_weight = sum(w for _, w in MIX)
    for fn, weight in MIX:
        count = round(n * weight / total_weight)
        for _ in range(count):
            records.append(fn(rng))
    records = records[:n]
    for i, rec in enumerate(records, start=1):
        rec_id = f"conv-{i:03d}"
        records[i - 1] = {
            "id": rec_id,
            "synthetic": True,
            "generator": "scripts/build_eval_set.py",
            "seed": seed,
            **rec,
        }
    return records


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    records = build(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} conversations to {args.out}")


if __name__ == "__main__":
    main()
