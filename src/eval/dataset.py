"""Loading of the synthetic conversation eval set."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_PATH = Path("data/eval/conversations.jsonl")


class Expected(BaseModel):
    """Ground truth for one conversation."""

    slots: dict[str, str | int] = Field(default_factory=dict)
    final_action: str


class Conversation(BaseModel):
    """One scripted conversation with ground truth.

    Attributes:
        id: Stable identifier.
        synthetic: Always ``True`` for this dataset (declared, never mixed).
        scenario: Scenario family (``incremental``, ``one_shot``...).
        turns: Scripted user utterances in order.
        expected: Ground truth slots and final action.
    """

    id: str
    synthetic: bool = True
    scenario: str
    turns: list[str]
    expected: Expected


def load_conversations(path: Path = DEFAULT_PATH) -> list[Conversation]:
    """Read the JSONL eval set.

    Args:
        path: File with one JSON object per line.

    Returns:
        Parsed conversations in file order.

    Raises:
        FileNotFoundError: If the file is missing (run ``scripts/build_eval_set.py``).
    """
    records: list[Conversation] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(Conversation.model_validate(json.loads(line)))
    return records
