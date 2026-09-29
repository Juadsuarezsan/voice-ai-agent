"""LLM-as-judge with an explicit, numbered rubric.

Two judges share the same rubric and output model:

* :class:`HeuristicJudge` scores criteria 1-4 deterministically from ground
  truth and criterion 5 with a surface proxy. It runs offline and is labelled
  as a proxy, never as the LLM judgement.
* :class:`ClaudeJudge` sends the transcript, ground truth and rubric to Claude
  and validates the JSON verdict. Requires ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import anthropic
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings
from src.eval.simulate import ConversationOutcome

#: (number, key, description, scale, example of a passing conversation)
RUBRIC: tuple[tuple[int, str, str, str, str], ...] = (
    (
        1,
        "task_completion",
        "The conversation ends with action `book` when the caller wanted a booking, or "
        "`transfer_to_human` when they asked for a person or for something outside table "
        "bookings (delivery, complaints, jobs).",
        "0 or 1",
        "Caller: 'Table for two tomorrow at 8 pm under Ana' -> ... -> agent books. Score 1.",
    ),
    (
        2,
        "slot_correctness",
        "Every slot in the final state (party_size, date, time HH:MM, name) equals the ground "
        "truth; no invented or missing values.",
        "0 or 1",
        "Ground truth {4, tomorrow, 20:00, Ana}; final slots identical. Score 1.",
    ),
    (
        3,
        "no_unnecessary_transfer",
        "The agent never transfers a caller who only wanted a table.",
        "0 or 1",
        "Caller asks for a table, agent asks for the date. Score 1. Agent transfers instead: 0.",
    ),
    (
        4,
        "efficiency",
        "The agent asks each missing slot at most once: turns used <= scripted turns + 1.",
        "0 or 1",
        "Six scripted turns, booking closed on turn 6. Score 1.",
    ),
    (
        5,
        "naturalness",
        "Replies are one or two short spoken sentences, polite, no markdown, lists or code.",
        "1 (robotic/unusable) to 5 (indistinguishable from a good human host)",
        "'Great, what time works best for you?' scores 5; a bulleted list scores 1.",
    ),
)


class JudgeVerdict(BaseModel):
    """Scores for one conversation, one field per rubric criterion."""

    conversation_id: str
    task_completion: int = Field(ge=0, le=1)
    slot_correctness: int = Field(ge=0, le=1)
    no_unnecessary_transfer: int = Field(ge=0, le=1)
    efficiency: int = Field(ge=0, le=1)
    naturalness: int = Field(ge=1, le=5)
    rationale: str = ""
    judge: str = "heuristic"

    @property
    def passed(self) -> bool:
        """A conversation passes when criteria 1-3 are all met."""
        return bool(self.task_completion and self.slot_correctness and self.no_unnecessary_transfer)


class Judge(Protocol):
    """Anything that can score a :class:`ConversationOutcome`."""

    name: str

    def score(self, outcome: ConversationOutcome, scripted_turns: int) -> JudgeVerdict:
        """Score one outcome."""
        ...


def rubric_text() -> str:
    """Render the rubric as numbered plain text for prompts and docs."""
    lines = []
    for number, key, description, scale, example in RUBRIC:
        lines.append(f"{number}. {key} [{scale}]: {description} Example: {example}")
    return "\n".join(lines)


_MARKDOWN_RE = re.compile(r"(^|\n)\s*([-*]|\d+\.)\s|\*\*|#|```")


class HeuristicJudge:
    """Deterministic proxy for the rubric (offline)."""

    name = "heuristic"

    def score(self, outcome: ConversationOutcome, scripted_turns: int) -> JudgeVerdict:
        """Score from ground truth; naturalness uses a surface proxy."""
        wanted_booking = outcome.expected_action == "book"
        task = int(outcome.finished and outcome.final_action == outcome.expected_action)
        slots = int(outcome.slots_exact) if wanted_booking else 1
        no_transfer = int(not (wanted_booking and outcome.final_action == "transfer_to_human"))
        efficiency = int(outcome.turns_used <= scripted_turns + 1)
        replies = [t["agent"] for t in outcome.transcript]
        natural = 5
        for reply in replies:
            sentences = [s for s in re.split(r"[.!?]+", reply) if s.strip()]
            if _MARKDOWN_RE.search(reply):
                natural = min(natural, 1)
            elif len(sentences) > 2 or len(reply) > 220:
                natural = min(natural, 3)
        return JudgeVerdict(
            conversation_id=outcome.id,
            task_completion=task,
            slot_correctness=slots,
            no_unnecessary_transfer=no_transfer,
            efficiency=efficiency,
            naturalness=natural,
            rationale="heuristic proxy computed from ground truth",
            judge=self.name,
        )


JUDGE_SYSTEM = (
    "You are grading a phone conversation between a caller and a restaurant booking voice "
    "agent. Score every criterion of the rubric strictly. Return ONLY a JSON object with the "
    "keys task_completion, slot_correctness, no_unnecessary_transfer, efficiency (each 0 or 1), "
    "naturalness (1-5) and rationale (one sentence).\n\nRubric:\n" + rubric_text()
)

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


class JudgeError(RuntimeError):
    """Raised when the LLM judge cannot produce a valid verdict."""


class ClaudeJudge:
    """LLM-as-judge backed by the Anthropic Messages API."""

    name = "claude"

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        """Create the judge (tests inject a mock client)."""
        if client is None and not settings.anthropic_api_key:
            raise JudgeError("ANTHROPIC_API_KEY is required for ClaudeJudge")
        self.settings = settings
        self.client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s, max_retries=0
        )
        self._call = retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(max(1, settings.llm_max_retries)),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )(self._create)

    def _create(self, prompt: str) -> anthropic.types.Message:
        return self.client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=300,
            temperature=0.0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

    def score(self, outcome: ConversationOutcome, scripted_turns: int) -> JudgeVerdict:
        """Ask Claude to grade the outcome against the rubric.

        Raises:
            JudgeError: On API failure or an invalid verdict.
        """
        prompt = json.dumps(
            {
                "conversation_id": outcome.id,
                "scripted_turns": scripted_turns,
                "transcript": outcome.transcript,
                "final_action": outcome.final_action,
                "final_slots": outcome.slots,
                "ground_truth": {
                    "final_action": outcome.expected_action,
                    "slots": outcome.expected_slots,
                },
            },
            ensure_ascii=False,
        )
        try:
            message = self._call(prompt)
        except _RETRYABLE as exc:
            raise JudgeError(f"Anthropic API unavailable after retries: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise JudgeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        text = "".join(b.text for b in message.content if b.type == "text")
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise JudgeError(f"judge reply is not JSON: {text[:120]!r}")
        try:
            data = json.loads(text[start : end + 1])
            data.update(conversation_id=outcome.id, judge=self.name)
            return JudgeVerdict.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise JudgeError(f"judge reply invalid: {exc}") from exc


def summarize(verdicts: list[JudgeVerdict]) -> dict[str, float]:
    """Aggregate verdicts into pass rate and mean score per criterion."""
    if not verdicts:
        return {"n": 0, "pass_rate": 0.0}
    n = len(verdicts)
    return {
        "n": n,
        "pass_rate": round(sum(v.passed for v in verdicts) / n, 4),
        "task_completion": round(sum(v.task_completion for v in verdicts) / n, 4),
        "slot_correctness": round(sum(v.slot_correctness for v in verdicts) / n, 4),
        "no_unnecessary_transfer": round(sum(v.no_unnecessary_transfer for v in verdicts) / n, 4),
        "efficiency": round(sum(v.efficiency for v in verdicts) / n, 4),
        "naturalness_mean": round(sum(v.naturalness for v in verdicts) / n, 3),
    }
