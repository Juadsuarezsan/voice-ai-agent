"""Baselines compared in ``eval/RESULTS.md``.

* ``stub_heuristic``          -- regex slot filling, no LLM (offline, measurable).
* ``claude_zero_shot_stateless`` -- Claude with the conversation history only:
  no slot state is passed between turns, so the model must re-derive every
  slot from the transcript each time. Needs ``ANTHROPIC_API_KEY``.
* ``claude_pipeline``         -- the production reasoner with slot state.
  Needs ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.reasoner import ClaudeReasoner, Reasoner, ReasonResult, StubReasoner
from src.api.schemas import Slot
from src.config import Settings


class StatelessClaudeReasoner:
    """Zero-shot baseline: Claude sees the history but never the slot state."""

    name = "claude_stateless"

    def __init__(self, inner: ClaudeReasoner) -> None:
        """Wrap a configured :class:`ClaudeReasoner`."""
        self.inner = inner

    def respond(
        self,
        *,
        transcript: str,
        slots: list[Slot],
        history: list[dict[str, str]],
        awaiting_confirmation: bool = False,
    ) -> ReasonResult:
        """Delegate with empty slot state and no confirmation flag."""
        result = self.inner.respond(
            transcript=transcript, slots=[], history=history, awaiting_confirmation=False
        )
        result.backend = self.name
        return result


@dataclass(frozen=True)
class Baseline:
    """A named reasoner configuration and whether it can run without keys."""

    name: str
    description: str
    requires_key: bool

    def build(self, settings: Settings) -> Reasoner:
        """Instantiate the reasoner (raises if a key is missing)."""
        if self.name == "stub_heuristic":
            return StubReasoner()
        inner = ClaudeReasoner(settings)
        if self.name == "claude_zero_shot_stateless":
            return StatelessClaudeReasoner(inner)
        return inner


BASELINES: tuple[Baseline, ...] = (
    Baseline("stub_heuristic", "Regex slot filling, no LLM (fallback determinista)", False),
    Baseline("claude_zero_shot_stateless", "Claude, history only, no slot state", True),
    Baseline("claude_pipeline", "Claude with slot state (production reasoner)", True),
)


def runnable_baselines(settings: Settings) -> list[Baseline]:
    """Baselines that can run with the current settings."""
    return [b for b in BASELINES if not b.requires_key or settings.anthropic_api_key]
