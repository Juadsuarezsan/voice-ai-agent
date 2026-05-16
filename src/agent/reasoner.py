"""Restaurant-booking reasoner. Maintains slots (date, party_size, name, time)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.api.schemas import Slot

REQUIRED_SLOTS = ("party_size", "date", "time", "name")


@dataclass
class ReasonResult:
    response_text: str
    action: str
    slots: list[Slot]
    finished: bool


class StubReasoner:
    """Slot-filling restaurant booking agent (no API key needed). Handles English."""

    NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    def respond(self, *, transcript: str, slots: list[Slot]) -> ReasonResult:
        s = {sl.name: sl for sl in slots}

        # Extract party_size
        if "party_size" not in s:
            n = self._extract_party(transcript)
            if n:
                s["party_size"] = Slot(name="party_size", value=n, confirmed=True)
        # Extract date
        if "date" not in s:
            d = self._extract_date(transcript)
            if d:
                s["date"] = Slot(name="date", value=d, confirmed=True)
        # Extract time
        if "time" not in s:
            t = self._extract_time(transcript)
            if t:
                s["time"] = Slot(name="time", value=t, confirmed=True)
        # Extract name
        if "name" not in s:
            n = self._extract_name(transcript)
            if n:
                s["name"] = Slot(name="name", value=n, confirmed=True)

        slot_list = [s[k] for k in REQUIRED_SLOTS if k in s] + [s[k] for k in s if k not in REQUIRED_SLOTS]
        missing = [k for k in REQUIRED_SLOTS if k not in s]

        if not missing:
            return ReasonResult(
                response_text=(
                    f"Great — booking confirmed for {s['name'].value}, "
                    f"party of {s['party_size'].value}, on {s['date'].value} at {s['time'].value}. "
                    "We'll send a confirmation email."
                ),
                action="book", slots=slot_list, finished=True,
            )

        prompts = {
            "party_size": "How many people will be dining?",
            "date":       "What date would you like to book?",
            "time":       "What time works best for you?",
            "name":       "And under what name should I make the booking?",
        }
        next_q = prompts[missing[0]]
        return ReasonResult(
            response_text=next_q, action="ask_clarification",
            slots=slot_list, finished=False,
        )

    def _extract_party(self, text: str) -> int | None:
        t = text.lower()
        # Numeric like "for 4" or "table for 6"
        m = re.search(r"\b(\d{1,2})\b\s*(?:people|persons|guests|of us)?", t)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 20:
                return n
        for word, n in self.NUM_WORDS.items():
            if word in t and ("people" in t or "of us" in t or "for" in t):
                return n
        return None

    def _extract_date(self, text: str) -> str | None:
        t = text.lower()
        for key in ("tomorrow", "today", "tonight"):
            if key in t:
                return key
        # ISO YYYY-MM-DD
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
        if m:
            return m.group(1)
        # Day-of-week
        for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
            if day in t:
                return day
        return None

    def _extract_time(self, text: str) -> str | None:
        t = text.lower()
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|h|hrs)\b", t)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2) or 0)
            if m.group(3) in ("pm",) and hh < 12:
                hh += 12
            return f"{hh:02d}:{mm:02d}"
        m2 = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
        if m2:
            return f"{int(m2.group(1)):02d}:{int(m2.group(2)):02d}"
        return None

    def _extract_name(self, text: str) -> str | None:
        # Anchor words are matched case-insensitively; the captured name
        # is required to start with an uppercase letter (so we don't match
        # 'for two').
        anchors = [
            (r"under\s+(?:the\s+)?name\s+", True),
            (r"name\s+is\s+",                True),
            (r"\bunder\s+",                  True),
            (r"\bname\s+",                   True),
            (r"\bfor\s+",                    True),
        ]
        for prefix, _ in anchors:
            full = prefix + r"([A-Z][a-zA-Z\-']+)"
            m = re.search(full, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = m.group(1)
                # Skip pure numbers / common stop words
                if val.lower() in {"the", "a", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}:
                    continue
                return val[0].upper() + val[1:]
        return None


class ClaudeReasoner:
    """Production: ask Claude with the SessionState + transcript for next action/slots."""

    SYSTEM = """You are a voice booking assistant for restaurants.

Maintain four slots: party_size (int), date (string), time (HH:MM), name (string).

Each turn you receive the conversation history + the user's latest transcript.
Output JSON ONLY:
{
  "response_text": "what to say aloud (1 sentence, natural, conversational)",
  "action": "respond | ask_clarification | book | transfer_to_human",
  "slots": [{"name":"...", "value": ..., "confirmed": true|false}],
  "finished": true|false
}

Mark finished=true when all 4 slots are confirmed.
Transfer_to_human if the user asks for one OR if they request something outside
restaurant booking.
"""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    def respond(self, *, transcript: str, slots: list[Slot]) -> ReasonResult:
        # The full Anthropic path lives here; we keep this synchronous to match
        # StubReasoner's contract — wire to ChatAnthropic in production code.
        raise NotImplementedError("Wire ChatAnthropic; for now use StubReasoner via build_reasoner()")


def build_reasoner() -> StubReasoner | ClaudeReasoner:
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        # ClaudeReasoner is wired but raises NotImplementedError for now — fall back to stub
        try:
            return ClaudeReasoner(model="claude-sonnet-4-5", api_key=api_key)
        except NotImplementedError:
            return StubReasoner()
    return StubReasoner()
