"""Restaurant-booking reasoner.

Two implementations share the :class:`Reasoner` protocol:

* :class:`StubReasoner` -- deterministic regex slot filling. It needs no API key
  and is the offline baseline reported in ``eval/RESULTS.md``.
* :class:`ClaudeReasoner` -- calls the Anthropic Messages API with a strict JSON
  contract, validates the reply with Pydantic and retries transient failures
  with tenacity. Used whenever ``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import anthropic
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.api.schemas import Action, Slot
from src.config import Settings

REQUIRED_SLOTS: tuple[str, ...] = ("party_size", "date", "time", "name")

SLOT_PROMPTS: dict[str, str] = {
    "party_size": "How many people will be dining?",
    "date": "What date would you like to book?",
    "time": "What time works best for you?",
    "name": "And under what name should I make the booking?",
}

HUMAN_RE = re.compile(
    r"\b(speak|talk)\s+(to|with)\s+(a\s+|an\s+|someone|somebody)?\s*(human|person|agent|manager|operator|representative)\b"
    r"|\b(real\s+person|human\s+being|operator|transfer\s+me)\b",
    re.IGNORECASE,
)
OUT_OF_SCOPE_RE = re.compile(
    r"\b(cancel|refund|delivery|takeaway|take-out|takeout|complain|complaint|allerg\w*|job|hiring)\b",
    re.IGNORECASE,
)
AFFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|correct|confirm(ed)?|that'?s right|sounds good|please do|go ahead|ok(ay)?|perfect|exactly)\b",
    re.IGNORECASE,
)
NEGATE_RE = re.compile(r"^\s*(no|nope|not quite|wrong|incorrect|change|actually)\b", re.IGNORECASE)


class ReasonResult(BaseModel):
    """Outcome of one reasoning step.

    Attributes:
        response_text: Sentence to speak back to the user.
        action: Decision taken this turn.
        slots: Updated slot list.
        finished: Whether the conversation is over (booked or transferred).
        awaiting_confirmation: Whether the agent is waiting for a yes/no.
        input_tokens: Prompt tokens consumed (0 for the stub).
        output_tokens: Completion tokens consumed (0 for the stub).
        backend: ``"stub"`` or ``"claude"``.
    """

    response_text: str
    action: Action
    slots: list[Slot] = Field(default_factory=list)
    finished: bool = False
    awaiting_confirmation: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    backend: str = "stub"


class Reasoner(Protocol):
    """Anything that can decide the next action of the booking dialogue."""

    name: str

    def respond(
        self,
        *,
        transcript: str,
        slots: list[Slot],
        history: list[dict[str, str]],
        awaiting_confirmation: bool = False,
    ) -> ReasonResult:
        """Decide the next action given the latest transcript and state."""
        ...


class ReasonerError(RuntimeError):
    """Raised when the LLM reasoner cannot produce a valid decision."""


def _confirmation_sentence(s: dict[str, Slot]) -> str:
    """Build the read-back sentence used before booking."""
    return (
        f"Just to confirm: a table for {s['party_size'].value} on {s['date'].value} "
        f"at {s['time'].value} under the name {s['name'].value}. Shall I book it?"
    )


def _booking_sentence(s: dict[str, Slot]) -> str:
    """Build the final confirmation sentence."""
    return (
        f"Great, your booking is confirmed for {s['name'].value}, party of "
        f"{s['party_size'].value}, on {s['date'].value} at {s['time'].value}. "
        "We'll send a confirmation message. Goodbye!"
    )


class StubReasoner:
    """Deterministic slot-filling agent (English only, no API key).

    The stub asks for missing slots in a fixed order, reads the booking back
    once every slot is known and books on an affirmative answer. Requests for a
    human or out-of-scope intents are transferred.
    """

    name = "stub"

    NUM_WORDS: dict[str, int] = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }

    def respond(
        self,
        *,
        transcript: str,
        slots: list[Slot],
        history: list[dict[str, str]],
        awaiting_confirmation: bool = False,
    ) -> ReasonResult:
        """Run one deterministic reasoning step.

        Args:
            transcript: Latest user utterance.
            slots: Slots filled so far.
            history: Prior messages (unused by the stub, kept for the protocol).
            awaiting_confirmation: Whether the previous turn asked for a yes/no.

        Returns:
            The decision for this turn.
        """
        s: dict[str, Slot] = {sl.name: sl for sl in slots}

        if HUMAN_RE.search(transcript):
            return ReasonResult(
                response_text="Of course, let me transfer you to a colleague. One moment please.",
                action="transfer_to_human",
                slots=list(slots),
                finished=True,
            )

        if awaiting_confirmation and len(s) == len(REQUIRED_SLOTS):
            if AFFIRM_RE.search(transcript):
                for sl in s.values():
                    sl.confirmed = True
                return ReasonResult(
                    response_text=_booking_sentence(s),
                    action="book",
                    slots=self._ordered(s),
                    finished=True,
                )
            if NEGATE_RE.search(transcript):
                s = self._apply_corrections(transcript, s)
                return ReasonResult(
                    response_text="No problem. " + _confirmation_sentence(s),
                    action="ask_clarification",
                    slots=self._ordered(s),
                    awaiting_confirmation=True,
                )

        if OUT_OF_SCOPE_RE.search(transcript) and not s:
            return ReasonResult(
                response_text=(
                    "I can only help with table reservations here. "
                    "Let me transfer you to someone who can help with that."
                ),
                action="transfer_to_human",
                slots=[],
                finished=True,
            )

        s = self._extract_all(transcript, s)
        missing = [k for k in REQUIRED_SLOTS if k not in s]
        if not missing:
            return ReasonResult(
                response_text=_confirmation_sentence(s),
                action="ask_clarification",
                slots=self._ordered(s),
                awaiting_confirmation=True,
            )

        if not s and not history:
            prefix = "Welcome to the restaurant, I can help you book a table. "
        else:
            prefix = ""
        return ReasonResult(
            response_text=prefix + SLOT_PROMPTS[missing[0]],
            action="ask_clarification",
            slots=self._ordered(s),
        )

    # -- helpers ---------------------------------------------------------------

    def _ordered(self, s: dict[str, Slot]) -> list[Slot]:
        ordered = [s[k] for k in REQUIRED_SLOTS if k in s]
        ordered.extend(s[k] for k in s if k not in REQUIRED_SLOTS)
        return ordered

    def _extract_all(self, text: str, s: dict[str, Slot]) -> dict[str, Slot]:
        if "party_size" not in s:
            n = self._extract_party(text)
            if n is not None:
                s["party_size"] = Slot(name="party_size", value=n, confirmed=True)
        if "date" not in s:
            d = self._extract_date(text)
            if d is not None:
                s["date"] = Slot(name="date", value=d, confirmed=True)
        if "time" not in s:
            t = self._extract_time(text)
            if t is not None:
                s["time"] = Slot(name="time", value=t, confirmed=True)
        if "name" not in s:
            name = self._extract_name(text)
            if name is not None:
                s["name"] = Slot(name="name", value=name, confirmed=True)
        return s

    def _apply_corrections(self, text: str, s: dict[str, Slot]) -> dict[str, Slot]:
        """Re-extract any slot mentioned in a correction utterance."""
        n = self._extract_party(text)
        if n is not None:
            s["party_size"] = Slot(name="party_size", value=n, confirmed=True)
        d = self._extract_date(text)
        if d is not None:
            s["date"] = Slot(name="date", value=d, confirmed=True)
        t = self._extract_time(text)
        if t is not None:
            s["time"] = Slot(name="time", value=t, confirmed=True)
        name = self._extract_name(text)
        if name is not None:
            s["name"] = Slot(name="name", value=name, confirmed=True)
        return s

    def _extract_party(self, text: str) -> int | None:
        t = text.lower()
        # Remove time expressions so "7 pm" is never read as 7 guests.
        t_no_time = re.sub(r"\b\d{1,2}(:\d{2})?\s*(am|pm|h|hrs|o'clock)\b", " ", t)
        t_no_time = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", t_no_time)
        t_no_time = re.sub(r"\b\d{1,2}:\d{2}\b", " ", t_no_time)
        m = re.search(
            r"\b(?:for|of|party of|table for|group of|be|are|there are)?\s*(\d{1,2})\s*(?:people|persons|guests|of us|diners|adults)?\b",
            t_no_time,
        )
        if m:
            n = int(m.group(1))
            if 1 <= n <= 20:
                return n
        for word, n in self.NUM_WORDS.items():
            if re.search(rf"\b{word}\b", t_no_time) and re.search(
                r"\b(people|persons|guests|of us|diners|for|party|table)\b", t_no_time
            ):
                return n
        if re.search(r"\bjust me\b|\bfor myself\b|\bmyself\b", t):
            return 1
        return None

    def _extract_date(self, text: str) -> str | None:
        t = text.lower()
        for key in ("tomorrow", "today", "tonight"):
            if re.search(rf"\b{key}\b", t):
                return key
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
        if m:
            return m.group(1)
        m = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
            t,
        )
        if m:
            return f"{m.group(1)} {int(m.group(2))}"
        m = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            t,
        )
        if m:
            return f"{m.group(2)} {int(m.group(1))}"
        for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
            if re.search(rf"\b{day}\b", t):
                return day
        return None

    def _extract_time(self, text: str) -> str | None:
        t = text.lower()
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|h|hrs|o'clock)\b", t)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2) or 0)
            if m.group(3) == "pm" and hh < 12:
                hh += 12
            if m.group(3) == "o'clock" and hh < 12 and re.search(r"\b(evening|night|dinner)\b", t):
                hh += 12
            if hh > 23 or mm > 59:
                return None
            return f"{hh:02d}:{mm:02d}"
        m2 = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
        if m2:
            hh, mm = int(m2.group(1)), int(m2.group(2))
            if hh > 23 or mm > 59:
                return None
            return f"{hh:02d}:{mm:02d}"
        if re.search(r"\bnoon\b|\bmidday\b", t):
            return "12:00"
        return None

    def _extract_name(self, text: str) -> str | None:
        anchors = (
            r"under\s+(?:the\s+)?name\s+(?:of\s+)?",
            r"(?:my\s+)?name\s+is\s+",
            r"(?:my\s+)?name'?s\s+",
            r"\bunder\s+",
            r"\bname\s+",
            r"\bit'?s\s+",
            r"\bfor\s+",
            r"\bthis\s+is\s+",
            r"\bi'?m\s+",
            r"\bbe\s+",
            r"\bto\s+",
        )
        stop = {"the", "a", "an", "table", "two", "three", "four", "five", "six", "seven"}
        stop |= {"eight", "nine", "ten", "tonight", "tomorrow", "today", "dinner", "lunch"}
        stop |= {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
        stop |= {"january", "february", "march", "april", "may", "june", "july", "august"}
        stop |= {"september", "october", "november", "december", "noon", "please", "instead"}
        stop |= {"sure", "yes", "no", "not", "just", "now", "later", "there", "here", "myself"}
        stop |= {"someone", "somebody", "anyone", "a", "one", "me", "us", "you", "it", "that"}
        for prefix in anchors:
            m = re.search(
                "(?i:" + prefix + r")([A-Z][a-zA-Z\-']+)(?:\s+([A-Z][a-zA-Z\-']+))?", text
            )
            if m:
                first = m.group(1)
                if first.lower() in stop:
                    continue
                last = m.group(2)
                if last and last.lower() not in stop:
                    return f"{first} {last}"
                return first
        return None


class LLMTurnOutput(BaseModel):
    """Strict JSON contract the model must return."""

    response_text: str = Field(min_length=1, max_length=600)
    action: Action
    slots: list[Slot] = Field(default_factory=list, max_length=8)
    finished: bool = False
    awaiting_confirmation: bool = False


SYSTEM_PROMPT = """You are a warm, concise voice assistant taking restaurant table bookings by phone.

Goal: collect four slots and book the table.
  - party_size (integer 1-20)
  - date (string as the user said it, e.g. "tomorrow", "friday", "2026-10-03")
  - time (24h "HH:MM")
  - name (string)

Rules:
1. Ask for ONE missing slot at a time, in this order: party_size, date, time, name.
2. When all four slots are known, read them back and ask the user to confirm
   (set "awaiting_confirmation": true, action "ask_clarification").
3. Only after an explicit yes, set action "book" and "finished": true.
4. If the user asks for a human, or asks for anything that is not a table booking
   (cancellations, delivery, complaints, jobs), set action "transfer_to_human"
   and "finished": true with a short polite handoff sentence.
5. Keep response_text to one or two short spoken sentences. No markdown, no lists.
6. Never invent slot values; only use what the user said.

Return ONLY a JSON object with exactly these keys and no surrounding text:
{"response_text": str, "action": "respond"|"ask_clarification"|"book"|"transfer_to_human",
 "slots": [{"name": str, "value": str|int|null, "confirmed": bool}],
 "finished": bool, "awaiting_confirmation": bool}
"""

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


class ClaudeReasoner:
    """Reasoner backed by the Anthropic Messages API.

    The client is created with an explicit timeout and SDK retries disabled so
    that tenacity owns the retry policy (exponential backoff, bounded attempts).
    The reply must be a JSON object matching :class:`LLMTurnOutput`; anything
    else raises :class:`ReasonerError` so the caller can decide how to degrade.
    """

    name = "claude"

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        """Create the reasoner.

        Args:
            settings: Active settings (model, timeout, retries, max tokens).
            client: Optional pre-built client (tests inject a mock here).

        Raises:
            ReasonerError: If no API key is configured and no client is given.
        """
        if client is None and not settings.anthropic_api_key:
            raise ReasonerError("ANTHROPIC_API_KEY is required for ClaudeReasoner")
        self.settings = settings
        self.model = settings.anthropic_model
        self.client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        self._call = retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(max(1, settings.llm_max_retries)),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )(self._create)

    def _create(self, messages: list[dict[str, str]]) -> anthropic.types.Message:
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=messages,  # type: ignore[arg-type]
        )

    def respond(
        self,
        *,
        transcript: str,
        slots: list[Slot],
        history: list[dict[str, str]],
        awaiting_confirmation: bool = False,
    ) -> ReasonResult:
        """Ask Claude for the next action.

        Args:
            transcript: Latest user utterance.
            slots: Slots filled so far (sent as JSON state).
            history: Prior ``user``/``assistant`` messages.
            awaiting_confirmation: Whether the previous turn asked for a yes/no.

        Returns:
            Validated decision with token usage attached.

        Raises:
            ReasonerError: On non-retryable API errors, exhausted retries, or a
                reply that is not valid JSON for :class:`LLMTurnOutput`.
        """
        state = {
            "slots": [s.model_dump() for s in slots],
            "awaiting_confirmation": awaiting_confirmation,
        }
        messages: list[dict[str, str]] = [
            {"role": m["role"], "content": m["content"]} for m in history[-12:]
        ]
        messages.append(
            {
                "role": "user",
                "content": f"<state>{json.dumps(state)}</state>\n<transcript>{transcript}</transcript>",
            }
        )
        try:
            message = self._call(messages)
        except _RETRYABLE as exc:
            raise ReasonerError(f"Anthropic API unavailable after retries: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ReasonerError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

        text = "".join(block.text for block in message.content if block.type == "text")
        parsed = self._parse(text)
        usage_in = int(getattr(message.usage, "input_tokens", 0) or 0)
        usage_out = int(getattr(message.usage, "output_tokens", 0) or 0)
        logger.debug(
            "claude reasoner action={} finished={} tokens_in={} tokens_out={}",
            parsed.action,
            parsed.finished,
            usage_in,
            usage_out,
        )
        return ReasonResult(
            response_text=parsed.response_text.strip(),
            action=parsed.action,
            slots=parsed.slots,
            finished=parsed.finished,
            awaiting_confirmation=parsed.awaiting_confirmation,
            input_tokens=usage_in,
            output_tokens=usage_out,
            backend=self.name,
        )

    @staticmethod
    def _parse(text: str) -> LLMTurnOutput:
        """Extract and validate the JSON object from the model reply."""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise ReasonerError(f"model reply is not JSON: {text[:120]!r}")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ReasonerError(f"model reply is malformed JSON: {exc}") from exc
        try:
            return LLMTurnOutput.model_validate(data)
        except ValidationError as exc:
            raise ReasonerError(f"model reply failed schema validation: {exc}") from exc


def build_reasoner(settings: Settings) -> Reasoner:
    """Pick the reasoner for the current configuration.

    Args:
        settings: Active settings.

    Returns:
        :class:`ClaudeReasoner` when an API key is present, else
        :class:`StubReasoner`. The choice is logged at INFO.
    """
    if settings.anthropic_api_key:
        logger.info("reasoner backend=claude model={}", settings.anthropic_model)
        return ClaudeReasoner(settings)
    logger.info("reasoner backend=stub (ANTHROPIC_API_KEY not set)")
    return StubReasoner()
