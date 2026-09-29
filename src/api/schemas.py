"""Pydantic schemas shared by the HTTP API, the agent graph and the evaluator.

Validation happens here so malformed input is rejected with HTTP 422 before it
reaches any model or audio backend.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import get_settings

Action = Literal["respond", "ask_clarification", "book", "transfer_to_human"]

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class Slot(BaseModel):
    """One piece of information the booking task needs.

    Attributes:
        name: Slot identifier (``party_size``, ``date``, ``time``, ``name``).
        value: Extracted value, or ``None`` while unknown.
        confirmed: Whether the user explicitly confirmed the value.
    """

    name: str = Field(min_length=1, max_length=40)
    value: str | int | float | None = None
    confirmed: bool = False


class TurnRequest(BaseModel):
    """One user turn: either a transcript or base64 WAV audio.

    Attributes:
        session_id: Conversation identifier; 1-64 URL-safe characters.
        text: Transcript to use instead of running STT (tests, text clients).
        audio_b64: Base64-encoded WAV bytes; size-limited by ``MAX_AUDIO_BYTES``.
        language: BCP-47 language code passed to STT/TTS.
    """

    session_id: str
    text: str | None = None
    audio_b64: str | None = None
    language: str = Field(default="en", min_length=2, max_length=8)

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: str) -> str:
        if not SESSION_ID_RE.match(value):
            raise ValueError("session_id must be 1-64 characters of [A-Za-z0-9_-]")
        return value

    @field_validator("text")
    @classmethod
    def _check_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        limit = get_settings().max_text_chars
        if len(stripped) > limit:
            raise ValueError(f"text exceeds {limit} characters")
        return stripped

    @field_validator("audio_b64")
    @classmethod
    def _check_audio(cls, value: str | None) -> str | None:
        if value is None:
            return None
        limit = get_settings().max_audio_b64_chars
        if len(value) > limit:
            raise ValueError(f"audio_b64 exceeds the {limit}-character limit")
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio_b64 is not valid base64") from exc
        if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            raise ValueError("audio_b64 must decode to a RIFF/WAVE file")
        return value

    @model_validator(mode="after")
    def _require_some_input(self) -> TurnRequest:
        if not self.text and not self.audio_b64:
            raise ValueError("provide either text or audio_b64")
        return self


class LatencyBreakdown(BaseModel):
    """Per-node wall-clock latency of a turn, in milliseconds."""

    vad_ms: int = 0
    stt_ms: int = 0
    llm_ms: int = 0
    tts_ms: int = 0
    logger_ms: int = 0
    total_ms: int = 0


class UsageInfo(BaseModel):
    """Token, character and cost accounting for one turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    tts_characters: int = 0
    audio_seconds: float = 0.0
    cost_usd: float = 0.0


class TurnResponse(BaseModel):
    """Agent reply for one turn.

    Attributes:
        session_id: Echoed conversation identifier.
        trace_id: Identifier shared by every log line of this turn.
        turn: 1-based turn counter within the session.
        transcript: Text the agent understood (STT output or echoed text).
        response_text: What the agent says.
        response_audio_b64: Synthesized speech (WAV/MP3) as base64.
        action: Decision taken by the reasoner.
        slots: Current slot state after this turn.
        finished: Whether the task has been completed or handed off.
        speech_detected: Whether VAD detected speech in the audio (always true
            for text input).
        latency: Per-node latency breakdown.
        usage: Token/character usage and cost.
        backends: Which backend served each stage (``stub``, ``claude``...).
    """

    session_id: str
    trace_id: str
    turn: int
    transcript: str = ""
    response_text: str = ""
    response_audio_b64: str | None = None
    action: Action = "respond"
    slots: list[Slot] = Field(default_factory=list)
    finished: bool = False
    speech_detected: bool = True
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    backends: dict[str, str] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Mutable conversation state kept per ``session_id``.

    Attributes:
        session_id: Conversation identifier.
        task: Task type; only ``restaurant_booking`` is implemented.
        slots: Filled slots so far.
        history: Alternating ``user``/``assistant`` messages.
        finished: Whether the session reached a terminal action.
        awaiting_confirmation: Whether all slots are filled and the agent asked
            the user to confirm the booking.
    """

    session_id: str
    task: str = "restaurant_booking"
    slots: list[Slot] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    finished: bool = False
    awaiting_confirmation: bool = False


class SessionTranscript(BaseModel):
    """Persisted view of a session returned by ``GET /api/sessions/{id}``."""

    session_id: str
    turns: list[dict[str, object]] = Field(default_factory=list)
    finished: bool = False
