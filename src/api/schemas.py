from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["respond", "ask_clarification", "book", "transfer_to_human"]


class Slot(BaseModel):
    name: str
    value: str | int | float | None = None
    confirmed: bool = False


class TurnRequest(BaseModel):
    session_id: str
    text: str | None = None             # transcript (testing) — alternative to audio
    audio_b64: str | None = None        # raw WAV bytes base64-encoded
    language: str = "en"


class TurnResponse(BaseModel):
    session_id: str
    turn: int
    transcript: str = ""
    response_text: str = ""
    response_audio_b64: str | None = None
    action: Action = "respond"
    slots: list[Slot] = Field(default_factory=list)
    finished: bool = False
    latency_stt_ms: int = 0
    latency_llm_ms: int = 0
    latency_tts_ms: int = 0
    latency_total_ms: int = 0


class SessionState(BaseModel):
    session_id: str
    task: str = "restaurant_booking"
    slots: list[Slot] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    finished: bool = False
