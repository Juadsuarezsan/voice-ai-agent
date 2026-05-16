"""Voice AI Conversational Agent — placeholder until v0.1.0 build out."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Voice AI Conversational Agent",
    version="0.1.0",
    description="Voice AI Agent — Whisper + Claude + ElevenLabs, sub-second turn target",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.1.0",
        "stage": "scaffolding",
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }

class AudioTurnRequest(BaseModel):
    session_id: str
    audio_b64: str | None = None
    text: str | None = None  # for testing without audio


class AudioTurnResponse(BaseModel):
    session_id: str
    transcript: str = ""
    response_text: str = ""
    response_audio_b64: str | None = None
    latency_stt_ms: int = 0
    latency_llm_ms: int = 0
    latency_tts_ms: int = 0
    latency_total_ms: int = 0


@app.post("/api/turn", response_model=AudioTurnResponse)
async def turn(req: AudioTurnRequest) -> AudioTurnResponse:
    return AudioTurnResponse(session_id=req.session_id, transcript="not_yet_implemented")
