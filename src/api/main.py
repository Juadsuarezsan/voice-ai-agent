"""Voice AI Agent API."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from src.agent.loop import VoiceLoop
from src.api.schemas import TurnRequest, TurnResponse
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.voice = VoiceLoop()
    yield


app = FastAPI(
    title="Voice AI Conversational Agent",
    version="0.5.0",
    description="STT → reasoner → TTS turn loop targeting sub-second turn latency.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {
        "status": "ok", "version": "0.5.0", "stage": "substantive",
        "whisper_backend": s.whisper_backend, "tts_backend": s.tts_backend,
        "elevenlabs": "set" if s.elevenlabs_api_key else "not set",
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }


@app.post("/api/turn", response_model=TurnResponse)
async def turn(req: TurnRequest) -> TurnResponse:
    try:
        return await app.state.voice.turn(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/reset")
async def reset(session_id: str) -> dict:
    app.state.voice.reset(session_id)
    return {"reset": session_id}


@app.get("/api/eval/run")
async def eval_endpoint() -> dict:
    from src.eval.runner import run_eval
    return await run_eval()
