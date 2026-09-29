"""FastAPI transport for the voice agent.

Endpoints:

* ``GET  /health``                      -- liveness plus configured backends.
* ``POST /api/turn``                    -- one conversational turn (rate limited).
* ``GET  /api/sessions/{id}``           -- persisted transcript of a session.
* ``POST /api/sessions/{id}/reset``     -- drop in-memory state.
* ``GET  /api/metrics``                 -- last 100 turns with latency/cost.

CORS origins and the rate limit come from :class:`src.config.Settings`; all
domain errors are mapped to typed JSON errors instead of ``500 str(exc)``.
"""

import statistics
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agent.loop import VoiceLoop
from src.agent.reasoner import ReasonerError
from src.agent.vad import AudioDecodeError
from src.api.schemas import SessionTranscript, TurnRequest, TurnResponse
from src.config import get_settings
from src.observability import configure_logging
from src.stt.whisper_wrapper import STTError
from src.tts.synth import TTSError

try:
    APP_VERSION = version("voice-agent")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    APP_VERSION = "1.0.0"

_settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[_settings.rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and build the pipeline once per process."""
    settings = get_settings()
    configure_logging(settings)
    app.state.voice = VoiceLoop(settings)
    logger.info("voice agent ready version={} model={}", APP_VERSION, settings.anthropic_model)
    yield
    app.state.voice.close()


app = FastAPI(
    title="Voice AI Conversational Agent",
    version=APP_VERSION,
    description="VAD -> STT -> Claude reasoner -> TTS turn loop orchestrated with LangGraph.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Trace-Id"],
)


@app.exception_handler(ReasonerError)
async def _reasoner_error(_: Request, exc: ReasonerError) -> JSONResponse:
    logger.error("reasoner failure: {}", exc)
    return JSONResponse(
        status_code=502, content={"error": "reasoner_unavailable", "detail": str(exc)}
    )


@app.exception_handler(STTError)
async def _stt_error(_: Request, exc: STTError) -> JSONResponse:
    logger.error("stt failure: {}", exc)
    return JSONResponse(status_code=502, content={"error": "stt_unavailable", "detail": str(exc)})


@app.exception_handler(TTSError)
async def _tts_error(_: Request, exc: TTSError) -> JSONResponse:
    logger.error("tts failure: {}", exc)
    return JSONResponse(status_code=502, content={"error": "tts_unavailable", "detail": str(exc)})


@app.exception_handler(AudioDecodeError)
async def _audio_error(_: Request, exc: AudioDecodeError) -> JSONResponse:
    logger.warning("bad audio payload: {}", exc)
    return JSONResponse(status_code=422, content={"error": "invalid_audio", "detail": str(exc)})


@app.middleware("http")
async def _trace_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Propagate the turn's trace id as a response header when available."""
    response = await call_next(request)
    trace_id = getattr(request.state, "trace_id", None)
    if trace_id:
        response.headers["X-Trace-Id"] = trace_id
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe with the active backend configuration."""
    s = get_settings()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": s.anthropic_model,
        "whisper_backend": s.whisper_backend,
        "tts_backend": s.tts_backend,
        "vad_backend": s.vad_backend,
        "logger_backend": s.logger_backend,
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
        "elevenlabs": "set" if s.elevenlabs_api_key else "not set",
    }


@app.post("/api/turn", response_model=TurnResponse)
@limiter.limit(_settings.rate_limit)
async def turn(request: Request, req: TurnRequest) -> TurnResponse:
    """Run one turn through the graph."""
    voice: VoiceLoop = request.app.state.voice
    result = await voice.turn(req)
    request.state.trace_id = result.trace_id
    return result


@app.get("/api/sessions/{session_id}", response_model=SessionTranscript)
async def get_session(request: Request, session_id: str) -> SessionTranscript:
    """Return the persisted transcript of a session (404 if unknown)."""
    voice: VoiceLoop = request.app.state.voice
    transcript = voice.transcript(session_id)
    if not transcript.turns:
        raise HTTPException(status_code=404, detail="session not found")
    return transcript


@app.post("/api/sessions/{session_id}/reset")
async def reset(request: Request, session_id: str) -> dict[str, str | bool]:
    """Drop the in-memory state of a session."""
    voice: VoiceLoop = request.app.state.voice
    existed = voice.reset(session_id)
    return {"reset": session_id, "existed": existed}


@app.get("/api/metrics")
async def metrics(request: Request) -> dict[str, object]:
    """Summarise the last 100 persisted turns (observability dashboard data)."""
    voice: VoiceLoop = request.app.state.voice
    recent = voice.conversation_logger.recent(100)
    totals = [int(round(sum(r.latency.values()))) for r in recent]
    costs = [float(r.usage.get("cost_usd", 0.0)) for r in recent]
    return {
        "turns": len(recent),
        "persisted_total": voice.conversation_logger.count(),
        "latency_ms": {
            "mean": round(statistics.mean(totals), 1) if totals else 0.0,
            "p95": sorted(totals)[max(0, int(round(len(totals) * 0.95)) - 1)] if totals else 0,
        },
        "cost_usd_total": round(sum(costs), 6),
        "actions": {a: sum(1 for r in recent if r.action == a) for a in {r.action for r in recent}},
        "recent": [r.model_dump(mode="json") for r in recent[:100]],
    }
