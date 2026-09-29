"""Session management around the LangGraph pipeline.

:class:`VoiceLoop` keeps per-session state (slots, history, confirmation flag),
assigns a ``trace_id`` per turn, runs the graph and maps the final graph state
to the HTTP response model.
"""

from __future__ import annotations

import time

from loguru import logger

from src.agent.graph import TurnState, VoicePipeline
from src.agent.reasoner import Reasoner, build_reasoner
from src.agent.vad import VoiceActivityDetector, build_vad
from src.api.schemas import (
    LatencyBreakdown,
    SessionState,
    SessionTranscript,
    TurnRequest,
    TurnResponse,
    UsageInfo,
)
from src.config import Settings, get_settings
from src.observability import new_trace_id, trace_context
from src.storage.logger import ConversationLogger, build_logger
from src.stt.whisper_wrapper import SpeechToText, build_stt
from src.tts.synth import TextToSpeech, build_tts


class VoiceLoop:
    """Turn loop with in-memory session state on top of :class:`VoicePipeline`."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        vad: VoiceActivityDetector | None = None,
        stt: SpeechToText | None = None,
        reasoner: Reasoner | None = None,
        tts: TextToSpeech | None = None,
        conversation_logger: ConversationLogger | None = None,
    ) -> None:
        """Build the pipeline from settings, allowing any backend to be injected.

        Args:
            settings: Active settings (defaults to :func:`get_settings`).
            vad: Voice activity detector override.
            stt: Speech-to-text override.
            reasoner: Reasoner override.
            tts: Text-to-speech override.
            conversation_logger: Persistence override.
        """
        self.settings = settings or get_settings()
        self.pipeline = VoicePipeline(
            self.settings,
            vad=vad or build_vad(self.settings),
            stt=stt or build_stt(self.settings),
            reasoner=reasoner or build_reasoner(self.settings),
            tts=tts or build_tts(self.settings),
            conversation_logger=conversation_logger or build_logger(self.settings),
        )
        self._sessions: dict[str, SessionState] = {}

    @property
    def conversation_logger(self) -> ConversationLogger:
        """Expose the persistence backend (used by the API read endpoints)."""
        return self.pipeline.conversation_logger

    def _get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    async def turn(self, req: TurnRequest) -> TurnResponse:
        """Process one user turn.

        Args:
            req: Validated request.

        Returns:
            The agent reply with latency, usage and backend metadata.
        """
        state = self._get_or_create(req.session_id)
        trace_id = new_trace_id()
        turn_idx = len(state.history) // 2 + 1
        t0 = time.perf_counter()

        with trace_context(trace_id, session_id=req.session_id):
            logger.info(
                "turn start turn={} has_audio={} has_text={}",
                turn_idx,
                bool(req.audio_b64),
                bool(req.text),
            )
            initial: TurnState = {
                "session_id": req.session_id,
                "trace_id": trace_id,
                "turn": turn_idx,
                "language": req.language,
                "text_in": req.text,
                "audio_b64": req.audio_b64,
                "history": list(state.history),
                "slots": list(state.slots),
                "awaiting_confirmation": state.awaiting_confirmation,
                "latency": {},
                "backends": {},
            }
            final = await self.pipeline.run(initial)
            total_ms = int((time.perf_counter() - t0) * 1000)

            transcript = final.get("transcript", "")
            response_text = final.get("response_text", "")
            if final.get("speech_detected", True):
                state.slots = list(final.get("slots", state.slots))
                state.awaiting_confirmation = final.get("awaiting_confirmation", False)
                state.history.extend(
                    [
                        {"role": "user", "content": transcript},
                        {"role": "assistant", "content": response_text},
                    ]
                )
                state.finished = final.get("finished", False)

            lat = final.get("latency", {})
            latency = LatencyBreakdown(
                vad_ms=int(round(lat.get("vad", 0))),
                stt_ms=int(round(lat.get("stt", 0))),
                llm_ms=int(round(lat.get("llm", 0))),
                tts_ms=int(round(lat.get("tts", 0))),
                logger_ms=int(round(lat.get("logger", 0))),
                total_ms=total_ms,
            )
            usage = UsageInfo(
                input_tokens=final.get("input_tokens", 0),
                output_tokens=final.get("output_tokens", 0),
                tts_characters=final.get("tts_characters", 0),
                audio_seconds=final.get("audio_seconds", 0.0),
                cost_usd=final.get("cost_usd", 0.0),
            )
            if total_ms > self.settings.turn_latency_target_ms:
                logger.warning(
                    "turn latency {} ms exceeds target {} ms",
                    total_ms,
                    self.settings.turn_latency_target_ms,
                )
            logger.info(
                "turn end total_ms={} action={} cost_usd={}",
                total_ms,
                final.get("action"),
                usage.cost_usd,
            )

        return TurnResponse(
            session_id=req.session_id,
            trace_id=trace_id,
            turn=turn_idx,
            transcript=transcript,
            response_text=response_text,
            response_audio_b64=final.get("audio_out_b64"),
            action=final.get("action", "respond"),
            slots=list(final.get("slots", [])),
            finished=final.get("finished", False),
            speech_detected=final.get("speech_detected", True),
            latency=latency,
            usage=usage,
            backends=dict(final.get("backends", {})),
        )

    def reset(self, session_id: str) -> bool:
        """Drop in-memory state for ``session_id``; returns whether it existed."""
        return self._sessions.pop(session_id, None) is not None

    def transcript(self, session_id: str) -> SessionTranscript:
        """Return the persisted turns of a session."""
        records = self.conversation_logger.get_session(session_id)
        return SessionTranscript(
            session_id=session_id,
            turns=[r.model_dump(mode="json") for r in records],
            finished=any(r.finished for r in records),
        )

    def close(self) -> None:
        """Release backend resources."""
        self.conversation_logger.close()
