"""Per-turn loop: STT → reasoner → TTS, with per-component latency tracking."""
from __future__ import annotations

import time

from src.agent.reasoner import StubReasoner, build_reasoner
from src.api.schemas import SessionState, Slot, TurnRequest, TurnResponse
from src.stt.whisper_wrapper import build_stt
from src.tts.synth import build_tts


class VoiceLoop:
    def __init__(self) -> None:
        self.stt = build_stt()
        self.tts = build_tts()
        self.reasoner_factory = build_reasoner
        self._sessions: dict[str, SessionState] = {}

    def _get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    async def turn(self, req: TurnRequest) -> TurnResponse:
        state = self._get_or_create(req.session_id)
        turn_idx = len(state.history) // 2 + 1

        # 1. STT
        t_stt = time.perf_counter()
        stt = await self.stt.transcribe(req.audio_b64, req.text, req.language)
        stt_ms = int((time.perf_counter() - t_stt) * 1000)

        # 2. Reasoning
        t_llm = time.perf_counter()
        reasoner = self.reasoner_factory()
        if isinstance(reasoner, StubReasoner):
            result = reasoner.respond(transcript=stt.text, slots=state.slots)
        else:
            try:
                result = reasoner.respond(transcript=stt.text, slots=state.slots)
            except NotImplementedError:
                result = StubReasoner().respond(transcript=stt.text, slots=state.slots)
        state.slots = result.slots
        state.history.extend([
            {"role": "user", "content": stt.text},
            {"role": "assistant", "content": result.response_text},
        ])
        state.finished = result.finished
        llm_ms = int((time.perf_counter() - t_llm) * 1000)

        # 3. TTS
        t_tts = time.perf_counter()
        audio = await self.tts.synthesize(result.response_text, req.language)
        tts_ms = int((time.perf_counter() - t_tts) * 1000)

        return TurnResponse(
            session_id=req.session_id, turn=turn_idx,
            transcript=stt.text, response_text=result.response_text,
            response_audio_b64=audio.audio_b64, action=result.action,
            slots=result.slots, finished=result.finished,
            latency_stt_ms=stt_ms, latency_llm_ms=llm_ms, latency_tts_ms=tts_ms,
            latency_total_ms=stt_ms + llm_ms + tts_ms,
        )

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
