"""LangGraph orchestration of one conversational turn.

The graph has five nodes wired in a straight line with one conditional branch::

    vad_node ──(speech)──> stt_node ──> reasoner_node ──> tts_node ──> logger_node
        └────(silence)───> no_speech_node ───────────────────┘

Node names end in ``_node`` so they can never collide with a state key (LangGraph
rejects a node whose name equals a channel). Every node is wrapped by
:func:`_timed`, which logs its input/output summary with the turn's ``trace_id``
and records its wall-clock latency into ``state["latency"]``.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from src.agent.reasoner import Reasoner
from src.agent.vad import VoiceActivityDetector, decode_wav_b64
from src.api.schemas import Action, Slot
from src.config import Settings
from src.observability import llm_cost_usd, stt_cost_usd, tts_cost_usd
from src.security.pii import names_from_slots, redact_slots, redact_text
from src.storage.logger import ConversationLogger, TurnRecord
from src.stt.whisper_wrapper import SpeechToText
from src.tts.synth import TextToSpeech

NO_SPEECH_REPLY = "Sorry, I didn't catch that. Could you say it again?"


class TurnState(TypedDict, total=False):
    """Channels flowing through the graph for one turn."""

    session_id: str
    trace_id: str
    turn: int
    language: str
    text_in: str | None
    audio_b64: str | None
    speech_detected: bool
    audio_seconds: float
    transcript: str
    history: list[dict[str, str]]
    slots: list[Slot]
    awaiting_confirmation: bool
    response_text: str
    action: Action
    finished: bool
    audio_out_b64: str | None
    audio_mime: str
    tts_characters: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency: dict[str, float]
    backends: dict[str, str]


NodeFn = Callable[[TurnState], Awaitable[dict[str, Any]]]


def _timed(name: str, fn: NodeFn) -> NodeFn:
    """Wrap ``fn`` with latency measurement and structured entry/exit logs."""

    async def wrapped(state: TurnState) -> dict[str, Any]:
        t0 = time.perf_counter()
        logger.debug("node={} enter turn={} keys={}", name, state.get("turn"), sorted(state.keys()))
        out = await fn(state)
        ms = round((time.perf_counter() - t0) * 1000, 3)
        latency = dict(state.get("latency", {}))
        latency[name] = ms
        out["latency"] = latency
        logger.info(
            "node={} exit ms={} out={}",
            name,
            ms,
            {
                k: (v if isinstance(v, bool | int | float) else type(v).__name__)
                for k, v in out.items()
                if k != "latency"
            },
        )
        return out

    return wrapped


class VoicePipeline:
    """Owns the backends and the compiled LangGraph for one deployment."""

    def __init__(
        self,
        settings: Settings,
        *,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        reasoner: Reasoner,
        tts: TextToSpeech,
        conversation_logger: ConversationLogger,
    ) -> None:
        """Bind the backends and compile the graph once."""
        self.settings = settings
        self.vad = vad
        self.stt = stt
        self.reasoner = reasoner
        self.tts = tts
        self.conversation_logger = conversation_logger
        self.graph = self._build()

    # -- nodes -----------------------------------------------------------------

    async def _vad_node(self, state: TurnState) -> dict[str, Any]:
        if not state.get("audio_b64"):
            return {"speech_detected": True, "audio_seconds": 0.0}
        pcm, rate = decode_wav_b64(state["audio_b64"] or "")
        result = self.vad.detect(pcm, rate)
        backends = dict(state.get("backends", {}))
        backends["vad"] = result.backend
        return {
            "speech_detected": result.speech_detected,
            "audio_seconds": result.duration_s,
            "backends": backends,
        }

    async def _no_speech_node(self, state: TurnState) -> dict[str, Any]:
        return {
            "transcript": "",
            "response_text": NO_SPEECH_REPLY,
            "action": "ask_clarification",
            "finished": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    async def _stt_node(self, state: TurnState) -> dict[str, Any]:
        result = await self.stt.transcribe(
            state.get("audio_b64"), state.get("text_in"), state.get("language", "en")
        )
        backends = dict(state.get("backends", {}))
        backends["stt"] = result.backend
        return {
            "transcript": result.text,
            "audio_seconds": result.audio_seconds or state.get("audio_seconds", 0.0),
            "backends": backends,
        }

    async def _reasoner_node(self, state: TurnState) -> dict[str, Any]:
        result = self.reasoner.respond(
            transcript=state.get("transcript", ""),
            slots=list(state.get("slots", [])),
            history=list(state.get("history", [])),
            awaiting_confirmation=state.get("awaiting_confirmation", False),
        )
        backends = dict(state.get("backends", {}))
        backends["reasoner"] = result.backend
        return {
            "response_text": result.response_text,
            "action": result.action,
            "slots": result.slots,
            "finished": result.finished,
            "awaiting_confirmation": result.awaiting_confirmation,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "backends": backends,
        }

    async def _tts_node(self, state: TurnState) -> dict[str, Any]:
        result = await self.tts.synthesize(
            state.get("response_text", ""), state.get("language", "en")
        )
        backends = dict(state.get("backends", {}))
        backends["tts"] = result.backend
        return {
            "audio_out_b64": result.audio_b64,
            "audio_mime": result.mime,
            "tts_characters": result.characters,
            "backends": backends,
        }

    async def _logger_node(self, state: TurnState) -> dict[str, Any]:
        s = self.settings
        backends = state.get("backends", {})
        cost = llm_cost_usd(
            state.get("input_tokens", 0),
            state.get("output_tokens", 0),
            price_in_per_mtok=s.price_input_per_mtok,
            price_out_per_mtok=s.price_output_per_mtok,
        )
        if backends.get("tts") == "elevenlabs":
            cost += tts_cost_usd(
                state.get("tts_characters", 0), price_per_1k_chars=s.elevenlabs_price_per_1k_chars
            )
        if backends.get("stt") == "whisper_api":
            cost += stt_cost_usd(
                state.get("audio_seconds", 0.0), price_per_min=s.whisper_api_price_per_min
            )
        cost = round(cost, 8)

        slots_dump = [sl.model_dump() for sl in state.get("slots", [])]
        transcript = state.get("transcript", "")
        response_text = state.get("response_text", "")
        if s.redact_pii:
            names = names_from_slots(slots_dump)
            transcript = redact_text(transcript, names)
            response_text = redact_text(response_text, names)
            slots_dump = redact_slots(slots_dump)

        latency = dict(state.get("latency", {}))
        record = TurnRecord(
            session_id=state["session_id"],
            trace_id=state["trace_id"],
            turn=state["turn"],
            transcript=transcript,
            response_text=response_text,
            action=state.get("action", "respond"),
            finished=state.get("finished", False),
            slots=slots_dump,
            latency=latency,
            usage={
                "input_tokens": state.get("input_tokens", 0),
                "output_tokens": state.get("output_tokens", 0),
                "tts_characters": state.get("tts_characters", 0),
                "audio_seconds": state.get("audio_seconds", 0.0),
                "cost_usd": cost,
            },
            backends=dict(backends),
        )
        self.conversation_logger.log_turn(record)
        logger.info(
            "turn logged session={} turn={} action={} tokens_in={} tokens_out={} cost_usd={}",
            record.session_id,
            record.turn,
            record.action,
            record.usage["input_tokens"],
            record.usage["output_tokens"],
            cost,
        )
        return {"cost_usd": cost}

    # -- graph -------------------------------------------------------------------

    @staticmethod
    def _route_after_vad(state: TurnState) -> str:
        return "stt_node" if state.get("speech_detected", True) else "no_speech_node"

    def _build(self) -> Any:
        graph: StateGraph = StateGraph(TurnState)
        graph.add_node("vad_node", _timed("vad", self._vad_node))
        graph.add_node("no_speech_node", _timed("no_speech", self._no_speech_node))
        graph.add_node("stt_node", _timed("stt", self._stt_node))
        graph.add_node("reasoner_node", _timed("llm", self._reasoner_node))
        graph.add_node("tts_node", _timed("tts", self._tts_node))
        graph.add_node("logger_node", _timed("logger", self._logger_node))
        graph.set_entry_point("vad_node")
        graph.add_conditional_edges(
            "vad_node",
            self._route_after_vad,
            {"stt_node": "stt_node", "no_speech_node": "no_speech_node"},
        )
        graph.add_edge("stt_node", "reasoner_node")
        graph.add_edge("reasoner_node", "tts_node")
        graph.add_edge("no_speech_node", "tts_node")
        graph.add_edge("tts_node", "logger_node")
        graph.add_edge("logger_node", END)
        return graph.compile()

    async def run(self, state: TurnState) -> TurnState:
        """Execute the graph for one turn and return the final state."""
        out: TurnState = await self.graph.ainvoke(state)
        return out

    def node_names(self) -> list[str]:
        """Return the node names in execution order (for docs and tests)."""
        return ["vad_node", "stt_node", "reasoner_node", "tts_node", "logger_node"]
