"""End-to-end turns through the LangGraph pipeline with stub backends."""

from __future__ import annotations

from typing import Any

import pytest

from src.agent.loop import VoiceLoop
from src.agent.reasoner import ReasonResult
from src.api.schemas import Slot, TurnRequest, TurnResponse
from src.config import Settings
from src.observability import llm_cost_usd
from src.security.pii import pseudonym


@pytest.mark.asyncio
async def test_six_turn_booking_completes(
    voice_loop: VoiceLoop, booking_utterances: list[str]
) -> None:
    responses: list[TurnResponse] = []
    for utterance in booking_utterances:
        responses.append(await voice_loop.turn(TurnRequest(session_id="e2e-1", text=utterance)))
    assert [r.turn for r in responses] == [1, 2, 3, 4, 5, 6]
    assert all(r.action == "ask_clarification" for r in responses[:-1])
    assert responses[-1].action == "book" and responses[-1].finished is True
    final_slots = {s.name: s.value for s in responses[-1].slots}
    assert final_slots == {
        "party_size": 4,
        "date": "tomorrow",
        "time": "19:30",
        "name": "John Smith",
    }
    assert responses[-1].backends == {"stt": "stub", "reasoner": "stub", "tts": "stub"}


@pytest.mark.asyncio
async def test_latency_breakdown_and_trace_id(voice_loop: VoiceLoop) -> None:
    out = await voice_loop.turn(TurnRequest(session_id="e2e-2", text="hello"))
    assert len(out.trace_id) == 32
    lat = out.latency
    assert lat.total_ms >= lat.stt_ms + lat.llm_ms + lat.tts_ms
    assert out.response_audio_b64 is not None
    assert out.usage.cost_usd == 0.0  # stub reasoner bills no tokens


@pytest.mark.asyncio
async def test_silence_short_circuits_to_no_speech(
    voice_loop: VoiceLoop, silence_wav_b64: str
) -> None:
    out = await voice_loop.turn(TurnRequest(session_id="e2e-3", audio_b64=silence_wav_b64))
    assert out.speech_detected is False
    assert "didn't catch" in out.response_text
    assert out.backends["vad"] == "energy" and "stt" not in out.backends
    # A silent turn must not advance the session's history.
    again = await voice_loop.turn(TurnRequest(session_id="e2e-3", text="hi"))
    assert again.turn == 1


@pytest.mark.asyncio
async def test_audio_with_speech_runs_stt(voice_loop: VoiceLoop, tone_wav_b64: str) -> None:
    out = await voice_loop.turn(TurnRequest(session_id="e2e-4", audio_b64=tone_wav_b64))
    assert out.speech_detected is True
    assert out.backends["stt"] == "stub"
    assert out.usage.audio_seconds == pytest.approx(1.4)


@pytest.mark.asyncio
async def test_transcript_is_persisted_and_redacted(
    voice_loop: VoiceLoop, booking_utterances: list[str]
) -> None:
    for utterance in booking_utterances:
        await voice_loop.turn(TurnRequest(session_id="e2e-5", text=utterance))
    transcript = voice_loop.transcript("e2e-5")
    assert len(transcript.turns) == 6 and transcript.finished is True
    fifth = transcript.turns[4]
    assert "John Smith" not in str(fifth["transcript"])
    assert pseudonym("John Smith") in str(fifth["transcript"])
    name_slot = next(s for s in fifth["slots"] if s["name"] == "name")
    assert name_slot["value"] == pseudonym("John Smith")
    assert fifth["latency"]["llm"] >= 0 and fifth["usage"]["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_reset_clears_session(voice_loop: VoiceLoop) -> None:
    await voice_loop.turn(TurnRequest(session_id="e2e-6", text="Table for 2"))
    assert voice_loop.reset("e2e-6") is True
    assert voice_loop.reset("e2e-6") is False
    out = await voice_loop.turn(TurnRequest(session_id="e2e-6", text="hello"))
    assert out.turn == 1 and out.slots == []


class _FakeLLM:
    name = "claude"

    def respond(self, **kwargs: Any) -> ReasonResult:
        return ReasonResult(
            response_text="How many people?",
            action="ask_clarification",
            slots=[Slot(name="date", value="friday", confirmed=True)],
            input_tokens=1000,
            output_tokens=100,
            backend="claude",
        )


@pytest.mark.asyncio
async def test_cost_is_computed_from_tokens(settings: Settings) -> None:
    loop = VoiceLoop(settings, reasoner=_FakeLLM())
    out = await loop.turn(TurnRequest(session_id="e2e-7", text="Friday"))
    expected = llm_cost_usd(1000, 100, price_in_per_mtok=3.0, price_out_per_mtok=15.0)
    assert out.usage.cost_usd == pytest.approx(expected) == pytest.approx(0.0045)
    assert out.backends["reasoner"] == "claude"
    assert out.usage.input_tokens == 1000


def test_graph_node_order(voice_loop: VoiceLoop) -> None:
    assert voice_loop.pipeline.node_names() == [
        "vad_node",
        "stt_node",
        "reasoner_node",
        "tts_node",
        "logger_node",
    ]
