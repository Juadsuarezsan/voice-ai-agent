import pytest

from src.agent.loop import VoiceLoop
from src.api.schemas import TurnRequest


@pytest.mark.asyncio
async def test_multi_turn_booking_completes(monkeypatch):
    monkeypatch.setenv("WHISPER_BACKEND", "stub")
    monkeypatch.setenv("TTS_BACKEND", "stub")
    loop = VoiceLoop()

    utterances = [
        "I'd like to book a table.",
        "Four people, please.",
        "Tomorrow at 7:30 pm.",
        "Under the name John Smith.",
    ]
    last = None
    for u in utterances:
        last = await loop.turn(TurnRequest(session_id="t1", text=u))
        if last.finished:
            break
    assert last is not None
    assert last.finished is True
    assert last.action == "book"


@pytest.mark.asyncio
async def test_turn_response_contains_latency_breakdown(monkeypatch):
    monkeypatch.setenv("WHISPER_BACKEND", "stub")
    monkeypatch.setenv("TTS_BACKEND", "stub")
    loop = VoiceLoop()
    out = await loop.turn(TurnRequest(session_id="t2", text="hello"))
    assert out.latency_stt_ms >= 0
    assert out.latency_llm_ms >= 0
    assert out.latency_tts_ms >= 0
    assert out.latency_total_ms == out.latency_stt_ms + out.latency_llm_ms + out.latency_tts_ms
    assert out.response_audio_b64 is not None
