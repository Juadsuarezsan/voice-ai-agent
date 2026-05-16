import pytest

from src.eval.runner import run_eval


@pytest.mark.asyncio
async def test_resolution_rate_acceptable(monkeypatch):
    monkeypatch.setenv("WHISPER_BACKEND", "stub")
    monkeypatch.setenv("TTS_BACKEND", "stub")
    r = await run_eval()
    assert r["n"] == 5
    # With the slot-filling stub the conversations should all complete
    assert r["resolution_rate"] >= 0.80
