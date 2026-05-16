import base64

import pytest

from src.tts.synth import StubTTS


@pytest.mark.asyncio
async def test_stub_returns_valid_wav():
    tts = StubTTS()
    r = await tts.synthesize("hello world")
    assert r.audio_b64 is not None
    raw = base64.b64decode(r.audio_b64)
    assert raw[:4] == b"RIFF"
    assert raw[8:12] == b"WAVE"
