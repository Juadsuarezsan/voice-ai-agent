"""Text-to-speech backends with mocked HTTP and a faked XTTS module."""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from typing import Any

import httpx
import pytest
import respx

from src.config import Settings
from src.stt.whisper_wrapper import ConfigurationError
from src.tts.synth import ElevenLabsTTS, StubTTS, TTSError, XTTSLocalTTS, build_tts

VOICE_URL = "https://api.elevenlabs.io/v1/text-to-speech/voice-1"


@pytest.mark.asyncio
async def test_stub_returns_valid_wav_scaled_by_words() -> None:
    short = await StubTTS().synthesize("hello")
    long = await StubTTS().synthesize("hello " * 10)
    assert short.audio_b64 is not None and long.audio_b64 is not None
    raw = base64.b64decode(short.audio_b64)
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert long.duration_ms > short.duration_ms
    assert short.characters == 5


@pytest.mark.asyncio
@respx.mock
async def test_elevenlabs_success() -> None:
    route = respx.post(VOICE_URL).mock(return_value=httpx.Response(200, content=b"ID3mp3bytes"))
    tts = ElevenLabsTTS("xi-key", "voice-1", timeout_s=2.0)
    out = await tts.synthesize("Hello there", "en")
    assert out.mime == "audio/mpeg" and out.backend == "elevenlabs"
    assert base64.b64decode(out.audio_b64 or "") == b"ID3mp3bytes"
    assert out.characters == 11
    req = route.calls[0].request
    assert req.headers["xi-api-key"] == "xi-key"
    assert b'"model_id": "eleven_multilingual_v2"' in req.content
    assert "output_format=mp3_44100_128" in str(req.url)


@pytest.mark.asyncio
@respx.mock
async def test_elevenlabs_retries_429() -> None:
    route = respx.post(VOICE_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, content=b"ok")]
    )
    tts = ElevenLabsTTS("k", "voice-1", max_attempts=3)
    tts._post.retry.wait = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    out = await tts.synthesize("Hi")
    assert base64.b64decode(out.audio_b64 or "") == b"ok" and route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_elevenlabs_client_error_not_retried() -> None:
    route = respx.post(VOICE_URL).mock(return_value=httpx.Response(400))
    tts = ElevenLabsTTS("k", "voice-1", max_attempts=3)
    with pytest.raises(TTSError, match="400"):
        await tts.synthesize("Hi")
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_elevenlabs_transport_error() -> None:
    respx.post(VOICE_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    tts = ElevenLabsTTS("k", "voice-1", max_attempts=2)
    tts._post.retry.wait = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    with pytest.raises(TTSError, match="unreachable"):
        await tts.synthesize("Hi")


@pytest.mark.asyncio
async def test_elevenlabs_empty_text_short_circuits() -> None:
    out = await ElevenLabsTTS("k", "voice-1").synthesize("   ")
    assert out.audio_b64 is None


def _fake_tts_module() -> ModuleType:
    mod = ModuleType("TTS.api")

    class _TTS:
        def __init__(self, name: str) -> None:
            self.name = name

        def tts(self, **kwargs: Any) -> list[float]:
            return [0.0, 0.5, -0.5, 0.0] * 6000

    mod.TTS = _TTS  # type: ignore[attr-defined]
    return mod


@pytest.mark.asyncio
async def test_xtts_wraps_float_pcm_as_wav() -> None:
    tts = XTTSLocalTTS(module=_fake_tts_module())
    out = await tts.synthesize("hola", "es")
    assert out.mime == "audio/wav" and out.backend == "xtts"
    assert out.duration_ms == 1000
    assert base64.b64decode(out.audio_b64 or "")[:4] == b"RIFF"


def test_build_tts_errors_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="ELEVENLABS_API_KEY"):
        build_tts(Settings(TTS_BACKEND="elevenlabs", ELEVENLABS_API_KEY=None))
    monkeypatch.setitem(sys.modules, "TTS.api", None)
    monkeypatch.setitem(sys.modules, "TTS", None)
    with pytest.raises(ConfigurationError, match="voice-agent\\[tts\\]"):
        build_tts(Settings(TTS_BACKEND="xtts"))
    assert isinstance(build_tts(Settings(TTS_BACKEND="stub")), StubTTS)
    assert isinstance(
        build_tts(Settings(TTS_BACKEND="elevenlabs", ELEVENLABS_API_KEY="k")), ElevenLabsTTS
    )


def test_build_tts_xtts_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "TTS.api", _fake_tts_module())
    assert isinstance(build_tts(Settings(TTS_BACKEND="xtts")), XTTSLocalTTS)
