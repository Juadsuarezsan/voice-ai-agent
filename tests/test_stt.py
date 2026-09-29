"""Speech-to-text backends with faked whisper and mocked HTTP."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from typing import Any

import httpx
import numpy as np
import pytest
import respx

from src.config import Settings
from src.stt.whisper_wrapper import (
    WHISPER_API_URL,
    ConfigurationError,
    LocalWhisperSTT,
    STTError,
    StubSTT,
    WhisperAPISTT,
    build_stt,
    resample_linear,
)


@pytest.mark.asyncio
async def test_stub_echoes_text() -> None:
    out = await StubSTT().transcribe(None, "hello there", "en")
    assert out.text == "hello there" and out.confidence == 1.0 and out.backend == "stub"


@pytest.mark.asyncio
async def test_stub_reports_audio_duration(silence_wav_b64: str) -> None:
    out = await StubSTT().transcribe(silence_wav_b64, None, "en")
    assert out.text == "" and out.audio_seconds == pytest.approx(1.0)


def test_resample_changes_length() -> None:
    pcm = np.zeros(8000, dtype=np.float32)
    assert resample_linear(pcm, 8000, 16000).size == 16000
    assert resample_linear(pcm, 8000, 8000) is not None


class _FakeModel:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"n": int(np.asarray(audio).size), **kwargs})
        if self.delay:
            import time

            time.sleep(self.delay)
        return {
            "text": " Table for four. ",
            "language": "en",
            "segments": [{"no_speech_prob": 0.1}, {"no_speech_prob": 0.3}],
        }


def _fake_whisper(model: _FakeModel) -> ModuleType:
    mod = ModuleType("whisper")
    mod.load_model = lambda name: model  # type: ignore[attr-defined]
    return mod


@pytest.mark.asyncio
async def test_local_whisper_transcribes_and_resamples(tone_wav_b64: str) -> None:
    model = _FakeModel()
    stt = LocalWhisperSTT("tiny", module=_fake_whisper(model))
    out = await stt.transcribe(tone_wav_b64, None, "en")
    assert out.text == "Table for four."
    assert out.confidence == pytest.approx(0.8)
    assert out.backend == "whisper_local"
    assert model.calls[0]["n"] == int(1.4 * 16_000)
    assert model.calls[0]["fp16"] is False


@pytest.mark.asyncio
async def test_local_whisper_timeout_raises(tone_wav_b64: str) -> None:
    stt = LocalWhisperSTT("tiny", timeout_s=0.05, module=_fake_whisper(_FakeModel(delay=0.5)))
    with pytest.raises(STTError, match="exceeded"):
        await stt.transcribe(tone_wav_b64, None, "en")


def test_local_whisper_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "whisper", _fake_whisper(_FakeModel()))
    assert LocalWhisperSTT("base").model_name == "base"


@pytest.mark.asyncio
@respx.mock
async def test_whisper_api_success(tone_wav_b64: str) -> None:
    route = respx.post(WHISPER_API_URL).mock(
        return_value=httpx.Response(200, json={"text": " hello world ", "language": "en"})
    )
    stt = WhisperAPISTT("sk-test", timeout_s=2.0)
    out = await stt.transcribe(tone_wav_b64, None, "en")
    assert out.text == "hello world" and out.backend == "whisper_api"
    assert out.audio_seconds == pytest.approx(1.4)
    assert route.calls[0].request.headers["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
@respx.mock
async def test_whisper_api_retries_5xx(tone_wav_b64: str) -> None:
    route = respx.post(WHISPER_API_URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"text": "ok"})]
    )
    stt = WhisperAPISTT("sk-test", max_attempts=3)
    stt._post.retry.wait = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    out = await stt.transcribe(tone_wav_b64, None, "en")
    assert out.text == "ok" and route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_whisper_api_auth_error_not_retried(tone_wav_b64: str) -> None:
    route = respx.post(WHISPER_API_URL).mock(return_value=httpx.Response(401))
    stt = WhisperAPISTT("bad", max_attempts=3)
    with pytest.raises(STTError, match="401"):
        await stt.transcribe(tone_wav_b64, None, "en")
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_whisper_api_transport_error(tone_wav_b64: str) -> None:
    respx.post(WHISPER_API_URL).mock(side_effect=httpx.ConnectError("down"))
    stt = WhisperAPISTT("k", max_attempts=2)
    stt._post.retry.wait = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    with pytest.raises(STTError, match="unreachable"):
        await stt.transcribe(tone_wav_b64, None, "en")


def test_build_stt_errors_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(ConfigurationError, match="voice-agent\\[stt\\]"):
        build_stt(Settings(WHISPER_BACKEND="local"))
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_stt(Settings(WHISPER_BACKEND="api", OPENAI_API_KEY=None))
    assert isinstance(build_stt(Settings(WHISPER_BACKEND="stub")), StubSTT)
    assert isinstance(build_stt(Settings(WHISPER_BACKEND="api", OPENAI_API_KEY="k")), WhisperAPISTT)


def test_build_stt_local_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "whisper", _fake_whisper(_FakeModel()))
    stt = build_stt(Settings(WHISPER_BACKEND="local", WHISPER_MODEL="tiny"))
    assert isinstance(stt, LocalWhisperSTT)


def test_event_loop_not_required_for_construction() -> None:
    # Guard against accidentally awaiting in __init__.
    assert asyncio.iscoroutinefunction(StubSTT().transcribe)
