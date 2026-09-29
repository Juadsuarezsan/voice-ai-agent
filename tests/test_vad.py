"""Voice activity detection: energy detector, Silero wrapper (faked) and WAV codec."""

from __future__ import annotations

import base64
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.agent.vad import (
    AudioDecodeError,
    EnergyVAD,
    SileroVAD,
    build_vad,
    decode_wav_b64,
    encode_wav_b64,
)
from src.config import Settings


def test_wav_roundtrip() -> None:
    pcm = (0.3 * np.sin(np.linspace(0, 100, 8000))).astype(np.float32)
    decoded, rate = decode_wav_b64(encode_wav_b64(pcm, 8000))
    assert rate == 8000
    assert decoded.shape == pcm.shape
    assert np.allclose(decoded, pcm, atol=1e-3)


def test_decode_rejects_garbage() -> None:
    with pytest.raises(AudioDecodeError):
        decode_wav_b64(base64.b64encode(b"RIFF....WAVEjunk").decode())


def test_energy_vad_silence(silence_wav_b64: str) -> None:
    pcm, rate = decode_wav_b64(silence_wav_b64)
    result = EnergyVAD().detect(pcm, rate)
    assert result.speech_detected is False
    assert result.end_of_turn is True
    assert result.duration_s == pytest.approx(1.0)


def test_energy_vad_tone_with_trailing_silence(tone_wav_b64: str) -> None:
    pcm, rate = decode_wav_b64(tone_wav_b64)
    result = EnergyVAD(min_silence_ms=500).detect(pcm, rate)
    assert result.speech_detected is True
    assert 0.4 < result.speech_ratio < 0.7
    assert result.end_of_turn is True


def test_energy_vad_no_end_of_turn_without_silence() -> None:
    t = np.arange(16_000, dtype=np.float32) / 16_000
    pcm = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    result = EnergyVAD(min_silence_ms=500).detect(pcm, 16_000)
    assert result.speech_detected is True
    assert result.end_of_turn is False


def test_energy_vad_empty_input() -> None:
    result = EnergyVAD().detect(np.zeros(0, dtype=np.float32), 16_000)
    assert result.speech_detected is False and result.duration_s == 0.0


def _fake_silero(segments: list[dict[str, int]]) -> ModuleType:
    mod = ModuleType("silero_vad")
    mod.load_silero_vad = lambda: "model"  # type: ignore[attr-defined]
    calls: list[dict[str, Any]] = []

    def get_speech_timestamps(tensor: Any, model: Any, **kwargs: Any) -> list[dict[str, int]]:
        calls.append(kwargs)
        return segments

    mod.get_speech_timestamps = get_speech_timestamps  # type: ignore[attr-defined]
    mod.calls = calls  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    torch = ModuleType("torch")
    torch.from_numpy = lambda arr: SimpleNamespace(shape=arr.shape)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_silero_vad_detects_segments(fake_torch: ModuleType) -> None:
    mod = _fake_silero([{"start": 0, "end": 8000}])
    vad = SileroVAD(threshold=0.6, min_silence_ms=300, module=mod)
    pcm = np.zeros(16_000, dtype=np.float32)
    result = vad.detect(pcm, 16_000)
    assert result.backend == "silero"
    assert result.speech_detected is True
    assert result.speech_ratio == pytest.approx(0.5)
    assert result.end_of_turn is True  # 500 ms trailing silence >= 300 ms
    assert mod.calls[0]["threshold"] == 0.6  # type: ignore[attr-defined]
    assert mod.calls[0]["sampling_rate"] == 16_000  # type: ignore[attr-defined]


def test_silero_vad_mid_speech_is_not_end_of_turn(fake_torch: ModuleType) -> None:
    vad = SileroVAD(module=_fake_silero([{"start": 0, "end": 15_900}]))
    result = vad.detect(np.zeros(16_000, dtype=np.float32), 16_000)
    assert result.speech_detected is True and result.end_of_turn is False


def test_silero_vad_no_speech(fake_torch: ModuleType) -> None:
    vad = SileroVAD(module=_fake_silero([]))
    result = vad.detect(np.zeros(16_000, dtype=np.float32), 16_000)
    assert result.speech_detected is False and result.end_of_turn is True


def test_silero_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "silero_vad", _fake_silero([]))
    vad = SileroVAD()
    assert vad.model == "model"


def test_build_vad_falls_back_when_silero_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "silero_vad", None)  # forces ImportError
    vad = build_vad(Settings(VAD_BACKEND="silero"))
    assert isinstance(vad, EnergyVAD)


def test_build_vad_energy_default() -> None:
    assert isinstance(build_vad(Settings(VAD_BACKEND="energy")), EnergyVAD)
