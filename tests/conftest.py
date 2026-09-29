"""Shared fixtures: deterministic seeds, isolated settings and stub pipelines."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

# Must run before src.api.main is imported: the limiter reads RATE_LIMIT at import.
os.environ.setdefault("RATE_LIMIT", "5/minute")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("WHISPER_BACKEND", "stub")
os.environ.setdefault("TTS_BACKEND", "stub")
os.environ.setdefault("LOGGER_BACKEND", "memory")
os.environ.pop("ANTHROPIC_API_KEY", None)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.agent.loop import VoiceLoop  # noqa: E402
from src.agent.vad import encode_wav_b64  # noqa: E402
from src.config import Settings, get_settings  # noqa: E402

SEED = 20260516


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Settings with every backend on its offline stub."""
    return Settings(
        WHISPER_BACKEND="stub",
        TTS_BACKEND="stub",
        VAD_BACKEND="energy",
        LOGGER_BACKEND="memory",
        ANTHROPIC_API_KEY=None,
        LOG_LEVEL="WARNING",
    )


@pytest.fixture
def voice_loop(settings: Settings) -> VoiceLoop:
    """Fully offline pipeline."""
    return VoiceLoop(settings)


@pytest.fixture
def silence_wav_b64() -> str:
    """One second of digital silence at 16 kHz."""
    return encode_wav_b64(np.zeros(16_000, dtype=np.float32), 16_000)


@pytest.fixture
def tone_wav_b64() -> str:
    """0.8 s of a 220 Hz tone followed by 0.6 s of silence (a finished turn)."""
    t = np.arange(int(16_000 * 0.8), dtype=np.float32) / 16_000
    tone = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    pcm = np.concatenate([tone, np.zeros(int(16_000 * 0.6), dtype=np.float32)])
    return encode_wav_b64(pcm.astype(np.float32), 16_000)


BOOKING_UTTERANCES: list[str] = [
    "Hi, I'd like to book a table.",
    "Four people, please.",
    "Tomorrow evening.",
    "7:30 pm.",
    "Under the name John Smith.",
    "Yes, that's right.",
]


@pytest.fixture
def booking_utterances() -> list[str]:
    """A six-turn conversation that completes a booking with the stub."""
    return list(BOOKING_UTTERANCES)
