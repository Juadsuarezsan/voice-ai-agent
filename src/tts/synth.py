"""Text-to-speech backends.

* :class:`StubTTS` renders a deterministic placeholder tone (not speech) whose
  length tracks the text, so the browser demo has audible feedback offline.
* :class:`ElevenLabsTTS` calls the ElevenLabs REST API directly with ``httpx``
  (timeout + tenacity) -- the vendor SDK is not needed for one endpoint.
* :class:`XTTSLocalTTS` wraps Coqui XTTS-v2 with a lazy import
  (``pip install "voice-agent[tts]"``).
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import wave
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import numpy as np
import numpy.typing as npt
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class TTSError(RuntimeError):
    """Raised when a text-to-speech backend fails after retries."""


@dataclass(frozen=True)
class TTSResult:
    """Synthesized audio.

    Attributes:
        audio_b64: Base64 audio bytes.
        mime: MIME type of the audio (``audio/wav`` or ``audio/mpeg``).
        duration_ms: Estimated playback duration.
        characters: Characters billed/synthesized.
        backend: Backend name.
    """

    audio_b64: str | None
    mime: str = "audio/wav"
    duration_ms: int = 0
    characters: int = 0
    backend: str = "stub"


class TextToSpeech(Protocol):
    """Anything that turns text into audio."""

    name: str

    async def synthesize(self, text: str, language: str = "en") -> TTSResult:
        """Synthesize ``text``."""
        ...


def _wav_b64(pcm16: npt.NDArray[np.int16], sample_rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


class StubTTS:
    """Deterministic placeholder tone: 90 ms per word, soft 440 Hz sine."""

    name = "stub"
    sample_rate = 16_000

    async def synthesize(self, text: str, language: str = "en") -> TTSResult:
        """Render a tone whose duration is proportional to the word count."""
        words = max(1, len(text.split()))
        duration_ms = min(3000, 200 + 90 * words)
        n = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        envelope = np.minimum(1.0, np.minimum(t, duration_ms / 1000 - t) * 20).astype(np.float32)
        tone = 0.2 * np.sin(2 * np.pi * 440.0 * t) * envelope
        pcm16 = (tone * 32767).astype(np.int16)
        return TTSResult(
            audio_b64=_wav_b64(pcm16, self.sample_rate),
            mime="audio/wav",
            duration_ms=duration_ms,
            characters=len(text),
            backend=self.name,
        )


def _is_retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


class ElevenLabsTTS:
    """ElevenLabs ``text-to-speech`` endpoint over HTTPS (MP3 output)."""

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
        timeout_s: float = 15.0,
        max_attempts: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create the client.

        Args:
            api_key: ElevenLabs API key.
            voice_id: Voice identifier.
            model_id: ElevenLabs model (``eleven_multilingual_v2`` covers EN/ES).
            timeout_s: Per-request timeout.
            max_attempts: Total attempts for retryable failures.
            client: Optional shared ``httpx.AsyncClient``.
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.client = client or httpx.AsyncClient(timeout=timeout_s)
        self._post = retry(
            retry=retry_if_exception(_is_retryable_http),
            stop=stop_after_attempt(max(1, max_attempts)),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )(self._post_once)

    async def _post_once(self, text: str, language: str) -> bytes:
        response = await self.client.post(
            ELEVENLABS_URL.format(voice_id=self.voice_id),
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": self.model_id, "language_code": language},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.content

    async def synthesize(self, text: str, language: str = "en") -> TTSResult:
        """Synthesize ``text`` to MP3.

        Raises:
            TTSError: When the API keeps failing or returns a non-2xx status.
        """
        if not text.strip():
            return TTSResult(audio_b64=None, mime="audio/mpeg", backend=self.name)
        try:
            audio = await self._post(text, language)
        except httpx.HTTPStatusError as exc:
            raise TTSError(f"ElevenLabs returned {exc.response.status_code}") from exc
        except httpx.TransportError as exc:
            raise TTSError(f"ElevenLabs unreachable: {exc}") from exc
        return TTSResult(
            audio_b64=base64.b64encode(audio).decode(),
            mime="audio/mpeg",
            duration_ms=int(len(text) * 65),
            characters=len(text),
            backend=self.name,
        )


class XTTSLocalTTS:
    """Coqui XTTS-v2 running locally (lazy import; slow on CPU)."""

    name = "xtts"
    sample_rate = 24_000

    def __init__(self, speaker_wav: str | None = None, module: Any | None = None) -> None:
        """Load XTTS-v2.

        Args:
            speaker_wav: Optional reference clip for voice cloning.
            module: Pre-imported ``TTS.api`` module (tests inject a fake).

        Raises:
            ImportError: If ``coqui-tts`` is not installed.
        """
        mod: Any = module if module is not None else importlib.import_module("TTS.api")
        self.model = mod.TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.speaker_wav = speaker_wav

    async def synthesize(self, text: str, language: str = "en") -> TTSResult:
        """Synthesize in an executor thread and wrap the float PCM as WAV."""
        loop = asyncio.get_running_loop()

        def _run() -> list[float]:
            out: list[float] = self.model.tts(
                text=text, language=language, speaker_wav=self.speaker_wav
            )
            return out

        samples = np.asarray(await loop.run_in_executor(None, _run), dtype=np.float32)
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        return TTSResult(
            audio_b64=_wav_b64(pcm16, self.sample_rate),
            mime="audio/wav",
            duration_ms=int(samples.size / self.sample_rate * 1000),
            characters=len(text),
            backend=self.name,
        )


def build_tts(settings: Settings) -> TextToSpeech:
    """Construct the TTS backend named by ``TTS_BACKEND``.

    Args:
        settings: Active settings.

    Returns:
        The configured backend.

    Raises:
        ConfigurationError: If ``elevenlabs`` is requested without an API key or
            ``xtts`` without ``coqui-tts`` installed.
    """
    from src.stt.whisper_wrapper import ConfigurationError  # noqa: PLC0415 -- avoid cycle

    if settings.tts_backend == "elevenlabs":
        if not settings.elevenlabs_api_key:
            raise ConfigurationError("TTS_BACKEND=elevenlabs requires ELEVENLABS_API_KEY")
        logger.info("tts backend=elevenlabs voice={}", settings.elevenlabs_voice_id)
        return ElevenLabsTTS(
            settings.elevenlabs_api_key,
            settings.elevenlabs_voice_id,
            settings.elevenlabs_model_id,
            settings.tts_timeout_s,
        )
    if settings.tts_backend == "xtts":
        try:
            tts = XTTSLocalTTS()
        except ImportError as exc:
            raise ConfigurationError(
                'TTS_BACKEND=xtts requires `pip install "voice-agent[tts]"`'
            ) from exc
        logger.info("tts backend=xtts")
        return tts
    logger.info("tts backend=stub")
    return StubTTS()
