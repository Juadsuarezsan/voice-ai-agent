"""Speech-to-text backends.

* :class:`StubSTT` echoes the text field (or returns an empty transcript for
  audio) so the whole pipeline runs offline.
* :class:`LocalWhisperSTT` runs ``openai-whisper`` on CPU/GPU; the package is
  imported lazily inside the constructor (``pip install "voice-agent[stt]"``).
* :class:`WhisperAPISTT` calls the hosted Whisper endpoint over HTTPS with an
  explicit timeout and tenacity retries.
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
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.agent.vad import AudioDecodeError, FloatArray, decode_wav_b64
from src.config import Settings

WHISPER_SAMPLE_RATE = 16_000
WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"


class STTError(RuntimeError):
    """Raised when a speech-to-text backend fails after retries."""


class ConfigurationError(RuntimeError):
    """Raised when the configured backend cannot be constructed."""


@dataclass(frozen=True)
class STTResult:
    """Transcription result.

    Attributes:
        text: Transcript (may be empty).
        language: Detected or requested language code.
        confidence: Backend confidence in ``[0, 1]`` (1.0 for pass-through text).
        audio_seconds: Duration of the processed audio (0 for text input).
        backend: Backend name.
    """

    text: str
    language: str
    confidence: float
    audio_seconds: float = 0.0
    backend: str = "stub"


class SpeechToText(Protocol):
    """Anything that turns audio (or pre-transcribed text) into a transcript."""

    name: str

    async def transcribe(self, audio_b64: str | None, text: str | None, language: str) -> STTResult:
        """Transcribe ``audio_b64`` unless ``text`` is already provided."""
        ...


def _audio_seconds(audio_b64: str) -> float:
    """Return the duration of a base64 WAV without decoding samples."""
    try:
        raw = base64.b64decode(audio_b64, validate=True)
        with wave.open(io.BytesIO(raw)) as wf:
            return float(round(wf.getnframes() / max(1, wf.getframerate()), 3))
    except (ValueError, wave.Error, EOFError) as exc:
        raise AudioDecodeError(f"cannot read WAV header: {exc}") from exc


def resample_linear(samples: FloatArray, src_rate: int, dst_rate: int) -> FloatArray:
    """Resample mono PCM with linear interpolation.

    Args:
        samples: Float32 mono samples.
        src_rate: Source sample rate.
        dst_rate: Target sample rate.

    Returns:
        Resampled float32 array (unchanged when the rates match).
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.size / src_rate
    n_out = int(round(duration * dst_rate))
    src_t = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(dst_t, src_t, samples).astype(np.float32)


class StubSTT:
    """Pass-through STT used for tests, CI and the offline demo."""

    name = "stub"

    async def transcribe(self, audio_b64: str | None, text: str | None, language: str) -> STTResult:
        """Echo ``text`` or return an empty transcript with the audio duration."""
        if text:
            return STTResult(text=text, language=language, confidence=1.0, backend=self.name)
        seconds = _audio_seconds(audio_b64) if audio_b64 else 0.0
        return STTResult(
            text="", language=language, confidence=0.0, audio_seconds=seconds, backend=self.name
        )


class LocalWhisperSTT:
    """``openai-whisper`` running in-process (lazy import, executor thread)."""

    name = "whisper_local"

    def __init__(
        self, model_name: str = "base", timeout_s: float = 30.0, module: Any | None = None
    ) -> None:
        """Load the model.

        Args:
            model_name: Whisper checkpoint (``tiny``, ``base``, ..., ``large-v3``).
            timeout_s: Maximum seconds allowed per transcription.
            module: Pre-imported ``whisper`` module (tests inject a fake).

        Raises:
            ImportError: If ``openai-whisper`` is not installed.
        """
        mod: Any = module if module is not None else importlib.import_module("whisper")
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.model = mod.load_model(model_name)

    async def transcribe(self, audio_b64: str | None, text: str | None, language: str) -> STTResult:
        """Transcribe base64 WAV audio with Whisper.

        Raises:
            STTError: If inference exceeds ``timeout_s``.
            AudioDecodeError: If the payload is not decodable WAV.
        """
        if text:
            return STTResult(text=text, language=language, confidence=1.0, backend=self.name)
        if not audio_b64:
            return STTResult(text="", language=language, confidence=0.0, backend=self.name)
        samples, rate = decode_wav_b64(audio_b64)
        pcm16k = resample_linear(samples, rate, WHISPER_SAMPLE_RATE)

        def _run() -> dict[str, Any]:
            out: dict[str, Any] = self.model.transcribe(pcm16k, language=language, fp16=False)
            return out

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _run), timeout=self.timeout_s
            )
        except TimeoutError as exc:
            raise STTError(f"whisper {self.model_name} exceeded {self.timeout_s}s") from exc
        segments = result.get("segments") or []
        probs = [float(s.get("no_speech_prob", 0.0)) for s in segments if isinstance(s, dict)]
        confidence = 1.0 - (sum(probs) / len(probs)) if probs else 0.5
        return STTResult(
            text=str(result.get("text", "")).strip(),
            language=str(result.get("language", language)),
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            audio_seconds=round(samples.size / rate, 3),
            backend=self.name,
        )


def _is_retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


class WhisperAPISTT:
    """Hosted Whisper (``whisper-1``) over HTTPS with timeout and retries."""

    name = "whisper_api"

    def __init__(
        self,
        api_key: str,
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create the client.

        Args:
            api_key: OpenAI API key.
            timeout_s: Per-request timeout in seconds.
            max_attempts: Total attempts for retryable failures.
            client: Optional shared ``httpx.AsyncClient``.
        """
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.client = client or httpx.AsyncClient(timeout=timeout_s)
        self._post = retry(
            retry=retry_if_exception(_is_retryable_http),
            stop=stop_after_attempt(max(1, max_attempts)),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )(self._post_once)

    async def _post_once(self, wav_bytes: bytes, language: str) -> dict[str, Any]:
        response = await self.client.post(
            WHISPER_API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data={"model": "whisper-1", "language": language, "response_format": "verbose_json"},
            files={"file": ("turn.wav", wav_bytes, "audio/wav")},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    async def transcribe(self, audio_b64: str | None, text: str | None, language: str) -> STTResult:
        """Transcribe via the hosted endpoint.

        Raises:
            STTError: When the API keeps failing or returns a non-2xx status.
        """
        if text:
            return STTResult(text=text, language=language, confidence=1.0, backend=self.name)
        if not audio_b64:
            return STTResult(text="", language=language, confidence=0.0, backend=self.name)
        seconds = _audio_seconds(audio_b64)
        wav_bytes = base64.b64decode(audio_b64)
        try:
            payload = await self._post(wav_bytes, language)
        except httpx.HTTPStatusError as exc:
            raise STTError(f"Whisper API returned {exc.response.status_code}") from exc
        except httpx.TransportError as exc:
            raise STTError(f"Whisper API unreachable: {exc}") from exc
        return STTResult(
            text=str(payload.get("text", "")).strip(),
            language=str(payload.get("language", language)),
            confidence=0.9,
            audio_seconds=seconds,
            backend=self.name,
        )


def build_stt(settings: Settings) -> SpeechToText:
    """Construct the STT backend named by ``WHISPER_BACKEND``.

    Args:
        settings: Active settings.

    Returns:
        The configured backend.

    Raises:
        ConfigurationError: If ``local`` is requested without ``openai-whisper``
            installed, or ``api`` without ``OPENAI_API_KEY``. Failing loudly here
            replaces the previous silent fallback to the stub.
    """
    if settings.whisper_backend == "local":
        try:
            stt = LocalWhisperSTT(settings.whisper_model, settings.stt_timeout_s)
        except ImportError as exc:
            raise ConfigurationError(
                'WHISPER_BACKEND=local requires `pip install "voice-agent[stt]"`'
            ) from exc
        logger.info("stt backend=whisper_local model={}", settings.whisper_model)
        return stt
    if settings.whisper_backend == "api":
        if not settings.openai_api_key:
            raise ConfigurationError("WHISPER_BACKEND=api requires OPENAI_API_KEY")
        logger.info("stt backend=whisper_api")
        return WhisperAPISTT(settings.openai_api_key, settings.stt_timeout_s)
    logger.info("stt backend=stub")
    return StubSTT()
