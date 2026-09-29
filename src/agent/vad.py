"""Voice activity detection for end-of-turn detection.

Two detectors implement :class:`VoiceActivityDetector`:

* :class:`EnergyVAD` -- numpy RMS energy over 30 ms frames, dependency free.
  Good enough for the browser demo (the client already segments turns) and
  for tests.
* :class:`SileroVAD` -- wraps the ``silero-vad`` package (torch). Imported
  lazily so the base install never loads torch; tests inject a fake module.
"""

from __future__ import annotations

import base64
import importlib
import io
import wave
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from loguru import logger

from src.config import Settings

FloatArray = npt.NDArray[np.float32]


class AudioDecodeError(ValueError):
    """Raised when the audio payload cannot be decoded as PCM WAV."""


@dataclass(frozen=True)
class VADResult:
    """Outcome of voice activity detection.

    Attributes:
        speech_detected: Whether any speech segment was found.
        speech_ratio: Fraction of frames classified as speech (0-1).
        end_of_turn: Whether the audio ends with enough silence to close a turn.
        duration_s: Total audio duration in seconds.
        backend: ``"energy"`` or ``"silero"``.
    """

    speech_detected: bool
    speech_ratio: float
    end_of_turn: bool
    duration_s: float
    backend: str


class VoiceActivityDetector(Protocol):
    """Anything that can classify speech vs. silence in a PCM buffer."""

    name: str

    def detect(self, pcm: FloatArray, sample_rate: int) -> VADResult:
        """Classify ``pcm`` (float32 mono in [-1, 1])."""
        ...


def decode_wav_b64(audio_b64: str) -> tuple[FloatArray, int]:
    """Decode base64 WAV into float32 mono PCM.

    Args:
        audio_b64: Base64-encoded RIFF/WAVE bytes (8/16/32-bit PCM).

    Returns:
        Tuple of ``(samples, sample_rate)`` with samples in ``[-1, 1]``.

    Raises:
        AudioDecodeError: If the payload is not a readable PCM WAV file.
    """
    try:
        raw = base64.b64decode(audio_b64, validate=True)
        with wave.open(io.BytesIO(raw)) as wf:
            n_channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (ValueError, wave.Error, EOFError) as exc:
        raise AudioDecodeError(f"cannot decode WAV: {exc}") from exc
    if width == 1:
        samples: FloatArray = (
            np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioDecodeError(f"unsupported sample width {width}")
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.float32)
    return samples, rate


def encode_wav_b64(samples: FloatArray, sample_rate: int) -> str:
    """Encode float32 mono PCM as a base64 16-bit WAV (inverse of :func:`decode_wav_b64`)."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


class EnergyVAD:
    """Frame-energy detector with a fixed RMS threshold."""

    name = "energy"

    def __init__(
        self, threshold: float = 0.5, min_silence_ms: int = 500, frame_ms: int = 30
    ) -> None:
        """Create the detector.

        Args:
            threshold: Sensitivity in ``[0, 1]``; mapped to an RMS floor of
                ``0.002 + 0.05 * threshold``.
            min_silence_ms: Trailing silence required to declare end of turn.
            frame_ms: Analysis frame length in milliseconds.
        """
        self.rms_floor = 0.002 + 0.05 * threshold
        self.min_silence_ms = min_silence_ms
        self.frame_ms = frame_ms

    def detect(self, pcm: FloatArray, sample_rate: int) -> VADResult:
        """Classify frames by RMS energy.

        Args:
            pcm: Float32 mono samples.
            sample_rate: Samples per second.

        Returns:
            Detection result; empty input yields ``speech_detected=False``.
        """
        if pcm.size == 0 or sample_rate <= 0:
            return VADResult(False, 0.0, True, 0.0, self.name)
        frame = max(1, int(sample_rate * self.frame_ms / 1000))
        n_frames = int(np.ceil(pcm.size / frame))
        padded = np.zeros(n_frames * frame, dtype=np.float32)
        padded[: pcm.size] = pcm
        rms = np.sqrt(np.mean(padded.reshape(n_frames, frame) ** 2, axis=1))
        speech = rms > self.rms_floor
        ratio = float(speech.mean()) if n_frames else 0.0
        trailing = 0
        for flag in speech[::-1]:
            if flag:
                break
            trailing += 1
        end_of_turn = trailing * self.frame_ms >= self.min_silence_ms or not speech.any()
        return VADResult(
            speech_detected=bool(speech.any()),
            speech_ratio=round(ratio, 4),
            end_of_turn=bool(end_of_turn),
            duration_s=round(pcm.size / sample_rate, 3),
            backend=self.name,
        )


class SileroVAD:
    """Silero VAD (ONNX/torch) via the ``silero-vad`` package, loaded lazily."""

    name = "silero"

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        module: Any | None = None,
    ) -> None:
        """Create the detector.

        Args:
            threshold: Speech probability threshold passed to Silero.
            min_silence_ms: Silence gap that closes a speech segment.
            module: Pre-imported ``silero_vad`` module (tests inject a fake).

        Raises:
            ImportError: If ``silero-vad`` is not installed; install with
                ``pip install "voice-agent[vad]"``.
        """
        mod: Any = module if module is not None else importlib.import_module("silero_vad")
        self._mod = mod
        self.model = mod.load_silero_vad()
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms

    def detect(self, pcm: FloatArray, sample_rate: int) -> VADResult:
        """Run Silero over the buffer and derive turn-end from the last segment."""
        if pcm.size == 0:
            return VADResult(False, 0.0, True, 0.0, self.name)
        torch: Any = importlib.import_module("torch")
        tensor = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32))
        segments: list[dict[str, int]] = self._mod.get_speech_timestamps(
            tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_ms,
        )
        duration = pcm.size / sample_rate
        speech_samples = sum(int(seg["end"]) - int(seg["start"]) for seg in segments)
        last_end = max((int(seg["end"]) for seg in segments), default=0)
        trailing_ms = (pcm.size - last_end) / sample_rate * 1000
        return VADResult(
            speech_detected=bool(segments),
            speech_ratio=round(speech_samples / max(1, pcm.size), 4),
            end_of_turn=bool(not segments or trailing_ms >= self.min_silence_ms),
            duration_s=round(duration, 3),
            backend=self.name,
        )


def build_vad(settings: Settings) -> VoiceActivityDetector:
    """Select the VAD backend from settings.

    Args:
        settings: Active settings.

    Returns:
        :class:`SileroVAD` when ``VAD_BACKEND=silero`` and the package is
        installed; otherwise :class:`EnergyVAD`. A missing optional package is
        logged as a warning and degrades to the energy detector, because VAD
        is advisory in the HTTP flow (the client already segments turns).
    """
    if settings.vad_backend == "silero":
        try:
            vad = SileroVAD(settings.vad_threshold, settings.vad_min_silence_ms)
        except ImportError as exc:
            logger.warning("silero-vad not installed ({}); falling back to energy VAD", exc)
        else:
            logger.info("vad backend=silero")
            return vad
    logger.info("vad backend=energy")
    return EnergyVAD(settings.vad_threshold, settings.vad_min_silence_ms)
