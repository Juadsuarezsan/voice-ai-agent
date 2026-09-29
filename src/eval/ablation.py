"""Ablations that run offline.

VAD on/off: with the energy detector on, silent turns are answered with a
re-prompt and never reach STT or the reasoner; with it off, every silent
buffer goes through STT (empty transcript) and the reasoner asks a slot
question as if the caller had spoken. Measured on synthetic audio.

Whisper ``tiny`` vs ``base`` cannot run here (the ``[stt]`` extra is not
installed); the runner reports it as pending.
"""

from __future__ import annotations

import statistics

import numpy as np
from pydantic import BaseModel

from src.agent.loop import VoiceLoop
from src.agent.vad import FloatArray, VADResult, encode_wav_b64
from src.api.schemas import TurnRequest
from src.config import Settings

SAMPLE_RATE = 16_000


class AlwaysSpeechVAD:
    """Ablation detector: treats every buffer as speech (VAD disabled)."""

    name = "off"

    def detect(self, pcm: FloatArray, sample_rate: int) -> VADResult:
        """Return speech detected regardless of content."""
        return VADResult(True, 1.0, True, round(pcm.size / sample_rate, 3), self.name)


class VADAblationResult(BaseModel):
    """Metrics for one VAD configuration."""

    vad: str
    n_silence: int
    n_speech: int
    silence_rejected: int
    speech_passed: int
    silence_rejection_rate: float
    speech_pass_rate: float
    stt_calls_saved: int
    vad_overhead_ms_mean: float


def _synthetic_turns(n: int, seed: int) -> list[tuple[str, bool]]:
    rng = np.random.default_rng(seed)
    turns: list[tuple[str, bool]] = []
    for i in range(n):
        speech = i % 2 == 0
        if speech:
            t = np.arange(int(SAMPLE_RATE * 0.9), dtype=np.float32) / SAMPLE_RATE
            freq = float(rng.uniform(150, 400))
            pcm = 0.4 * np.sin(2 * np.pi * freq * t)
            pcm = np.concatenate([pcm, np.zeros(int(SAMPLE_RATE * 0.6), dtype=np.float32)])
        else:
            pcm = rng.normal(0.0, 0.0008, int(SAMPLE_RATE * 1.2))  # room noise floor
        turns.append((encode_wav_b64(pcm.astype(np.float32), SAMPLE_RATE), speech))
    return turns


async def run_vad_ablation(
    settings: Settings, n_turns: int = 20, seed: int = 20260516
) -> list[VADAblationResult]:
    """Compare VAD on (energy) vs off on synthetic speech/silence turns."""
    turns = _synthetic_turns(n_turns, seed)
    results: list[VADAblationResult] = []
    for label, vad in (("energy", None), ("off", AlwaysSpeechVAD())):
        loop = VoiceLoop(settings, vad=vad) if vad else VoiceLoop(settings)
        silence_rejected = speech_passed = 0
        overhead: list[int] = []
        for i, (audio, is_speech) in enumerate(turns):
            out = await loop.turn(TurnRequest(session_id=f"vad-{label}-{i}", audio_b64=audio))
            overhead.append(out.latency.vad_ms)
            if is_speech and out.speech_detected:
                speech_passed += 1
            if not is_speech and not out.speech_detected:
                silence_rejected += 1
        n_silence = sum(1 for _, s in turns if not s)
        n_speech = len(turns) - n_silence
        results.append(
            VADAblationResult(
                vad=label,
                n_silence=n_silence,
                n_speech=n_speech,
                silence_rejected=silence_rejected,
                speech_passed=speech_passed,
                silence_rejection_rate=round(silence_rejected / max(1, n_silence), 4),
                speech_pass_rate=round(speech_passed / max(1, n_speech), 4),
                stt_calls_saved=silence_rejected,
                vad_overhead_ms_mean=round(statistics.mean(overhead), 2) if overhead else 0.0,
            )
        )
    return results
