"""TTS — ElevenLabs in prod, silent-WAV stub for tests."""
from __future__ import annotations

import base64
import os
import struct
from dataclasses import dataclass


@dataclass
class TTSResult:
    audio_b64: str | None
    duration_ms: int = 0


class StubTTS:
    """Generates a deterministic silent 200ms WAV — keeps the contract."""

    name = "stub"

    async def synthesize(self, text: str, language: str = "en") -> TTSResult:
        sample_rate = 16000
        n_samples = int(sample_rate * 0.2)
        # 16-bit PCM header
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + n_samples * 2, b"WAVE", b"fmt ", 16, 1, 1,
            sample_rate, sample_rate * 2, 2, 16, b"data", n_samples * 2,
        )
        body = b"\x00\x00" * n_samples
        return TTSResult(audio_b64=base64.b64encode(header + body).decode(), duration_ms=200)


class ElevenLabsTTS:
    """ElevenLabs streaming TTS. Requires ELEVENLABS_API_KEY."""

    name = "elevenlabs"

    def __init__(self, api_key: str, voice_id: str) -> None:
        from elevenlabs.client import ElevenLabs
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id

    async def synthesize(self, text: str, language: str = "en") -> TTSResult:
        # eleven_multilingual_v2 covers EN/ES/DE/FR etc.
        audio_bytes = b"".join(self.client.text_to_speech.convert(
            voice_id=self.voice_id, text=text, model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        ))
        return TTSResult(audio_b64=base64.b64encode(audio_bytes).decode(),
                          duration_ms=int(len(text) * 65))


def build_tts() -> StubTTS | ElevenLabsTTS:
    backend = os.environ.get("TTS_BACKEND", "elevenlabs")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if backend == "elevenlabs" and api_key:
        try:
            return ElevenLabsTTS(api_key=api_key, voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"))
        except Exception:
            pass
    return StubTTS()
