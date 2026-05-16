"""Whisper STT wrapper. Local model in prod; stub passthrough for tests."""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from dataclasses import dataclass


@dataclass
class STTResult:
    text: str
    language: str
    confidence: float


class StubSTT:
    """No-op STT — when the caller already passes text, this just echoes it."""
    name = "stub"

    async def transcribe(self, audio_b64: str | None, text: str | None, language: str) -> STTResult:
        if text:
            return STTResult(text=text, language=language, confidence=1.0)
        return STTResult(text="", language=language, confidence=0.0)


class LocalWhisperSTT:
    """openai-whisper local model. Lazy-imports torch + whisper; heavy."""
    name = "whisper_local"

    def __init__(self, model_name: str = "base") -> None:
        import whisper
        self.model = whisper.load_model(model_name)

    async def transcribe(self, audio_b64: str | None, text: str | None, language: str) -> STTResult:
        if text:
            return STTResult(text=text, language=language, confidence=1.0)
        if not audio_b64:
            return STTResult(text="", language=language, confidence=0.0)

        def _run() -> tuple[str, str, float]:
            audio = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                path = f.name
            try:
                result = self.model.transcribe(path, language=language)
                return str(result.get("text", "")), str(result.get("language", language)), 0.9
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        text_out, lang_out, conf = await asyncio.get_event_loop().run_in_executor(None, _run)
        return STTResult(text=text_out, language=lang_out, confidence=conf)


def build_stt() -> StubSTT | LocalWhisperSTT:
    backend = os.environ.get("WHISPER_BACKEND", "local")
    if backend != "local":
        return StubSTT()
    try:
        model = os.environ.get("WHISPER_MODEL", "base")
        return LocalWhisperSTT(model)
    except Exception:
        return StubSTT()
