from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    whisper_model: str = Field(default="large-v3", alias="WHISPER_MODEL")
    whisper_backend: str = Field(default="local", alias="WHISPER_BACKEND")
    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", alias="ELEVENLABS_VOICE_ID")
    tts_backend: str = Field(default="elevenlabs", alias="TTS_BACKEND")
    turn_latency_target_ms: int = Field(default=800, alias="TURN_LATENCY_TARGET_MS")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
