"""Environment-driven configuration.

Every knob the system exposes lives here and is read from the environment (or a
``.env`` file) through :class:`Settings`. Nothing else in ``src/`` reads
``os.environ`` directly, so the full configuration surface is this one file plus
``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Pinned production model. Dated ID so evaluation results stay reproducible.
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

WhisperBackend = Literal["stub", "local", "api"]
TTSBackend = Literal["stub", "elevenlabs", "xtts"]
VADBackend = Literal["energy", "silero"]
LoggerBackend = Literal["memory", "sqlite", "postgres"]


class Settings(BaseSettings):
    """Runtime settings for the voice agent.

    Attributes are populated from environment variables whose names match the
    ``alias`` of each field. Unknown variables are ignored so the same ``.env``
    can be shared with docker-compose.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    # --- LLM -----------------------------------------------------------------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default=DEFAULT_MODEL, alias="ANTHROPIC_MODEL")
    llm_timeout_s: float = Field(default=20.0, alias="LLM_TIMEOUT_S", gt=0)
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES", ge=0, le=10)
    llm_max_tokens: int = Field(default=400, alias="LLM_MAX_TOKENS", ge=64, le=4096)
    price_input_per_mtok: float = Field(default=3.0, alias="PRICE_INPUT_PER_MTOK", ge=0)
    price_output_per_mtok: float = Field(default=15.0, alias="PRICE_OUTPUT_PER_MTOK", ge=0)

    # --- Speech to text --------------------------------------------------------
    whisper_backend: WhisperBackend = Field(default="stub", alias="WHISPER_BACKEND")
    whisper_model: str = Field(default="base", alias="WHISPER_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    whisper_api_price_per_min: float = Field(default=0.006, alias="WHISPER_API_PRICE_PER_MIN", ge=0)
    stt_timeout_s: float = Field(default=30.0, alias="STT_TIMEOUT_S", gt=0)

    # --- Text to speech --------------------------------------------------------
    tts_backend: TTSBackend = Field(default="stub", alias="TTS_BACKEND")
    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", alias="ELEVENLABS_VOICE_ID")
    elevenlabs_model_id: str = Field(default="eleven_multilingual_v2", alias="ELEVENLABS_MODEL_ID")
    elevenlabs_price_per_1k_chars: float = Field(
        default=0.30, alias="ELEVENLABS_PRICE_PER_1K_CHARS", ge=0
    )
    tts_timeout_s: float = Field(default=15.0, alias="TTS_TIMEOUT_S", gt=0)

    # --- Voice activity detection ---------------------------------------------
    vad_backend: VADBackend = Field(default="energy", alias="VAD_BACKEND")
    vad_threshold: float = Field(default=0.5, alias="VAD_THRESHOLD", ge=0, le=1)
    vad_min_silence_ms: int = Field(default=500, alias="VAD_MIN_SILENCE_MS", ge=0)

    # --- Persistence -----------------------------------------------------------
    logger_backend: LoggerBackend = Field(default="memory", alias="LOGGER_BACKEND")
    sqlite_path: str = Field(default="data/conversations.db", alias="SQLITE_PATH")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    db_pool_min: int = Field(default=1, alias="DB_POOL_MIN", ge=0)
    db_pool_max: int = Field(default=8, alias="DB_POOL_MAX", ge=1)
    redact_pii: bool = Field(default=True, alias="REDACT_PII")

    # --- HTTP API --------------------------------------------------------------
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000",
        alias="CORS_ORIGINS",
    )
    rate_limit: str = Field(default="30/minute", alias="RATE_LIMIT")
    max_audio_bytes: int = Field(default=5_000_000, alias="MAX_AUDIO_BYTES", ge=1024)
    max_text_chars: int = Field(default=2000, alias="MAX_TEXT_CHARS", ge=1)
    turn_latency_target_ms: int = Field(default=800, alias="TURN_LATENCY_TARGET_MS", gt=0)

    # --- Observability ---------------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="voice-ai-agent", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")

    @property
    def cors_origin_list(self) -> list[str]:
        """Return the configured CORS origins as a cleaned list.

        Returns:
            Origins split on commas with whitespace stripped; empty entries are
            dropped so a trailing comma does not become a wildcard.
        """
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_audio_b64_chars(self) -> int:
        """Return the maximum accepted length of the ``audio_b64`` field.

        Base64 inflates payloads by 4/3; the extra 4 characters absorb padding.
        """
        return (self.max_audio_bytes * 4) // 3 + 4


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    Tests that mutate environment variables must call
    ``get_settings.cache_clear()`` afterwards.
    """
    return Settings()
