"""Structured logging, trace identifiers and cost accounting.

Every request gets a ``trace_id`` that is bound to the loguru context so that all
log lines emitted while the turn is processed (one per graph node) can be joined
later. Cost is computed from token/character usage with the prices in
:class:`src.config.Settings` and never guessed.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.config import Settings

_CONFIGURED = False


def configure_logging(settings: Settings) -> None:
    """Configure loguru once with a structured, single-line format.

    Args:
        settings: Active settings; ``log_level`` selects the sink threshold and
            the LangSmith fields wire tracing without importing the client.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level:<7} | {name}:{function} | "
            "trace_id={extra[trace_id]} | {message}"
        ),
        serialize=False,
        backtrace=False,
        diagnose=False,
    )
    logger.configure(extra={"trace_id": "-"})
    wire_langsmith(settings)
    _CONFIGURED = True


def wire_langsmith(settings: Settings) -> bool:
    """Export LangSmith tracing variables for LangGraph if a key is configured.

    LangChain/LangGraph pick up ``LANGCHAIN_TRACING_V2``, ``LANGCHAIN_API_KEY``
    and ``LANGCHAIN_PROJECT`` from the environment; nothing is imported or
    called here, so this is safe to run without network access.

    Args:
        settings: Active settings.

    Returns:
        ``True`` when tracing was enabled, ``False`` otherwise.
    """
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    logger.info("LangSmith tracing enabled project={}", settings.langsmith_project)
    return True


def new_trace_id() -> str:
    """Return a new 32-hex-character trace identifier."""
    return uuid.uuid4().hex


@contextmanager
def trace_context(trace_id: str, **fields: Any) -> Iterator[None]:
    """Bind ``trace_id`` (and optional fields) to every log line in the block.

    Args:
        trace_id: Identifier for the current turn.
        **fields: Extra key/value pairs added to the log context.
    """
    with logger.contextualize(trace_id=trace_id, **fields):
        yield


def llm_cost_usd(
    input_tokens: int, output_tokens: int, *, price_in_per_mtok: float, price_out_per_mtok: float
) -> float:
    """Compute the USD cost of one LLM call.

    Args:
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        price_in_per_mtok: Input price in USD per million tokens.
        price_out_per_mtok: Output price in USD per million tokens.

    Returns:
        Cost in USD rounded to 8 decimals.
    """
    cost = (input_tokens * price_in_per_mtok + output_tokens * price_out_per_mtok) / 1_000_000
    return round(cost, 8)


def tts_cost_usd(characters: int, *, price_per_1k_chars: float) -> float:
    """Compute the USD cost of one TTS call billed per character.

    Args:
        characters: Characters synthesized.
        price_per_1k_chars: Price in USD per 1,000 characters.

    Returns:
        Cost in USD rounded to 8 decimals.
    """
    return round(characters * price_per_1k_chars / 1000, 8)


def stt_cost_usd(audio_seconds: float, *, price_per_min: float) -> float:
    """Compute the USD cost of one STT call billed per minute of audio.

    Args:
        audio_seconds: Audio duration in seconds.
        price_per_min: Price in USD per minute.

    Returns:
        Cost in USD rounded to 8 decimals.
    """
    return round(audio_seconds / 60 * price_per_min, 8)
