"""Cost helpers, trace ids and LangSmith wiring."""

from __future__ import annotations

import pytest
from loguru import logger

from src.config import Settings
from src.observability import (
    llm_cost_usd,
    new_trace_id,
    stt_cost_usd,
    trace_context,
    tts_cost_usd,
    wire_langsmith,
)


def test_llm_cost_matches_pinned_prices() -> None:
    # 1M input at $3 + 1M output at $15
    assert (
        llm_cost_usd(1_000_000, 1_000_000, price_in_per_mtok=3.0, price_out_per_mtok=15.0) == 18.0
    )
    assert llm_cost_usd(0, 0, price_in_per_mtok=3.0, price_out_per_mtok=15.0) == 0.0


def test_tts_and_stt_costs() -> None:
    assert tts_cost_usd(1000, price_per_1k_chars=0.30) == pytest.approx(0.30)
    assert stt_cost_usd(60.0, price_per_min=0.006) == pytest.approx(0.006)


def test_trace_ids_are_unique_hex() -> None:
    a, b = new_trace_id(), new_trace_id()
    assert a != b and len(a) == 32 and int(a, 16) >= 0


def test_trace_context_binds_extra() -> None:
    captured: list[str] = []
    sink_id = logger.add(
        lambda m: captured.append(m.record["extra"].get("trace_id", "")), level="INFO"
    )
    try:
        with trace_context("abc123"):
            logger.info("inside")
    finally:
        logger.remove(sink_id)
    assert captured == ["abc123"]


def test_wire_langsmith_requires_key_and_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    assert wire_langsmith(Settings(LANGSMITH_TRACING=True, LANGSMITH_API_KEY=None)) is False
    assert wire_langsmith(Settings(LANGSMITH_TRACING=False, LANGSMITH_API_KEY="ls")) is False
    assert (
        wire_langsmith(
            Settings(LANGSMITH_TRACING=True, LANGSMITH_API_KEY="ls", LANGSMITH_PROJECT="p")
        )
        is True
    )
    import os

    assert os.environ["LANGCHAIN_TRACING_V2"] == "true" and os.environ["LANGCHAIN_PROJECT"] == "p"
