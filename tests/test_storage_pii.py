"""Conversation logger backends and PII redaction."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from src.config import Settings
from src.security.pii import names_from_slots, pseudonym, redact_slots, redact_text
from src.storage.logger import (
    InMemoryConversationLogger,
    PostgresConversationLogger,
    SQLiteConversationLogger,
    TurnRecord,
    build_logger,
)


def _record(session: str, turn: int, finished: bool = False) -> TurnRecord:
    return TurnRecord(
        session_id=session,
        trace_id=f"trace-{session}-{turn}",
        turn=turn,
        transcript=f"user {turn}",
        response_text=f"agent {turn}",
        action="book" if finished else "ask_clarification",
        finished=finished,
        slots=[{"name": "party_size", "value": 2, "confirmed": True}],
        latency={"stt": 1, "llm": 2, "tts": 3},
        usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.0001},
        backends={"reasoner": "stub"},
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_logger_roundtrip(backend: str) -> None:
    store = (
        InMemoryConversationLogger()
        if backend == "memory"
        else SQLiteConversationLogger(":memory:")
    )
    store.log_turn(_record("a", 1))
    store.log_turn(_record("b", 1))
    store.log_turn(_record("a", 2, finished=True))
    assert store.count() == 3
    session = store.get_session("a")
    assert [t.turn for t in session] == [1, 2]
    assert session[1].finished is True
    assert session[1].slots[0]["value"] == 2
    assert session[1].latency == {"stt": 1, "llm": 2, "tts": 3}
    recent = store.recent(2)
    assert [(t.session_id, t.turn) for t in recent] == [("a", 2), ("b", 1)]
    assert store.get_session("missing") == []
    store.close()


def test_sqlite_persists_to_file(tmp_path: Any) -> None:
    path = str(tmp_path / "nested" / "conv.db")
    SQLiteConversationLogger(path).log_turn(_record("s", 1))
    reopened = SQLiteConversationLogger(path)
    assert reopened.count() == 1
    reopened.close()


class _FakeConn:
    def __init__(self, store: list[tuple[Any, ...]]) -> None:
        self.store = store
        self.executed: list[str] = []

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> _FakeConn:
        self.executed.append(sql)
        if sql.startswith("\nINSERT") and params:
            self.store.append(params)
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.store)

    def fetchone(self) -> tuple[int]:
        return (len(self.store),)


class _FakePool:
    def __init__(self) -> None:
        self.rows: list[tuple[Any, ...]] = []
        self.closed = False

    @contextmanager
    def connection(self) -> Any:
        yield _FakeConn(self.rows)

    def close(self) -> None:
        self.closed = True


def test_postgres_logger_with_fake_pool() -> None:
    pool = _FakePool()
    store = PostgresConversationLogger("postgresql://x", pool=pool)
    store.log_turn(_record("pg", 1))
    assert store.count() == 1
    row = pool.rows[0]
    assert row[0] == "pg" and isinstance(row[3], datetime) and row[3].tzinfo is UTC
    fetched = store.get_session("pg")
    assert fetched[0].usage["input_tokens"] == 10
    assert store.recent(1)[0].session_id == "pg"
    store.close()
    assert pool.closed is True


def test_build_logger_variants(tmp_path: Any) -> None:
    assert isinstance(build_logger(Settings(LOGGER_BACKEND="memory")), InMemoryConversationLogger)
    sqlite = build_logger(Settings(LOGGER_BACKEND="sqlite", SQLITE_PATH=str(tmp_path / "c.db")))
    assert isinstance(sqlite, SQLiteConversationLogger)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        build_logger(Settings(LOGGER_BACKEND="postgres", DATABASE_URL=None))


def test_redact_text_masks_contact_data() -> None:
    text = "Call me at +57 300 123 4567 or maria@example.com, card 4111 1111 1111 1111"
    out = redact_text(text)
    assert "[PHONE]" in out and "[EMAIL]" in out and "[CARD]" in out
    assert "4567" not in out and "example.com" not in out


def test_redact_text_keeps_short_numbers() -> None:
    assert redact_text("table for 4 at 19:30") == "table for 4 at 19:30"


def test_redact_names_and_slots() -> None:
    slots = [{"name": "name", "value": "John Smith"}, {"name": "party_size", "value": 4}]
    names = names_from_slots(slots)
    assert names == ["John Smith"]
    out = redact_text("Booking for john smith confirmed", names)
    assert "john smith" not in out.lower() and pseudonym("John Smith") in out
    redacted = redact_slots(slots)
    assert redacted[0]["value"] == pseudonym("John Smith") and redacted[1]["value"] == 4
    assert slots[0]["value"] == "John Smith"  # original untouched


def test_pseudonym_is_stable_and_case_insensitive() -> None:
    assert pseudonym("Ana") == pseudonym("ana ") and pseudonym("Ana") != pseudonym("Bob")
