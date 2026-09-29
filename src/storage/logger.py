"""Conversation logger: persists every turn with its decision and latencies.

Three backends implement :class:`ConversationLogger`:

* :class:`InMemoryConversationLogger` -- default; tests and the demo.
* :class:`SQLiteConversationLogger` -- single-file persistence for local runs.
* :class:`PostgresConversationLogger` -- production, backed by a ``psycopg``
  connection pool. The pool is injectable so it can be exercised with a fake.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel, Field

from src.config import Settings

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    ts TEXT NOT NULL,
    transcript TEXT NOT NULL,
    response_text TEXT NOT NULL,
    action TEXT NOT NULL,
    finished INTEGER NOT NULL,
    slots_json TEXT NOT NULL,
    latency_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    backends_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn);
"""

DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS turns (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    transcript TEXT NOT NULL,
    response_text TEXT NOT NULL,
    action TEXT NOT NULL,
    finished BOOLEAN NOT NULL,
    slots JSONB NOT NULL,
    latency JSONB NOT NULL,
    usage JSONB NOT NULL,
    backends JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn);
"""

INSERT_POSTGRES = """
INSERT INTO turns (session_id, trace_id, turn, ts, transcript, response_text, action, finished,
                   slots, latency, usage, backends)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
"""


class TurnRecord(BaseModel):
    """One persisted conversational turn.

    Attributes:
        session_id: Conversation identifier.
        trace_id: Request trace identifier.
        turn: 1-based turn index.
        ts: UTC timestamp of completion.
        transcript: What the user said (PII-redacted when enabled).
        response_text: What the agent said.
        action: Reasoner decision.
        finished: Whether the session ended on this turn.
        slots: Slot state after the turn.
        latency: Per-node latency in ms.
        usage: Tokens, characters and cost.
        backends: Backend name per stage.
    """

    session_id: str
    trace_id: str
    turn: int
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    transcript: str
    response_text: str
    action: str
    finished: bool
    slots: list[dict[str, Any]] = Field(default_factory=list)
    latency: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, float] = Field(default_factory=dict)
    backends: dict[str, str] = Field(default_factory=dict)


class ConversationLogger(Protocol):
    """Persistence contract for turns."""

    name: str

    def log_turn(self, record: TurnRecord) -> None:
        """Persist one turn."""
        ...

    def get_session(self, session_id: str) -> list[TurnRecord]:
        """Return the turns of a session ordered by turn index."""
        ...

    def recent(self, limit: int = 100) -> list[TurnRecord]:
        """Return the most recent turns across sessions, newest first."""
        ...

    def count(self) -> int:
        """Return the total number of persisted turns."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class InMemoryConversationLogger:
    """Bounded in-process store (last ``maxlen`` turns)."""

    name = "memory"

    def __init__(self, maxlen: int = 10_000) -> None:
        """Create the store with a bounded deque."""
        self._turns: deque[TurnRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def log_turn(self, record: TurnRecord) -> None:
        """Append the record."""
        with self._lock:
            self._turns.append(record)

    def get_session(self, session_id: str) -> list[TurnRecord]:
        """Return the turns of one session."""
        with self._lock:
            return sorted(
                (t for t in self._turns if t.session_id == session_id), key=lambda t: t.turn
            )

    def recent(self, limit: int = 100) -> list[TurnRecord]:
        """Return the newest ``limit`` turns."""
        with self._lock:
            return list(self._turns)[-limit:][::-1]

    def count(self) -> int:
        """Return the number of stored turns."""
        return len(self._turns)

    def close(self) -> None:
        """No resources to release."""


class SQLiteConversationLogger:
    """SQLite-backed store; a ``:memory:`` path works for tests."""

    name = "sqlite"

    def __init__(self, path: str) -> None:
        """Open (and create) the database.

        Args:
            path: File path or ``":memory:"``.
        """
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(DDL_SQLITE)

    def log_turn(self, record: TurnRecord) -> None:
        """Insert the record."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, trace_id, turn, ts, transcript, response_text, action, "
                "finished, slots_json, latency_json, usage_json, backends_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.session_id,
                    record.trace_id,
                    record.turn,
                    record.ts.isoformat(),
                    record.transcript,
                    record.response_text,
                    record.action,
                    int(record.finished),
                    json.dumps(record.slots),
                    json.dumps(record.latency),
                    json.dumps(record.usage),
                    json.dumps(record.backends),
                ),
            )
            self._conn.commit()

    def _rows_to_records(self, rows: list[tuple[Any, ...]]) -> list[TurnRecord]:
        return [
            TurnRecord(
                session_id=r[0],
                trace_id=r[1],
                turn=r[2],
                ts=datetime.fromisoformat(r[3]),
                transcript=r[4],
                response_text=r[5],
                action=r[6],
                finished=bool(r[7]),
                slots=json.loads(r[8]),
                latency=json.loads(r[9]),
                usage=json.loads(r[10]),
                backends=json.loads(r[11]),
            )
            for r in rows
        ]

    _COLS = (
        "session_id, trace_id, turn, ts, transcript, response_text, action, finished, "
        "slots_json, latency_json, usage_json, backends_json"
    )

    def get_session(self, session_id: str) -> list[TurnRecord]:
        """Return the turns of one session."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._COLS} FROM turns WHERE session_id = ? ORDER BY turn", (session_id,)
            ).fetchall()
        return self._rows_to_records(rows)

    def recent(self, limit: int = 100) -> list[TurnRecord]:
        """Return the newest ``limit`` turns."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._COLS} FROM turns ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return self._rows_to_records(rows)

    def count(self) -> int:
        """Return the number of stored turns."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()


class PostgresConversationLogger:
    """PostgreSQL store using a ``psycopg_pool.ConnectionPool``.

    The pool is created lazily from ``dsn`` unless one is injected, which keeps
    the module importable and testable without a database.
    """

    name = "postgres"

    def __init__(
        self, dsn: str, min_size: int = 1, max_size: int = 8, pool: Any | None = None
    ) -> None:
        """Create the store and ensure the schema exists.

        Args:
            dsn: PostgreSQL connection string.
            min_size: Minimum pooled connections.
            max_size: Maximum pooled connections.
            pool: Pre-built pool (tests inject a fake exposing ``connection()``).
        """
        if pool is None:
            from psycopg_pool import ConnectionPool  # noqa: PLC0415 -- optional at import

            pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True)
        self._pool = pool
        with self._pool.connection() as conn:
            conn.execute(DDL_POSTGRES)

    def log_turn(self, record: TurnRecord) -> None:
        """Insert the record."""
        with self._pool.connection() as conn:
            conn.execute(
                INSERT_POSTGRES,
                (
                    record.session_id,
                    record.trace_id,
                    record.turn,
                    record.ts,
                    record.transcript,
                    record.response_text,
                    record.action,
                    record.finished,
                    json.dumps(record.slots),
                    json.dumps(record.latency),
                    json.dumps(record.usage),
                    json.dumps(record.backends),
                ),
            )

    @staticmethod
    def _to_record(row: tuple[Any, ...]) -> TurnRecord:
        def _obj(v: Any) -> Any:
            return json.loads(v) if isinstance(v, str) else v

        return TurnRecord(
            session_id=row[0],
            trace_id=row[1],
            turn=row[2],
            ts=row[3],
            transcript=row[4],
            response_text=row[5],
            action=row[6],
            finished=bool(row[7]),
            slots=_obj(row[8]),
            latency=_obj(row[9]),
            usage=_obj(row[10]),
            backends=_obj(row[11]),
        )

    _COLS = (
        "session_id, trace_id, turn, ts, transcript, response_text, action, finished, "
        "slots, latency, usage, backends"
    )

    def get_session(self, session_id: str) -> list[TurnRecord]:
        """Return the turns of one session."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {self._COLS} FROM turns WHERE session_id = %s ORDER BY turn", (session_id,)
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def recent(self, limit: int = 100) -> list[TurnRecord]:
        """Return the newest ``limit`` turns."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {self._COLS} FROM turns ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def count(self) -> int:
        """Return the number of stored turns."""
        with self._pool.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM turns").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the pool."""
        self._pool.close()


def build_logger(settings: Settings) -> ConversationLogger:
    """Construct the logger named by ``LOGGER_BACKEND``.

    Args:
        settings: Active settings.

    Returns:
        The configured logger.

    Raises:
        ValueError: If ``postgres`` is requested without ``DATABASE_URL``.
    """
    if settings.logger_backend == "sqlite":
        logger.info("conversation logger backend=sqlite path={}", settings.sqlite_path)
        return SQLiteConversationLogger(settings.sqlite_path)
    if settings.logger_backend == "postgres":
        if not settings.database_url:
            raise ValueError("LOGGER_BACKEND=postgres requires DATABASE_URL")
        logger.info(
            "conversation logger backend=postgres pool={}..{}",
            settings.db_pool_min,
            settings.db_pool_max,
        )
        return PostgresConversationLogger(
            settings.database_url, settings.db_pool_min, settings.db_pool_max
        )
    logger.info("conversation logger backend=memory")
    return InMemoryConversationLogger()
