"""HTTP API: happy paths, 422 validation, typed errors and rate limiting."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.agent.reasoner import ReasonerError
from src.api.main import app, limiter


@pytest.fixture
def client() -> Iterator[TestClient]:
    limiter.reset()
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model"] == "claude-sonnet-4-5-20250929"
    assert data["whisper_backend"] == "stub" and data["llm_enabled"] == "no"


def test_turn_text_returns_trace_header(client: TestClient) -> None:
    r = client.post("/api/turn", json={"session_id": "api-1", "text": "Table for 2 tonight"})
    assert r.status_code == 200
    body = r.json()
    assert body["turn"] == 1 and body["action"] == "ask_clarification"
    assert r.headers["X-Trace-Id"] == body["trace_id"]
    assert {s["name"] for s in body["slots"]} == {"party_size", "date"}
    assert body["latency"]["total_ms"] >= 0


def test_turn_audio_silence(client: TestClient, silence_wav_b64: str) -> None:
    r = client.post("/api/turn", json={"session_id": "api-2", "audio_b64": silence_wav_b64})
    assert r.status_code == 200 and r.json()["speech_detected"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "api-3"},
        {"session_id": "api-3", "text": ""},
        {"session_id": "api-3", "text": "x" * 3000},
        {"session_id": "api-3", "audio_b64": "@@not-base64@@"},
        {"session_id": "bad id!", "text": "hi"},
        {"text": "hi"},
    ],
)
def test_turn_validation_422(client: TestClient, payload: dict[str, Any]) -> None:
    r = client.post("/api/turn", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_session_transcript_and_reset(client: TestClient) -> None:
    assert client.get("/api/sessions/api-4").status_code == 404
    client.post("/api/turn", json={"session_id": "api-4", "text": "hello"})
    r = client.get("/api/sessions/api-4")
    assert r.status_code == 200 and len(r.json()["turns"]) == 1
    reset = client.post("/api/sessions/api-4/reset")
    assert reset.status_code == 200 and reset.json() == {"reset": "api-4", "existed": True}


def test_metrics_endpoint(client: TestClient) -> None:
    client.post("/api/turn", json={"session_id": "api-5", "text": "hello"})
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["turns"] >= 1 and "p95" in data["latency_ms"]
    assert data["recent"][0]["session_id"] == "api-5"


def test_reasoner_failure_maps_to_502(client: TestClient) -> None:
    class _Broken:
        name = "claude"

        def respond(self, **kwargs: Any) -> Any:
            raise ReasonerError("model reply is not JSON")

    original = client.app.state.voice.pipeline.reasoner  # type: ignore[attr-defined]
    client.app.state.voice.pipeline.reasoner = _Broken()  # type: ignore[attr-defined]
    try:
        r = client.post("/api/turn", json={"session_id": "api-6", "text": "hi"})
    finally:
        client.app.state.voice.pipeline.reasoner = original  # type: ignore[attr-defined]
    assert r.status_code == 502
    assert r.json()["error"] == "reasoner_unavailable"


def test_rate_limit_returns_429(client: TestClient) -> None:
    statuses = [
        client.post("/api/turn", json={"session_id": "api-7", "text": "hi"}).status_code
        for _ in range(6)
    ]
    assert statuses[:5] == [200] * 5 and statuses[5] == 429


def test_cors_is_restricted(client: TestClient) -> None:
    allowed = client.options(
        "/api/turn",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"
    denied = client.options(
        "/api/turn",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in denied.headers
