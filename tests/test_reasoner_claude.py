"""ClaudeReasoner against a mocked Anthropic client (no network, no key)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pytest_mock import MockerFixture

from src.agent.reasoner import ClaudeReasoner, ReasonerError, StubReasoner, build_reasoner
from src.api.schemas import Slot
from src.config import Settings

VALID_REPLY = {
    "response_text": "How many people will be dining?",
    "action": "ask_clarification",
    "slots": [{"name": "date", "value": "friday", "confirmed": True}],
    "finished": False,
    "awaiting_confirmation": False,
}


def _message(text: str, tokens_in: int = 120, tokens_out: int = 40) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
        stop_reason="end_turn",
    )


def _settings(retries: int = 2) -> Settings:
    return Settings(ANTHROPIC_API_KEY="test-key", LLM_MAX_RETRIES=retries, LLM_TIMEOUT_S=5.0)


def _client(mocker: MockerFixture, side_effect: Any) -> Any:
    client = mocker.MagicMock(spec=anthropic.Anthropic)
    client.messages = mocker.MagicMock()
    client.messages.create = mocker.MagicMock(side_effect=side_effect)
    return client


def _status_error(status: int, cls: type[anthropic.APIStatusError]) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


def test_valid_json_is_parsed_and_usage_tracked(mocker: MockerFixture) -> None:
    client = _client(mocker, [_message(json.dumps(VALID_REPLY))])
    reasoner = ClaudeReasoner(_settings(), client=client)
    out = reasoner.respond(transcript="Friday", slots=[], history=[])
    assert out.action == "ask_clarification"
    assert out.slots == [Slot(name="date", value="friday", confirmed=True)]
    assert (out.input_tokens, out.output_tokens) == (120, 40)
    assert out.backend == "claude"
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert kwargs["temperature"] == 0.0
    assert "<transcript>Friday</transcript>" in kwargs["messages"][-1]["content"]


def test_history_and_state_are_sent(mocker: MockerFixture) -> None:
    client = _client(mocker, [_message(json.dumps(VALID_REPLY))])
    reasoner = ClaudeReasoner(_settings(), client=client)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    reasoner.respond(
        transcript="Friday",
        slots=[Slot(name="party_size", value=2)],
        history=history,
        awaiting_confirmation=True,
    )
    messages = client.messages.create.call_args.kwargs["messages"]
    assert messages[0] == history[0] and messages[1] == history[1]
    assert '"awaiting_confirmation": true' in messages[-1]["content"]
    assert '"party_size"' in messages[-1]["content"]


def test_json_wrapped_in_prose_is_extracted(mocker: MockerFixture) -> None:
    text = "Sure! " + json.dumps(VALID_REPLY) + "\nDone."
    reasoner = ClaudeReasoner(_settings(), client=_client(mocker, [_message(text)]))
    assert reasoner.respond(transcript="x", slots=[], history=[]).action == "ask_clarification"


def test_non_json_reply_raises(mocker: MockerFixture) -> None:
    reasoner = ClaudeReasoner(_settings(), client=_client(mocker, [_message("I cannot help")]))
    with pytest.raises(ReasonerError, match="not JSON"):
        reasoner.respond(transcript="x", slots=[], history=[])


def test_malformed_json_raises(mocker: MockerFixture) -> None:
    reasoner = ClaudeReasoner(_settings(), client=_client(mocker, [_message('{"response_text": ')]))
    with pytest.raises(ReasonerError, match="not JSON|malformed"):
        reasoner.respond(transcript="x", slots=[], history=[])


def test_schema_violation_raises(mocker: MockerFixture) -> None:
    bad = dict(VALID_REPLY, action="dance")
    reasoner = ClaudeReasoner(_settings(), client=_client(mocker, [_message(json.dumps(bad))]))
    with pytest.raises(ReasonerError, match="schema"):
        reasoner.respond(transcript="x", slots=[], history=[])


def test_retries_rate_limit_then_succeeds(mocker: MockerFixture) -> None:
    client = _client(
        mocker,
        [_status_error(429, anthropic.RateLimitError), _message(json.dumps(VALID_REPLY))],
    )
    reasoner = ClaudeReasoner(_settings(retries=3), client=client)
    reasoner._call.retry.wait = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    out = reasoner.respond(transcript="x", slots=[], history=[])
    assert out.action == "ask_clarification"
    assert client.messages.create.call_count == 2


def test_exhausted_retries_raise_reasoner_error(mocker: MockerFixture) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = _client(mocker, anthropic.APIConnectionError(request=request))
    reasoner = ClaudeReasoner(_settings(retries=2), client=client)
    reasoner._call.retry.wait = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    with pytest.raises(ReasonerError, match="unavailable after retries"):
        reasoner.respond(transcript="x", slots=[], history=[])
    assert client.messages.create.call_count == 2


def test_bad_request_is_not_retried(mocker: MockerFixture) -> None:
    client = _client(mocker, _status_error(400, anthropic.BadRequestError))
    reasoner = ClaudeReasoner(_settings(retries=3), client=client)
    with pytest.raises(ReasonerError, match="400"):
        reasoner.respond(transcript="x", slots=[], history=[])
    assert client.messages.create.call_count == 1


def test_client_built_with_timeout_and_no_sdk_retries(mocker: MockerFixture) -> None:
    ctor = mocker.patch("anthropic.Anthropic")
    ClaudeReasoner(_settings())
    ctor.assert_called_once_with(api_key="test-key", timeout=5.0, max_retries=0)


def test_missing_key_raises() -> None:
    with pytest.raises(ReasonerError, match="ANTHROPIC_API_KEY"):
        ClaudeReasoner(Settings(ANTHROPIC_API_KEY=None))


def test_build_reasoner_selects_backend(mocker: MockerFixture) -> None:
    mocker.patch("anthropic.Anthropic")
    assert isinstance(build_reasoner(Settings(ANTHROPIC_API_KEY=None)), StubReasoner)
    assert isinstance(build_reasoner(Settings(ANTHROPIC_API_KEY="k")), ClaudeReasoner)
