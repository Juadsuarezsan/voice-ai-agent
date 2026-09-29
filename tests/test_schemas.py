"""Edge cases rejected at the schema boundary (HTTP 422 through FastAPI)."""

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from src.api.schemas import TurnRequest


def test_empty_input_rejected() -> None:
    with pytest.raises(ValidationError, match="either text or audio_b64"):
        TurnRequest(session_id="s1")


def test_blank_text_is_empty_input() -> None:
    with pytest.raises(ValidationError):
        TurnRequest(session_id="s1", text="   ")


def test_oversized_text_rejected() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        TurnRequest(session_id="s1", text="a" * 5000)


def test_malformed_base64_rejected() -> None:
    with pytest.raises(ValidationError, match="not valid base64"):
        TurnRequest(session_id="s1", audio_b64="!!!not-base64$$$")


def test_non_wav_payload_rejected() -> None:
    payload = base64.b64encode(b"ID3" + b"\x00" * 100).decode()
    with pytest.raises(ValidationError, match="RIFF/WAVE"):
        TurnRequest(session_id="s1", audio_b64=payload)


def test_oversized_audio_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_AUDIO_BYTES", "2048")
    from src.config import get_settings

    get_settings.cache_clear()
    payload = base64.b64encode(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 5000).decode()
    with pytest.raises(ValidationError, match="character limit"):
        TurnRequest(session_id="s1", audio_b64=payload)


@pytest.mark.parametrize("bad", ["", "a b", "x" * 65, "s/1", "id;drop"])
def test_invalid_session_id_rejected(bad: str) -> None:
    with pytest.raises(ValidationError, match="session_id"):
        TurnRequest(session_id=bad, text="hello")


def test_valid_wav_accepted(silence_wav_b64: str) -> None:
    req = TurnRequest(session_id="ok-1", audio_b64=silence_wav_b64)
    assert req.audio_b64 == silence_wav_b64
    assert req.text is None


def test_text_is_stripped() -> None:
    assert TurnRequest(session_id="s1", text="  hello  ").text == "hello"
