"""Deterministic slot-filling reasoner."""

from __future__ import annotations

import pytest

from src.agent.reasoner import REQUIRED_SLOTS, StubReasoner
from src.api.schemas import Slot


def _slot(result_slots: list[Slot], name: str) -> Slot | None:
    return next((s for s in result_slots if s.name == name), None)


def test_extract_party_size_numeric() -> None:
    result = StubReasoner().respond(transcript="Table for 4 please", slots=[], history=[])
    party = _slot(result.slots, "party_size")
    assert party is not None and party.value == 4


def test_party_size_ignores_time_digits() -> None:
    result = StubReasoner().respond(transcript="At 7 pm", slots=[], history=[])
    assert _slot(result.slots, "party_size") is None
    time_slot = _slot(result.slots, "time")
    assert time_slot is not None and time_slot.value == "19:00"


def test_asks_for_missing_slot_in_order() -> None:
    result = StubReasoner().respond(transcript="Table for 4 please", slots=[], history=[])
    assert result.action == "ask_clarification"
    assert result.finished is False
    assert "date" in result.response_text.lower()


def test_greeting_on_first_turn_only() -> None:
    r = StubReasoner()
    first = r.respond(transcript="Hello", slots=[], history=[])
    assert first.response_text.startswith("Welcome")
    second = r.respond(transcript="Hmm", slots=[], history=[{"role": "user", "content": "Hello"}])
    assert not second.response_text.startswith("Welcome")


def test_reads_back_then_books_on_confirmation() -> None:
    r = StubReasoner()
    slots = [
        Slot(name="party_size", value=4, confirmed=True),
        Slot(name="date", value="tomorrow", confirmed=True),
        Slot(name="time", value="19:30", confirmed=True),
    ]
    readback = r.respond(transcript="Under the name John", slots=slots, history=[])
    assert readback.action == "ask_clarification"
    assert readback.awaiting_confirmation is True
    assert "confirm" in readback.response_text.lower()
    booked = r.respond(
        transcript="yes", slots=readback.slots, history=[], awaiting_confirmation=True
    )
    assert booked.action == "book"
    assert booked.finished is True
    assert all(s.confirmed for s in booked.slots)


def test_correction_during_confirmation_updates_slot() -> None:
    r = StubReasoner()
    slots = [
        Slot(name=n, value=v, confirmed=True)
        for n, v in zip(REQUIRED_SLOTS, [2, "friday", "20:00", "Ana"])
    ]
    out = r.respond(
        transcript="No, make it 6 people", slots=slots, history=[], awaiting_confirmation=True
    )
    assert out.action == "ask_clarification"
    assert out.awaiting_confirmation is True
    party = _slot(out.slots, "party_size")
    assert party is not None and party.value == 6


def test_legacy_confirm_with_all_slots_books() -> None:
    slots = [
        Slot(name=n, value=v, confirmed=True)
        for n, v in zip(REQUIRED_SLOTS, [4, "tomorrow", "19:30", "John"])
    ]
    result = StubReasoner().respond(
        transcript="confirm", slots=slots, history=[], awaiting_confirmation=True
    )
    assert result.action == "book"
    assert result.finished is True


def test_transfer_to_human_on_request() -> None:
    result = StubReasoner().respond(
        transcript="Can I speak to a human please?", slots=[], history=[]
    )
    assert result.action == "transfer_to_human"
    assert result.finished is True


def test_out_of_scope_transfers_when_no_slots() -> None:
    result = StubReasoner().respond(
        transcript="I want to cancel my delivery order", slots=[], history=[]
    )
    assert result.action == "transfer_to_human"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7:30 pm", "19:30"),
        ("at 8 pm", "20:00"),
        ("12 pm", "12:00"),
        ("noon works", "12:00"),
        ("19:15 please", "19:15"),
        ("7 o'clock in the evening", "19:00"),
    ],
)
def test_extracts_time(text: str, expected: str) -> None:
    result = StubReasoner().respond(transcript=text, slots=[], history=[])
    time_slot = _slot(result.slots, "time")
    assert time_slot is not None and time_slot.value == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tomorrow night", "tomorrow"),
        ("on Friday", "friday"),
        ("2026-10-03", "2026-10-03"),
        ("October 3rd", "october 3"),
        ("the 3rd of October", "october 3"),
    ],
)
def test_extracts_date(text: str, expected: str) -> None:
    result = StubReasoner().respond(transcript=text, slots=[], history=[])
    date_slot = _slot(result.slots, "date")
    assert date_slot is not None and date_slot.value == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Under the name John Smith", "John Smith"),
        ("My name is Maria", "Maria"),
        ("It's Carlos Lopez", "Carlos Lopez"),
        ("Under Alice", "Alice"),
    ],
)
def test_extracts_name(text: str, expected: str) -> None:
    result = StubReasoner().respond(transcript=text, slots=[], history=[])
    name_slot = _slot(result.slots, "name")
    assert name_slot is not None and name_slot.value == expected


def test_name_does_not_capture_numbers_or_stopwords() -> None:
    result = StubReasoner().respond(transcript="Table for Two tonight", slots=[], history=[])
    assert _slot(result.slots, "name") is None


def test_one_shot_utterance_fills_all_slots() -> None:
    result = StubReasoner().respond(
        transcript="Booking for 3 people tonight at 6:00 pm under Carlos.", slots=[], history=[]
    )
    assert {s.name for s in result.slots} == set(REQUIRED_SLOTS)
    assert result.awaiting_confirmation is True
