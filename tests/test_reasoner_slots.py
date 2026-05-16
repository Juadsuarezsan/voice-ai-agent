from src.agent.reasoner import StubReasoner
from src.api.schemas import Slot


def test_extract_party_size_numeric():
    r = StubReasoner()
    result = r.respond(transcript="Table for 4 please", slots=[])
    party = next((s for s in result.slots if s.name == "party_size"), None)
    assert party and party.value == 4


def test_asks_for_missing_slot():
    r = StubReasoner()
    result = r.respond(transcript="Table for 4 please", slots=[])
    assert result.action == "ask_clarification"
    assert result.finished is False


def test_fills_all_slots_and_books():
    r = StubReasoner()
    slots = [
        Slot(name="party_size", value=4, confirmed=True),
        Slot(name="date", value="tomorrow", confirmed=True),
        Slot(name="time", value="19:30", confirmed=True),
        Slot(name="name", value="John", confirmed=True),
    ]
    result = r.respond(transcript="confirm", slots=slots)
    assert result.action == "book"
    assert result.finished is True


def test_extracts_time_pm():
    r = StubReasoner()
    result = r.respond(transcript="7:30 pm", slots=[])
    time_slot = next((s for s in result.slots if s.name == "time"), None)
    assert time_slot and str(time_slot.value).startswith("19:30")


def test_extracts_date_keyword():
    r = StubReasoner()
    result = r.respond(transcript="tomorrow night", slots=[])
    date_slot = next((s for s in result.slots if s.name == "date"), None)
    assert date_slot and date_slot.value == "tomorrow"


def test_extracts_name():
    r = StubReasoner()
    result = r.respond(transcript="Under the name John Smith", slots=[])
    name_slot = next((s for s in result.slots if s.name == "name"), None)
    assert name_slot and name_slot.value == "John"
