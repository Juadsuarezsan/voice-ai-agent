"""PII redaction applied before transcripts are persisted.

The agent needs the caller's name to complete the booking, so redaction is
applied only at the storage boundary: the live turn keeps real values, the
logged record keeps a stable pseudonym (``NAME_<hash>``) plus masked phone
numbers, e-mails and card-like digit runs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{6,}\d)(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def pseudonym(value: str, salt: str = "voice-agent") -> str:
    """Return a stable, non-reversible pseudonym for ``value``.

    Args:
        value: Personal value (a name).
        salt: Namespace salt; change it to rotate pseudonyms.

    Returns:
        ``NAME_`` followed by 8 hex characters.
    """
    digest = hashlib.sha256(f"{salt}:{value.strip().lower()}".encode()).hexdigest()
    return f"NAME_{digest[:8]}"


def redact_text(text: str, names: list[str] | None = None) -> str:
    """Mask e-mails, phone numbers, card numbers and the given names.

    Args:
        text: Free text.
        names: Names to replace by their pseudonym (longest first).

    Returns:
        Redacted text.
    """
    out = CARD_RE.sub("[CARD]", text)
    out = EMAIL_RE.sub("[EMAIL]", out)
    out = PHONE_RE.sub(
        lambda m: "[PHONE]" if sum(c.isdigit() for c in m.group(0)) >= 7 else m.group(0), out
    )
    for name in sorted({n for n in (names or []) if n and len(n) >= 2}, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(name)}\b", pseudonym(name), out, flags=re.IGNORECASE)
    return out


def redact_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of ``slots`` with the ``name`` slot pseudonymised."""
    redacted: list[dict[str, Any]] = []
    for slot in slots:
        item = dict(slot)
        if item.get("name") == "name" and isinstance(item.get("value"), str):
            item["value"] = pseudonym(str(item["value"]))
        redacted.append(item)
    return redacted


def names_from_slots(slots: list[dict[str, Any]]) -> list[str]:
    """Collect the string values of ``name`` slots."""
    return [
        str(s["value"])
        for s in slots
        if s.get("name") == "name" and isinstance(s.get("value"), str)
    ]
