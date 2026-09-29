"""Word error rate with the standard normalisation used for LibriSpeech.

Normalisation: lower-case, strip punctuation, collapse whitespace. Numbers are
left as written because LibriSpeech references spell them out; callers that
compare Whisper output against digit references should pre-normalise.
"""

from __future__ import annotations

from pathlib import Path

import jiwer
from pydantic import BaseModel

NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


class WERReport(BaseModel):
    """Word-level error metrics over a set of utterances."""

    n_utterances: int
    n_reference_words: int
    wer: float
    mer: float
    wil: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int


def normalize(text: str) -> str:
    """Apply the benchmark normalisation to one string."""
    words = NORMALIZE(text)
    return " ".join(words[0]) if words else ""


def compute_wer(references: list[str], hypotheses: list[str]) -> WERReport:
    """Compute WER/MER/WIL between aligned reference and hypothesis lists.

    Args:
        references: Ground-truth transcripts.
        hypotheses: Model transcripts, same order and length.

    Returns:
        Aggregate word-level metrics.

    Raises:
        ValueError: If the lists differ in length or are empty.
    """
    if not references or len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be non-empty and aligned")
    out = jiwer.process_words(
        references,
        hypotheses,
        reference_transform=NORMALIZE,
        hypothesis_transform=NORMALIZE,
    )
    n_ref_words = out.hits + out.substitutions + out.deletions
    return WERReport(
        n_utterances=len(references),
        n_reference_words=n_ref_words,
        wer=round(float(out.wer), 4),
        mer=round(float(out.mer), 4),
        wil=round(float(out.wil), 4),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
        hits=int(out.hits),
    )


def read_lines(path: Path) -> list[str]:
    """Read non-empty lines from a UTF-8 text file."""
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
