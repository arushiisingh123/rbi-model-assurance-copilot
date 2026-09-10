"""RBI evidence records and source attribution (owner: Nidhi). Phase 3A, Task 6.

Turns a successful Task 5 retrieval result into immutable, traceable
``RBIEvidence`` records. Each record lets a reviewer answer:

  1. Which RBI document?              -> doc_id, title, source_url, document_type
  2. Which exact chunk?               -> chunk_id, chunk_index
  3. What exact text?                 -> text  (verbatim, never paraphrased)
  4. What reference / URL?            -> source_url, plus the full provenance dict
  5. Historical excerpt or current?   -> is_excerpt, is_current
  6. What provenance supports it?     -> provenance  (the stored chunk metadata)

WHAT THIS MODULE DOES NOT DO
    - It does not retrieve. It never runs a vector search, opens the index,
      or re-reads a source file -- it only reshapes a result that
      ``app/rag/retrieval.py`` already produced.
    - It does not interpret. No regulatory meaning, compliance conclusion,
      recommendation, or section/clause reference is created. Every value
      comes straight from the retrieval result.
    - It never invents a title, URL, date, reference, or requirement. A
      metadata field that was unavailable stays unavailable (``None`` /
      absent from provenance) -- never a placeholder that looks like a
      real citation.
    - Retrieving an excerpt does not make it "current regulation":
      ``is_excerpt`` / ``is_current`` are carried through unchanged.

This module does not import ``app/rag/smoke_test.py`` and does not change it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.rag.retrieval import EVIDENCE_RETRIEVED, NO_VERIFIED_EVIDENCE

# Re-exported so callers have one import for the whole evidence layer.
__all__ = [
    "RBIEvidence",
    "build_evidence",
    "is_no_evidence",
    "no_evidence_reason",
    "EvidenceError",
    "EvidenceConsistencyError",
    "NO_EVIDENCE_MESSAGE",
    "EVIDENCE_RETRIEVED",
    "NO_VERIFIED_EVIDENCE",
]

NO_EVIDENCE_MESSAGE = "No verified RBI evidence was retrieved from the indexed corpus."

# Attribution fields carried on every RBIEvidence. Where the provenance dict
# also carries one of these, the record's value must match it.
_ATTRIBUTION_KEYS = (
    "doc_id",
    "chunk_id",
    "chunk_index",
    "source_url",
    "title",
    "publication_date",
    "document_type",
    "is_excerpt",
    "is_current",
)

# Keys a Task 5 ``results`` entry must contain to be attributable.
_REQUIRED_HIT_KEYS = ("text", "doc_id", "chunk_id", "chunk_index", "provenance")


class EvidenceError(Exception):
    """The input is not a usable Task 5 retrieval result."""


class EvidenceConsistencyError(EvidenceError, ValueError):
    """A record's attribution field disagrees with its provenance metadata.

    Raised instead of silently 'correcting' the mismatch -- inconsistent
    attribution must be surfaced, not papered over.
    """


@dataclass(frozen=True)
class RBIEvidence:
    """One immutable, source-grounded piece of retrieved RBI evidence.

    Every field is copied from a Task 5 retrieval result. ``text`` is the
    exact retrieved chunk text -- not paraphrased, summarised, or annotated
    with interpretation. ``provenance`` is the stored chunk metadata
    (a copy; treat as read-only).

    A field that the source never provided is ``None`` here (and absent
    from ``provenance``), never a guessed value.
    """

    text: str
    doc_id: str
    chunk_id: str
    chunk_index: int
    source: str | None
    source_url: str | None
    title: str | None
    publication_date: str | None
    document_type: str | None
    is_excerpt: bool | None
    is_current: bool | None
    provenance: dict

    def __post_init__(self) -> None:
        # Own our copy of the provenance dict, then verify the record and
        # its provenance agree before the (frozen) instance is handed out.
        object.__setattr__(self, "provenance", dict(self.provenance))
        _check_consistency(self)

    def to_dict(self) -> dict:
        """A plain, JSON-serialisable dict with every attribution field.

        Enough for a downstream report layer to cite and trace this
        evidence; it contains nothing the record does not already hold.
        """
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "source_url": self.source_url,
            "title": self.title,
            "publication_date": self.publication_date,
            "document_type": self.document_type,
            "is_excerpt": self.is_excerpt,
            "is_current": self.is_current,
            "provenance": dict(self.provenance),
        }


def _check_consistency(evidence: RBIEvidence) -> None:
    prov = evidence.provenance
    if not isinstance(prov, dict):
        raise EvidenceConsistencyError(
            f"provenance must be a dict, got {type(prov).__name__}"
        )
    for key in _ATTRIBUTION_KEYS:
        if key in prov and getattr(evidence, key) != prov[key]:
            raise EvidenceConsistencyError(
                f"attribution field {key!r} = {getattr(evidence, key)!r} "
                f"disagrees with provenance[{key!r}] = {prov[key]!r}"
            )


def build_evidence(result: dict) -> list[RBIEvidence]:
    """Convert a Task 5 retrieval result into evidence records, in rank order.

    - ``evidence_status == EVIDENCE_RETRIEVED`` -> one ``RBIEvidence`` per
      entry in ``result["results"]``, in the same (ranked) order.
    - ``evidence_status == NO_VERIFIED_EVIDENCE`` -> ``[]``. No fabricated
      record, no placeholder citation.

    Does not retrieve anything -- it only reshapes ``result``.

    Raises
    ------
    EvidenceError
        If ``result`` is not a Task 5 retrieval result, or an
        ``EVIDENCE_RETRIEVED`` result carries no usable ``results`` entries.
    EvidenceConsistencyError
        If a ``results`` entry's attribution disagrees with its own
        provenance metadata.
    """
    if not isinstance(result, dict):
        raise EvidenceError(
            f"build_evidence() expects a Task 5 result dict, got {type(result).__name__}"
        )

    status = result.get("evidence_status")
    if status == NO_VERIFIED_EVIDENCE:
        return []
    if status != EVIDENCE_RETRIEVED:
        raise EvidenceError(
            f"unrecognised evidence_status {status!r}; expected "
            f"{EVIDENCE_RETRIEVED!r} or {NO_VERIFIED_EVIDENCE!r}"
        )

    hits = result.get("results")
    if not isinstance(hits, list) or not hits:
        raise EvidenceError(
            "an EVIDENCE_RETRIEVED result must carry a non-empty 'results' list"
        )
    return [_evidence_from_hit(hit) for hit in hits]


def _evidence_from_hit(hit: Any) -> RBIEvidence:
    if not isinstance(hit, dict):
        raise EvidenceError(
            f"each 'results' entry must be a dict, got {type(hit).__name__}"
        )
    missing = [key for key in _REQUIRED_HIT_KEYS if key not in hit]
    if missing:
        raise EvidenceError(f"'results' entry is missing required key(s): {missing}")

    return RBIEvidence(
        text=hit["text"],
        doc_id=hit["doc_id"],
        chunk_id=hit["chunk_id"],
        chunk_index=hit["chunk_index"],
        # Task 5 convention: the top-level `source` mirrors `title`; a
        # `results` entry only carries `title`, so `source` mirrors it here.
        source=hit.get("title"),
        source_url=hit.get("source_url"),
        title=hit.get("title"),
        publication_date=hit.get("publication_date"),
        document_type=hit.get("document_type"),
        is_excerpt=hit.get("is_excerpt"),
        is_current=hit.get("is_current"),
        provenance=hit["provenance"],
    )


def is_no_evidence(result: dict) -> bool:
    """True if ``result`` is a Task 5 ``NO_VERIFIED_EVIDENCE`` result."""
    return isinstance(result, dict) and result.get("evidence_status") == NO_VERIFIED_EVIDENCE


def no_evidence_reason(result: dict) -> str | None:
    """The reason retrieval gave for finding no verified evidence, or ``None``.

    Preserves the ``reason`` string Task 5 already recorded (for example
    "query is empty"); falls back to ``NO_EVIDENCE_MESSAGE`` if none was
    given. Invents nothing and never claims that no RBI rule exists -- only
    that nothing was retrieved from the indexed corpus.
    """
    if not is_no_evidence(result):
        return None
    return result.get("reason") or NO_EVIDENCE_MESSAGE
