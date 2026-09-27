"""Register clause <-> source-document clause cross-reference (owner: Nidhi).

WHY THIS FILE EXISTS
    ``app/rbi/verified_requirements.py`` cites clauses as they are numbered on
    the RBI WEBSITE, because that is where they were read from. The stored
    PDFs number some of the same provisions differently. Both numberings are
    correct for their own source, and neither may be rewritten into the other.

    So the correspondence is recorded HERE, separately, as a cross-reference.
    The extractor keeps reporting exactly what the PDF prints; the register
    keeps citing exactly what the website printed; this table says which is
    which. Nothing downstream has to guess, and no citation is falsified.

THE DIVERGENCE IS NOT UNIFORM -- WHICH IS THE POINT
    Of the 16 verified requirements, only 2 have identical numbering in both
    sources and 14 differ. The 2023 IT Outsourcing Directions letter their
    sub-items a), b), c) ... while the website renders the same sub-items
    16.1, 16.2, 16.3 ... "16.13" and "16(m)" are the same provision (m is the
    13th letter), but only one of those strings appears in the PDF, and it is
    not the one the register cites.

HOW THESE ROWS WERE ESTABLISHED
    Each row was produced by locating the register's own VERBATIM QUOTE in
    the extracted PDF text and reading back the page and clause mark that the
    extractor reported at that offset. Nothing was matched on clause number,
    because matching on the number would assume the answer. The locating run
    is reproducible: see tests/rbi/test_clause_crossref.py, which re-derives
    every row from the PDFs and fails if any drifts.

WHAT THIS IS NOT
    - Not an RBI-published concordance. RBI does not publish one. This records
      an observed correspondence between two renderings of one instrument.
    - Not a compliance input. It changes no status, and nothing here feeds the
      rule engine. A requirement's status is decided in
      ``app/rbi/verified_requirements.py`` exactly as before.
    - Not authoritative over either source. Where the two disagree, the source
      document governs what may be quoted, and the register governs what was
      verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClauseCrossReference:
    """One verified requirement, located in its own source PDF.

    Attributes
    ----------
    requirement_id
        The id in ``app/rbi/verified_requirements.py``.
    document_id
        The manifest document the requirement was verified against.
    register_clause
        The clause reference as the register cites it (website numbering).
    source_clause
        The clause mark the extractor reports at the quote's position in the
        PDF (the document's own printed numbering). ``None`` would mean the
        text was found but under no detected clause -- currently never the
        case, and represented so a future document can say so honestly.
    page
        1-based page of the PDF where the quote begins.
    numbering_matches
        True when both sources number the provision identically. Recorded
        rather than derived at read time so the split is visible in the data.
    """

    requirement_id: str
    document_id: str
    register_clause: str
    source_clause: Optional[str]
    page: int
    numbering_matches: bool


# Verified 2026-09-21 against the stored PDFs by quote location, not by
# clause number. Ordered by document, then by page.
CLAUSE_CROSS_REFERENCES: tuple[ClauseCrossReference, ...] = (
    # --- Digital Lending 2025: numbering agrees on the clause, and the
    # register's roman sub-item is not separately marked in the PDF.
    ClauseCrossReference(
        "RBI-DL-2025-5.ii", "RBI-DIGITAL-LENDING-2025", "5.ii", "5", 5, False
    ),
    ClauseCrossReference(
        "RBI-DL-2025-5.iii", "RBI-DIGITAL-LENDING-2025", "5.iii", "5", 6, False
    ),
    ClauseCrossReference(
        "RBI-DL-2025-7.i", "RBI-DIGITAL-LENDING-2025", "7.i", "7", 7, False
    ),
    ClauseCrossReference(
        "RBI-DL-2025-8.i", "RBI-DIGITAL-LENDING-2025", "8.i", "8", 8, False
    ),
    ClauseCrossReference(
        "RBI-DL-2025-11.i", "RBI-DIGITAL-LENDING-2025", "11.i", "11", 10, False
    ),
    ClauseCrossReference(
        "RBI-DL-2025-12.i", "RBI-DIGITAL-LENDING-2025", "12.i", "12", 11, False
    ),
    # --- IT Outsourcing 2023: the website's N.M is the PDF's N(letter).
    ClauseCrossReference(
        "RBI-ITO-2023-4.1", "RBI-IT-OUTSOURCING-2023", "4.1", "4(a)", 9, False
    ),
    ClauseCrossReference(
        "RBI-ITO-2023-13.1", "RBI-IT-OUTSOURCING-2023", "13.1", "13(a)", 13, False
    ),
    ClauseCrossReference(
        "RBI-ITO-2023-16.13", "RBI-IT-OUTSOURCING-2023", "16.13", "16(m)", 16, False
    ),
    ClauseCrossReference(
        "RBI-ITO-2023-18.1", "RBI-IT-OUTSOURCING-2023", "18.1", "18(a)", 19, False
    ),
    ClauseCrossReference(
        "RBI-ITO-2023-19.6", "RBI-IT-OUTSOURCING-2023", "19.6", "19(f)", 21, False
    ),
    ClauseCrossReference(
        "RBI-ITO-2023-22.1", "RBI-IT-OUTSOURCING-2023", "22.1", "22(a)", 23, False
    ),
    # --- IT Governance 2023: both sources agree.
    ClauseCrossReference(
        "RBI-ITGOV-2023-4.b", "RBI-IT-GOVERNANCE-2023", "4(b)", "4(b)", 7, True
    ),
    ClauseCrossReference(
        "RBI-ITGOV-2023-23", "RBI-IT-GOVERNANCE-2023", "23", "23", 16, True
    ),
    # --- Fraud Risk Management 2024: the register prefixes the chapter, the
    # PDF prints the clause number alone. Same provision.
    ClauseCrossReference(
        "RBI-FRAUD-2024-2.3", "RBI-FRAUD-NBFC-2024", "Chapter II, 2.3", "2.3", 5, False
    ),
    ClauseCrossReference(
        "RBI-FRAUD-2024-3.1.3",
        "RBI-FRAUD-NBFC-2024",
        "Chapter III, 3.1.3",
        "3.1.3",
        6,
        False,
    ),
)

# The register and the manifest chose different ids for the same two
# documents. Recorded here because this module is already the bridge between
# the two, so neither side has to learn about the other.
#
#   register id (verified_requirements.py)  ->  manifest id (corpus_manifest.json)
REGISTER_TO_MANIFEST_DOCUMENT_ID = {
    "RBI-IT-GOVERNANCE-2023": "RBI-IT-GOV-2023",
    "RBI-IT-OUTSOURCING-2023": "RBI-IT-OUTSOURCE-2023",
    "RBI-DIGITAL-LENDING-2025": "RBI-DIGITAL-LENDING-2025",
    "RBI-FRAUD-NBFC-2024": "RBI-FRAUD-NBFC-2024",
}


def manifest_document_id(register_document_id: str) -> str:
    """The manifest's id for a register document id (unchanged if the same)."""
    return REGISTER_TO_MANIFEST_DOCUMENT_ID.get(
        register_document_id, register_document_id
    )


_BY_REQUIREMENT = {row.requirement_id: row for row in CLAUSE_CROSS_REFERENCES}

# Keyed the way a RETRIEVED chunk knows itself: manifest doc id + the clause
# the PDF prints. Used to answer "is this retrieved clause one the register
# cites, and under what number?"
_BY_SOURCE = {
    (manifest_document_id(row.document_id), row.source_clause): row
    for row in CLAUSE_CROSS_REFERENCES
    if row.source_clause
}


def register_clause_for_source(
    manifest_doc_id: Optional[str], source_clause: Optional[str]
) -> Optional[str]:
    """The register's clause reference for a clause printed in a PDF.

    ``None`` when this clause is not one the verified register cites -- which
    is the common case, and an honest one. Most of an RBI Direction has no
    entry in a curated sixteen-requirement register, and saying so is better
    than implying the register covers more than it does.

    Never derives a register number from a source number. The pairing exists
    only where it was established by locating the register's own quote in the
    document; see this module's docstring.
    """
    if not manifest_doc_id or not source_clause:
        return None
    row = _BY_SOURCE.get((manifest_doc_id, source_clause))
    return row.register_clause if row else None


def crossref_for(requirement_id: str) -> Optional[ClauseCrossReference]:
    """The cross-reference for one verified requirement, or ``None``.

    ``None`` means no correspondence has been established -- which is the
    honest answer for a requirement whose source document is not held.
    """
    return _BY_REQUIREMENT.get(requirement_id)


def crossrefs_for_document(document_id: str) -> tuple[ClauseCrossReference, ...]:
    """Every established cross-reference for one manifest document."""
    return tuple(
        row for row in CLAUSE_CROSS_REFERENCES if row.document_id == document_id
    )


def divergent() -> tuple[ClauseCrossReference, ...]:
    """Rows where the two sources number the same provision differently."""
    return tuple(row for row in CLAUSE_CROSS_REFERENCES if not row.numbering_matches)


__all__ = [
    "ClauseCrossReference",
    "CLAUSE_CROSS_REFERENCES",
    "REGISTER_TO_MANIFEST_DOCUMENT_ID",
    "manifest_document_id",
    "register_clause_for_source",
    "crossref_for",
    "crossrefs_for_document",
    "divergent",
]
