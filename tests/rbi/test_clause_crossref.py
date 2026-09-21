"""The register <-> source clause cross-reference must stay true to the PDFs.

Every row in ``app/rbi/clause_crossref.py`` is re-derived here FROM THE
DOCUMENTS: the register's verbatim quote is located in the extracted text, and
the page and clause the extractor reports at that position are compared with
the recorded row. Nothing is matched on clause number, because matching on the
number would assume the very thing being checked.

If a row drifts -- because a PDF was replaced, or the extractor changed -- this
fails, rather than a wrong clause reference reaching a citation.
"""
import re

import pytest

from app.rag.corpus import corpus_from_manifest
from app.rag.pdf_extract import extract_pdf
from app.rbi.clause_crossref import (
    CLAUSE_CROSS_REFERENCES,
    crossref_for,
    crossrefs_for_document,
    divergent,
)
from app.rbi.verified_requirements import VERIFIED_REQUIREMENTS

# verified_requirements and the manifest name the same two documents
# differently; neither module is changed to accommodate the other.
MANIFEST_ID = {
    "RBI-IT-GOVERNANCE-2023": "RBI-IT-GOV-2023",
    "RBI-IT-OUTSOURCING-2023": "RBI-IT-OUTSOURCE-2023",
}


def _squash(text):
    return re.sub(r"\s+", " ", text).strip().lower()


@pytest.fixture(scope="module")
def extracted():
    """Extracted text + an offset map, per manifest document id."""
    out = {}
    for source in corpus_from_manifest():
        document = extract_pdf(source.resolved_path())
        flat, offsets = _flatten(document.text)
        out[source.doc_id] = (document, flat, offsets)
    return out


def _flatten(text):
    """Whitespace-collapsed text, plus index -> original-offset mapping.

    Needed because the PDFs wrap lines mid-sentence, so a quote read from the
    website never matches the PDF's raw text byte for byte.
    """
    flat_chars = []
    offsets = []
    previous_space = True
    for index, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                flat_chars.append(" ")
                offsets.append(index)
                previous_space = True
        else:
            flat_chars.append(char.lower())
            offsets.append(index)
            previous_space = False
    return "".join(flat_chars), offsets


def _locate(flat, offsets, quote):
    words = _squash(quote).split()
    for size in (12, 9, 7, 5):
        for start in (0, 2, 4):
            if len(words) < start + size:
                continue
            probe = " ".join(words[start : start + size])
            found = flat.find(probe)
            if found >= 0:
                return offsets[found]
    return None


@pytest.mark.parametrize(
    "requirement",
    VERIFIED_REQUIREMENTS,
    ids=[r.requirement.requirement_id for r in VERIFIED_REQUIREMENTS],
)
def test_each_cross_reference_matches_the_document(extracted, requirement):
    xref = crossref_for(requirement.requirement.requirement_id)
    assert xref is not None, "every verified requirement needs a cross-reference"

    doc_id = MANIFEST_ID.get(xref.document_id, xref.document_id)
    document, flat, offsets = extracted[doc_id]

    offset = _locate(flat, offsets, requirement.quote)
    assert offset is not None, (
        f"{xref.register_clause}: the register's quote is not in the PDF -- "
        "either the wrong document is stored, or the quote is wrong"
    )

    assert document.page_for_offset(offset) == xref.page
    mark = document.section_for_offset(offset)
    assert (mark.section_id if mark else None) == xref.source_clause


def test_numbering_matches_flag_is_accurate():
    for row in CLAUSE_CROSS_REFERENCES:
        expected = row.register_clause == row.source_clause
        assert row.numbering_matches is expected, row.requirement_id


def test_divergent_rows_are_the_ones_that_differ():
    assert set(divergent()) == {
        row for row in CLAUSE_CROSS_REFERENCES if not row.numbering_matches
    }
    assert divergent(), "the divergence is real and must stay visible"


def test_lookup_by_document_returns_only_that_document():
    rows = crossrefs_for_document("RBI-IT-OUTSOURCING-2023")

    assert rows
    assert {row.document_id for row in rows} == {"RBI-IT-OUTSOURCING-2023"}


def test_an_unknown_requirement_has_no_cross_reference():
    """None means "no correspondence established", not "they are the same"."""
    assert crossref_for("RBI-NOT-A-REQUIREMENT") is None


def test_the_cross_reference_never_rewrites_a_source_clause():
    """The PDF's own numbering is recorded, never the register's.

    16.13 is what the website prints. The document prints 16(m). If this
    table ever stored "16.13" as the SOURCE clause, a citation would name a
    clause reference that does not appear in the document it cites.
    """
    audit = crossref_for("RBI-ITO-2023-16.13")

    assert audit.register_clause == "16.13"
    assert audit.source_clause == "16(m)"
    assert audit.source_clause != audit.register_clause
