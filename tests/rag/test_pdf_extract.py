"""Phase A tests: PDF extraction with page and clause provenance.

These run against the REAL stored RBI PDFs rather than a synthetic fixture.
A synthetic PDF would prove the regex works; it would not prove the thing
that actually matters, which is that a clause a reviewer can name is found at
a page a reviewer can open in the document this project ships.

The clause numbers asserted here are the ones
``app/rbi/verified_requirements.py`` cites, so these tests also pin the
correspondence between the two regulatory lanes.
"""
import pytest

from app.rag.corpus import corpus_from_manifest
from app.rag.pdf_extract import (
    ExtractedDocument,
    PDFExtractionError,
    SectionMark,
    extract_pdf,
)

IT_OUTSOURCING = "RBI-IT-OUTSOURCE-2023"
IT_GOVERNANCE = "RBI-IT-GOV-2023"
FS_OUTSOURCING_NBFC = "RBI-FS-OUTSOURCE-NBFC-2017"


@pytest.fixture(scope="module")
def sources():
    """Resolved paths for the downloaded manifest documents, by doc_id."""
    corpus = corpus_from_manifest()
    return {source.doc_id: source.resolved_path() for source in corpus}


@pytest.fixture(scope="module")
def outsourcing(sources) -> ExtractedDocument:
    return extract_pdf(sources[IT_OUTSOURCING])


@pytest.fixture(scope="module")
def governance(sources) -> ExtractedDocument:
    return extract_pdf(sources[IT_GOVERNANCE])


@pytest.fixture(scope="module")
def nbfc_outsourcing(sources) -> ExtractedDocument:
    return extract_pdf(sources[FS_OUTSOURCING_NBFC])


def _locate(document: ExtractedDocument, needle: str) -> int:
    offset = document.text.lower().find(needle.lower())
    assert offset >= 0, f"text not found in document: {needle!r}"
    return offset


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def test_every_phase_a_pdf_extracts_with_pages(outsourcing, governance, nbfc_outsourcing):
    assert outsourcing.page_count == 31
    assert governance.page_count == 26
    assert nbfc_outsourcing.page_count == 13


def test_page_numbers_are_contiguous_and_one_based(outsourcing):
    numbers = [page.page_number for page in outsourcing.pages]
    assert numbers == list(range(1, outsourcing.page_count + 1))


def test_page_spans_tile_the_text_without_gaps_or_overlap(outsourcing):
    """Every character belongs to exactly one page, or the mapping lies."""
    previous_end = 0
    for page_number, start, end in outsourcing.page_spans:
        assert start >= previous_end
        assert end > start, f"page {page_number} is empty"
        previous_end = end


def test_page_for_offset_resolves_a_real_offset(outsourcing):
    offset = _locate(outsourcing, "right to conduct audit")
    assert outsourcing.page_for_offset(offset) == 16


def test_page_for_offset_returns_none_outside_the_document(outsourcing):
    """Out of range yields None, never a plausible page.

    Clamping to page 1 would turn a caller bug into a citation that looks
    checkable and points at the wrong place.
    """
    assert outsourcing.page_for_offset(-1) is None
    assert outsourcing.page_for_offset(len(outsourcing.text) + 5_000) is None


def test_line_structure_is_preserved(outsourcing):
    """Clause numbers sit at line starts; collapsing newlines destroys them."""
    assert "\n" in outsourcing.text
    assert any("\n" in page.text for page in outsourcing.pages)


# ---------------------------------------------------------------------------
# Clause detection -- the six IT Outsourcing clauses the register cites
# ---------------------------------------------------------------------------

# (register clause, a distinctive phrase from its quote, expected PDF page,
#  expected PDF section id).
#
# NOTE the numbering divergence, which is real and deliberate: the RBI website
# renders clause 16's sub-items 16.1 ... 16.13, while this PDF letters them
# a) ... m). The extractor reports what the PDF prints. See
# app/rag/pdf_extract.py, "KNOWN DIVERGENCE".
#
# The phrases are chosen NOT to span a line break: pypdf preserves the PDF's
# own line wrapping, so "robust framework" is really "robust \nframework" in
# several places and a naive substring search misses it.
OUTSOURCING_CLAUSES = [
    ("4.1", "shall not diminish", 9, "4(a)"),
    ("13.1", "appropriate due diligence", 13, "13(a)"),
    ("16.13", "right to conduct audit", 16, "16(m)"),
    ("18.1", "develop and establish", 19, "18(a)"),
    ("19.6", "periodically review the financial", 21, "19(f)"),
    ("22.1", "clear exit strategy", 23, "22(a)"),
]


@pytest.mark.parametrize(
    "register_clause,phrase,expected_page,expected_section",
    OUTSOURCING_CLAUSES,
    ids=[row[0] for row in OUTSOURCING_CLAUSES],
)
def test_cited_outsourcing_clauses_are_locatable(
    outsourcing, register_clause, phrase, expected_page, expected_section
):
    """Each clause verified_requirements.py cites resolves to a page+section."""
    offset = _locate(outsourcing, phrase)

    assert outsourcing.page_for_offset(offset) == expected_page
    mark = outsourcing.section_for_offset(offset)
    assert mark is not None, f"{register_clause}: no clause mark in force"
    assert mark.section_id == expected_section


def test_governance_clause_23_is_detected_with_its_title(governance):
    """Clause 23 is numbered identically in the PDF and on the website.

    Asserted against the detected mark rather than by searching for the
    clause text, because the same words appear first in the contents
    listing -- and a citation that resolves to the contents page is exactly
    the failure the contents-page guard exists to prevent.
    """
    marks = [m for m in governance.sections if m.section_id == "23"]

    assert len(marks) == 1, "clause 23 should be detected once, in the body"
    mark = marks[0]
    assert mark.page_number == 16
    assert mark.section_title == "IT and Information Security Risk Management Framework"


def test_nbfc_outsourcing_uses_decimal_clause_numbering(nbfc_outsourcing):
    """A third numbering style, detected without special-casing the file."""
    ids = {mark.section_id for mark in nbfc_outsourcing.sections}
    assert {"1.1", "1.2", "1.3", "1.4", "1.5", "1.6"} <= ids

    offset = _locate(nbfc_outsourcing, "Activities that shall not be outsourced")
    mark = nbfc_outsourcing.section_for_offset(offset)
    assert mark is not None
    assert mark.section_id == "2"
    assert mark.section_title == "Activities that shall not be outsourced"


# ---------------------------------------------------------------------------
# Nothing is fabricated
# ---------------------------------------------------------------------------


def test_no_section_mark_points_outside_its_own_page(outsourcing):
    for mark in outsourcing.sections:
        assert outsourcing.page_for_offset(mark.offset) == mark.page_number


def test_section_marks_are_in_document_order(outsourcing):
    offsets = [mark.offset for mark in outsourcing.sections]
    assert offsets == sorted(offsets), (
        "section_for_offset() walks this list assuming order; unsorted marks "
        "would silently return the wrong clause"
    )


def test_front_matter_has_no_clause_in_force(governance):
    """The covering letter precedes clause 1, and is not attributed to it."""
    assert governance.section_for_offset(0) is None


def test_contents_pages_produce_no_clause_marks(governance):
    """A contents listing carries real clause numbers against page numbers.

    Treating one as a clause would anchor citations to the contents page --
    a page reference that resolves, and resolves to the wrong place.
    """
    contents_pages = {
        page.page_number
        for page in governance.pages
        if page.text.count("....") >= 3
    }
    assert contents_pages, "expected this document to have a contents page"

    marked_pages = {mark.page_number for mark in governance.sections}
    assert not (contents_pages & marked_pages)


def test_a_section_title_is_never_a_slice_of_requirement_text(outsourcing):
    """Titles are headings. A sentence fragment is not a heading."""
    for mark in outsourcing.sections:
        if mark.section_title is None:
            continue
        assert len(mark.section_title) <= 60
        assert "," not in mark.section_title
        assert not mark.section_title.endswith((".", ";", ":"))


def test_cross_references_do_not_become_clause_marks(governance):
    """"paragraphs 1.5, 1.4 and 1.3 ... of the Annex to RBI circular ..."

    That line wraps so a decimal lands at the start of a line while the
    document is inside clause 2. Accepting it would relabel every following
    chunk with a clause number belonging to a different instrument.
    """
    offset = _locate(governance, "Scale Based Regulation")
    mark = governance.section_for_offset(offset)
    assert mark is not None
    assert not mark.section_id.startswith("1."), (
        f"cross-reference leaked into clause detection: {mark.section_id}"
    )


# ---------------------------------------------------------------------------
# Failure behaviour
# ---------------------------------------------------------------------------


def test_a_missing_file_raises_pdf_extraction_error(tmp_path):
    with pytest.raises(PDFExtractionError):
        extract_pdf(tmp_path / "nope.pdf")


def test_a_non_pdf_raises_pdf_extraction_error(tmp_path):
    decoy = tmp_path / "not-really.pdf"
    decoy.write_text("this is plain text, not a PDF", encoding="utf-8")

    with pytest.raises(PDFExtractionError):
        extract_pdf(decoy)


def test_section_mark_is_immutable():
    mark = SectionMark("16(m)", None, 16, 1234)
    with pytest.raises(Exception):
        mark.section_id = "16.13"


# ---------------------------------------------------------------------------
# Phase D4: section titles, where the document actually supplies one
# ---------------------------------------------------------------------------


def test_a_sub_item_inherits_its_parent_clause_title(outsourcing):
    """16(m) has no heading of its own; it sits under clause 16's heading.

    That is what the document says, so that is what is reported. The title
    names the section the text is in; ``section_id`` still identifies the
    sub-item exactly. Nothing is invented -- a sub-item under an untitled
    parent stays untitled.
    """
    parent = next(m for m in outsourcing.sections if m.section_id == "16")
    assert parent.section_title == "Aspects to be considered in agreement"

    sub_items = [
        m for m in outsourcing.sections if m.section_id.startswith("16(")
    ]
    assert len(sub_items) > 10
    for mark in sub_items:
        assert mark.section_title == parent.section_title


def test_a_sub_item_carries_exactly_the_title_of_the_clause_it_follows(
    governance, outsourcing
):
    """Inheritance copies what exists; it does not manufacture a heading.

    Walked in document ORDER rather than keyed by section_id: a clause number
    is not unique across a Direction -- "5" appears once in the body and again
    inside an Annex, with different headings -- so a dict keyed on the number
    would compare a sub-item against the wrong parent.
    """
    for document in (governance, outsourcing):
        parent_title = None
        checked = 0
        for mark in document.sections:
            if mark.section_id.startswith("Chapter"):
                continue
            if "(" in mark.section_id:
                assert mark.section_title == parent_title, (
                    f"{mark.section_id} on page {mark.page_number} carries "
                    f"{mark.section_title!r}, not its parent's {parent_title!r}"
                )
                checked += 1
            else:
                parent_title = mark.section_title
        assert checked, "expected sub-items in this document"


def test_a_chapter_takes_the_name_printed_beneath_it(outsourcing):
    """RBI prints a chapter's name on the line below its number."""
    chapters = {
        m.section_id: m.section_title
        for m in outsourcing.sections
        if m.section_id.startswith("Chapter")
    }

    assert chapters, "expected chapter headings in this document"
    assert chapters.get("Chapter I") == "Preliminary"
    # Every detected chapter should have found a name on the next line.
    assert all(title for title in chapters.values())


def test_no_title_is_a_slice_of_requirement_text(
    outsourcing, governance, nbfc_outsourcing
):
    """Inherited and looked-ahead titles obey the same conservative test."""
    for document in (outsourcing, governance, nbfc_outsourcing):
        for mark in document.sections:
            if mark.section_title is None:
                continue
            assert len(mark.section_title) <= 60
            assert "," not in mark.section_title
            assert not mark.section_title.endswith((".", ";", ":"))


def test_titles_are_still_absent_where_the_document_prints_none(nbfc_outsourcing):
    """Coverage improved; it is not complete, and must not pretend to be.

    Many RBI clauses open straight into their requirement text with no
    heading at all. Those keep ``section_title = None``.
    """
    untitled = [m for m in nbfc_outsourcing.sections if m.section_title is None]
    assert untitled, "expected some clauses to genuinely have no heading"
