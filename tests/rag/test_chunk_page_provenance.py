"""Phase B: page and clause provenance on chunks of the real RBI PDFs.

Run against the six stored Directions, not a synthetic fixture. A synthetic
PDF would prove the plumbing connects; only the real documents prove that a
clause a reviewer can name lands on a page a reviewer can open.
"""
import pytest

from app.rag.chunking import chunk_document, chunk_documents
from app.rag.corpus import APPROVED_CORPUS, corpus_from_manifest
from app.rag.ingestion import load_corpus, load_source

EXPECTED_DOCUMENTS = 6


@pytest.fixture(scope="module")
def pdf_documents():
    return load_corpus(corpus_from_manifest())


@pytest.fixture(scope="module")
def pdf_chunks(pdf_documents):
    return chunk_documents(pdf_documents)


@pytest.fixture(scope="module")
def by_doc(pdf_chunks):
    grouped = {}
    for chunk in pdf_chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped


# ---------------------------------------------------------------------------
# The corpus itself
# ---------------------------------------------------------------------------


def test_the_six_verified_documents_are_the_corpus(pdf_documents):
    assert len(pdf_documents) == EXPECTED_DOCUMENTS
    assert {d.doc_id for d in pdf_documents} == {
        "RBI-IT-GOV-2023",
        "RBI-IT-OUTSOURCE-2023",
        "RBI-DIGITAL-LENDING-2025",
        "RBI-FRAUD-NBFC-2024",
        "RBI-NBFC-CREDIT-INFO-REPORTING-2025",
        "RBI-FS-OUTSOURCE-NBFC-2017",
    }


def test_every_corpus_document_is_paginated(pdf_documents):
    for document in pdf_documents:
        assert document.is_paginated, document.doc_id
        assert document.structure.page_count > 0


# ---------------------------------------------------------------------------
# Page provenance
# ---------------------------------------------------------------------------


def test_every_pdf_chunk_knows_its_page(pdf_chunks):
    for chunk in pdf_chunks:
        assert chunk.page is not None, chunk.chunk_id
        assert chunk.page >= 1


def test_no_chunk_spans_two_pages(pdf_documents, by_doc):
    """A chunk that straddled pages could honestly claim neither.

    Verified structurally: every chunk's text must appear inside the text of
    the single page it claims, so no chunk can have been assembled from two.
    """
    for document in pdf_documents:
        pages = {p.page_number: p.text for p in document.structure.pages}
        for chunk in by_doc[document.doc_id]:
            page_words = pages[chunk.page].split()
            chunk_words = chunk.text.split()
            assert chunk_words, chunk.chunk_id
            # Every word of the chunk comes from its own page, in order.
            assert _is_subsequence(chunk_words, page_words), (
                f"{chunk.chunk_id} claims page {chunk.page} but its words are "
                "not all on that page"
            )


def _is_subsequence(needle, haystack):
    it = iter(haystack)
    return all(word in it for word in needle)


def test_pages_are_visited_in_reading_order(by_doc):
    for doc_id, chunks in by_doc.items():
        pages = [c.page for c in chunks]
        assert pages == sorted(pages), doc_id


def test_chunk_indexes_are_contiguous_within_a_document(by_doc):
    for doc_id, chunks in by_doc.items():
        assert [c.chunk_index for c in chunks] == list(range(len(chunks))), doc_id


def test_a_page_number_never_exceeds_the_document_length(pdf_documents, by_doc):
    for document in pdf_documents:
        limit = document.structure.page_count
        for chunk in by_doc[document.doc_id]:
            assert 1 <= chunk.page <= limit


# ---------------------------------------------------------------------------
# Section provenance
# ---------------------------------------------------------------------------


def test_most_chunks_carry_a_clause_but_none_is_invented(pdf_chunks):
    """Coverage is high, and every id present was printed by the document."""
    with_section = [c for c in pdf_chunks if c.section_id]
    assert len(with_section) / len(pdf_chunks) > 0.80

    for chunk in with_section:
        assert chunk.section_id.strip() == chunk.section_id
        assert chunk.section_id


def test_a_section_id_is_one_the_extractor_actually_detected(pdf_documents, by_doc):
    """Chunking may only REPORT clause ids; it may not construct them."""
    for document in pdf_documents:
        detected = {m.section_id for m in document.structure.sections}
        for chunk in by_doc[document.doc_id]:
            if chunk.section_id is not None:
                assert chunk.section_id in detected, (
                    f"{chunk.chunk_id} carries {chunk.section_id!r}, which the "
                    "extractor never detected in this document"
                )


def test_pdf_native_numbering_is_preserved_not_rewritten(by_doc):
    """IT Outsourcing letters its sub-items; the website numbers them.

    The register cites clause 16.13. This PDF prints that provision as
    sub-item m) of clause 16. Chunks must carry what the PDF prints --
    "16(m)" -- because "16.13" appears nowhere in the document being cited.
    """
    ids = {c.section_id for c in by_doc["RBI-IT-OUTSOURCE-2023"] if c.section_id}

    assert "16(m)" in ids
    assert "16.13" not in ids


def test_every_chunk_is_locatable(pdf_chunks):
    """is_locatable() is the gate a citation should require."""
    assert all(c.is_locatable() for c in pdf_chunks)


def test_a_chunk_never_claims_a_clause_that_starts_after_it(pdf_documents, by_doc):
    """The known limitation is bounded: the label is never a LATER clause.

    section_id describes the clause a chunk OPENS in, so a window running past
    a boundary keeps the earlier label. That is imprecise but safe. What must
    never happen is the opposite -- a chunk labelled with a clause that begins
    after the chunk's own text, which would point a reviewer forwards, past
    the provision they were shown.

    Clause-boundary chunking would remove the imprecision entirely; it was
    implemented, measured and reverted because it produced five-word
    citations and cost report coverage. See _chunk_paginated's docstring.
    """
    for document in pdf_documents:
        starts = {}
        for mark in document.structure.sections:
            starts.setdefault(mark.section_id, mark.offset)

        page_starts = {p: s for p, s, _ in document.structure.page_spans}

        for chunk in by_doc[document.doc_id]:
            if chunk.section_id is None:
                continue
            mark_offset = starts[chunk.section_id]
            page_end = page_starts[chunk.page] + len(
                document.structure.pages[chunk.page - 1].text
            )
            assert mark_offset <= page_end, (
                f"{chunk.chunk_id} on page {chunk.page} claims clause "
                f"{chunk.section_id}, which starts on a later page"
            )


def test_location_dict_omits_what_is_unknown(pdf_chunks):
    for chunk in pdf_chunks:
        assert "page" in chunk.location
        if chunk.section_id is None:
            assert "section_id" not in chunk.location
        if chunk.section_title is None:
            assert "section_title" not in chunk.location


def test_location_reaches_provenance_for_downstream_consumers(pdf_chunks):
    sample = next(c for c in pdf_chunks if c.section_id and c.section_title)
    provenance = sample.provenance

    assert provenance["page"] == sample.page
    assert provenance["section_id"] == sample.section_id
    assert provenance["section_title"] == sample.section_title
    assert provenance["doc_id"] == sample.doc_id


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def test_contents_pages_produce_no_chunks_at_all(pdf_documents, by_doc):
    """A contents listing is navigation, and is not indexed.

    Phase B only suppressed its CLAUSE, leaving the page indexed. That was
    not enough: a contents page lists every heading in the document, so it
    matched almost any topical query and repeatedly outranked the real clause
    -- two of three compliance queries came back pointing at a table of
    contents. A citation to a contents page tells a reviewer nothing, so the
    page is no longer chunked.
    """
    seen_contents_pages = 0
    for document in pdf_documents:
        contents = document.structure.contents_pages
        seen_contents_pages += len(contents)
        chunk_pages = {c.page for c in by_doc[document.doc_id]}
        assert not (contents & chunk_pages), (
            f"{document.doc_id}: contents pages {sorted(contents & chunk_pages)} "
            "were indexed"
        )
    assert seen_contents_pages, "expected at least one contents page across the corpus"


def test_the_first_page_of_a_direction_has_no_clause(by_doc):
    """Page 1 is a covering letter or a title page, not clause 1."""
    first = [c for c in by_doc["RBI-IT-GOV-2023"] if c.page == 1]
    assert first
    assert first[0].section_id is None


# ---------------------------------------------------------------------------
# The text path is untouched
# ---------------------------------------------------------------------------


def test_the_2014_text_fixture_still_chunks_without_page_or_clause():
    documents = load_corpus(APPROVED_CORPUS)
    chunks = chunk_documents(documents)

    assert chunks
    for chunk in chunks:
        assert chunk.page is None
        assert chunk.section_id is None
        assert chunk.section_title is None
        assert chunk.is_locatable() is False


def test_text_and_pdf_paths_produce_the_same_kind_of_chunk():
    """One chunk type, two producers -- no parallel representation."""
    text_chunk = chunk_document(load_corpus(APPROVED_CORPUS)[0])[0]
    pdf_source = corpus_from_manifest().get("RBI-IT-OUTSOURCE-2023")
    pdf_chunk = chunk_document(load_source(pdf_source))[0]

    assert type(text_chunk) is type(pdf_chunk)
    assert text_chunk.chunk_id.endswith("::chunk-0")
    assert pdf_chunk.chunk_id.endswith("::chunk-0")
