"""Chunk-level page/section provenance, and its survival into ChromaDB.

A retrieved chunk must stay traceable to the exact place in the RBI document
it came from. Without page or section, a compliance citation reads "as per
the RBI circular" — unverifiable, and therefore worthless as evidence.

The companion property: unknown location is represented as ABSENT, never as
a placeholder. A citation pointing at "page 1" because no page was known is
worse than one admitting it has none, since it looks checkable and isn't.
"""

from __future__ import annotations

import pytest

from app.rag.chunking import DocumentChunk, chunk_document
from app.rag.corpus import RBISourceMetadata
from app.rag.ingestion import LoadedDocument
from app.rag.vector_store import ChunkVectorStore, chunk_metadata


@pytest.fixture
def source():
    return RBISourceMetadata(
        doc_id="TEST-DOC-2024",
        title="Test Source Document",
        issuing_authority="RBI",
        document_type="master_direction",
        publication_date="2024-01-01",
        source_url="local-fixture",
        local_path="tests/fixtures/none.txt",
        is_current=True,
    )


def test_chunk_accepts_page_and_section(source):
    chunk = DocumentChunk(
        text="fixture text",
        chunk_index=0,
        source_metadata=source,
        page=12,
        section_id="4.2",
        section_title="Model Validation",
        paragraph=3,
    )

    assert chunk.page == 12
    assert chunk.section_id == "4.2"
    assert chunk.section_title == "Model Validation"
    assert chunk.paragraph == 3


def test_every_chunk_has_document_id(source):
    chunk = DocumentChunk(text="t", chunk_index=0, source_metadata=source)
    assert chunk.doc_id == "TEST-DOC-2024"
    assert chunk.provenance["doc_id"] == "TEST-DOC-2024"


def test_chunk_preserves_document_id_through_chroma_metadata(source):
    chunk = DocumentChunk(text="t", chunk_index=0, source_metadata=source, page=5)
    metadata = chunk_metadata(chunk)

    assert metadata["doc_id"] == "TEST-DOC-2024"
    assert metadata["chunk_id"] == "TEST-DOC-2024::chunk-0"


def test_page_and_section_survive_into_chroma_metadata(source):
    """Chroma metadata is flat scalars; provenance must not be lost to that."""
    chunk = DocumentChunk(
        text="t",
        chunk_index=0,
        source_metadata=source,
        page=12,
        section_id="4.2",
        section_title="Model Validation",
        paragraph=3,
    )
    metadata = chunk_metadata(chunk)

    assert metadata["page"] == 12
    assert metadata["section_id"] == "4.2"
    assert metadata["section_title"] == "Model Validation"
    assert metadata["paragraph"] == 3


def test_unknown_location_is_absent_not_a_placeholder(source):
    """The property that keeps citations honest."""
    chunk = DocumentChunk(text="t", chunk_index=0, source_metadata=source)
    metadata = chunk_metadata(chunk)

    for field in ("page", "section_id", "section_title", "paragraph"):
        assert field not in metadata, f"{field} was invented for a chunk that has none"
    assert chunk.location == {}


def test_is_locatable_reports_whether_a_citation_can_be_anchored(source):
    plain = DocumentChunk(text="t", chunk_index=0, source_metadata=source)
    paged = DocumentChunk(text="t", chunk_index=0, source_metadata=source, page=7)
    sectioned = DocumentChunk(text="t", chunk_index=0, source_metadata=source, section_id="2.1")

    assert plain.is_locatable() is False
    assert paged.is_locatable() is True
    assert sectioned.is_locatable() is True


def test_existing_plain_text_chunking_still_produces_unlocatable_chunks(source):
    """The 2014 excerpt is plain text with no structure; it must not pretend.

    This is the backward-compatibility case: existing chunking keeps working
    and honestly reports that it cannot anchor a page.
    """
    document = LoadedDocument(
        metadata=source,
        text="alpha beta gamma delta epsilon zeta",
        path="tests/fixtures/none.txt",
    )
    chunks = chunk_document(document, chunk_size_words=3)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.page is None
        assert chunk.section_id is None
        assert chunk.is_locatable() is False


def test_chunks_round_trip_through_the_vector_store_with_location(source):
    """End to end: page/section must come back out of ChromaDB."""
    store = ChunkVectorStore.in_memory(name_prefix="test-provenance")
    try:
        store.add_chunks([
            DocumentChunk(
                text="model validation independent review",
                chunk_index=0,
                source_metadata=source,
                page=12,
                section_id="4.2",
            )
        ])

        record = store.get_chunk("TEST-DOC-2024::chunk-0")
        assert record is not None
        assert record["metadata"]["page"] == 12
        assert record["metadata"]["section_id"] == "4.2"
        assert record["metadata"]["doc_id"] == "TEST-DOC-2024"
    finally:
        store.clear()
