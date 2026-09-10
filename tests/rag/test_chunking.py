"""Phase 3A, Task 3 tests: RBI document chunking with metadata preservation.

Owner: Nidhi. Covers only splitting an ingested document into ordered
chunks that keep their source metadata. Embeddings, vector search,
retrieval, and report generation are later tasks and are not exercised
here. The Phase 0 smoke-test files are left untouched.
"""
import collections
from pathlib import Path

import pytest

from app.rag.corpus import RBISourceMetadata
from app.rag.ingestion import PROVENANCE_FIELDS, LoadedDocument, load_by_id, provenance
from app.rag.chunking import (
    DEFAULT_CHUNK_SIZE_WORDS,
    DocumentChunk,
    chunk_document,
    chunk_documents,
)

APPROVED_2014_ID = "rbi-master-circular-irac-advances-2014-07-01"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _doc(text: str, *, doc_id: str = "doc-x", **meta_overrides) -> LoadedDocument:
    """A LoadedDocument with arbitrary in-memory text and synthetic metadata."""
    meta_kwargs = dict(
        doc_id=doc_id,
        title=f"Title for {doc_id}",
        issuing_authority="Reserve Bank of India (RBI)",
        document_type="Master Direction",
        local_path=f"data/rbi_sources/{doc_id}.txt",
    )
    meta_kwargs.update(meta_overrides)
    return LoadedDocument(
        metadata=RBISourceMetadata(**meta_kwargs),
        text=text,
        path=Path(f"/tmp/{doc_id}.txt"),
    )


@pytest.fixture(scope="module")
def real_2014_document() -> LoadedDocument:
    return load_by_id(APPROVED_2014_ID)


# --------------------------------------------------------------------------
# Text becomes multiple chunks / short document works
# --------------------------------------------------------------------------


def test_real_document_splits_into_multiple_chunks(real_2014_document):
    chunks = chunk_document(real_2014_document)
    assert len(chunks) > 1
    assert all(isinstance(c, DocumentChunk) for c in chunks)


def test_hundred_words_at_size_forty_makes_three_chunks():
    chunks = chunk_document(_doc(" ".join(["word"] * 100)), chunk_size_words=40)
    assert [len(c.text.split()) for c in chunks] == [40, 40, 20]


def test_short_document_yields_a_single_chunk():
    chunks = chunk_document(_doc("only five short words here"))
    assert len(chunks) == 1
    assert chunks[0].text == "only five short words here"
    assert chunks[0].chunk_index == 0


def test_document_shorter_than_one_chunk_is_not_padded():
    chunks = chunk_document(_doc("a b c"), chunk_size_words=40)
    assert len(chunks) == 1
    assert chunks[0].text == "a b c"


# --------------------------------------------------------------------------
# No empty chunks
# --------------------------------------------------------------------------


def test_no_empty_chunks_for_the_real_document(real_2014_document):
    chunks = chunk_document(real_2014_document)
    for c in chunks:
        assert c.text.strip()
        assert c.text.split()


def test_messy_whitespace_never_produces_an_empty_chunk():
    text = "  alpha\n\n\n   beta\t\t gamma \r\n   delta   "
    chunks = chunk_document(_doc(text), chunk_size_words=2)
    assert [c.text for c in chunks] == ["alpha beta", "gamma delta"]
    for c in chunks:
        assert c.text.strip()


def test_document_with_no_words_raises_clearly():
    with pytest.raises(ValueError, match="no words to chunk"):
        chunk_document(_doc("   \n\t\r\n   "))


# --------------------------------------------------------------------------
# Ordering, indices, determinism
# --------------------------------------------------------------------------


def test_chunk_indices_are_contiguous_and_start_at_zero(real_2014_document):
    chunks = chunk_document(real_2014_document)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_are_in_reading_order():
    chunks = chunk_document(_doc(" ".join(str(i) for i in range(25))), chunk_size_words=5)
    assert [c.text for c in chunks] == [
        "0 1 2 3 4",
        "5 6 7 8 9",
        "10 11 12 13 14",
        "15 16 17 18 19",
        "20 21 22 23 24",
    ]


def test_chunking_is_deterministic(real_2014_document):
    first = chunk_document(real_2014_document)
    second = chunk_document(real_2014_document)
    assert [(c.chunk_index, c.text) for c in first] == [
        (c.chunk_index, c.text) for c in second
    ]
    # Same source object each run, not a rebuilt copy.
    assert all(c.source_metadata is real_2014_document.metadata for c in first)


def test_chunk_ids_are_stable_and_corpus_unique():
    a = chunk_document(_doc(" ".join(["x"] * 90), doc_id="doc-a"), chunk_size_words=40)
    b = chunk_document(_doc(" ".join(["y"] * 90), doc_id="doc-b"), chunk_size_words=40)
    assert [c.chunk_id for c in a] == [
        "doc-a::chunk-0",
        "doc-a::chunk-1",
        "doc-a::chunk-2",
    ]
    all_ids = [c.chunk_id for c in a + b]
    assert len(all_ids) == len(set(all_ids))


# --------------------------------------------------------------------------
# No loss of metadata / no invented text
# --------------------------------------------------------------------------


def test_every_chunk_preserves_the_exact_source_metadata_object(real_2014_document):
    chunks = chunk_document(real_2014_document)
    for c in chunks:
        assert c.source_metadata is real_2014_document.metadata


def test_every_chunk_carries_all_provenance_fields(real_2014_document):
    source = real_2014_document.metadata
    expected = provenance(source)
    for c in chunk_document(real_2014_document):
        assert set(c.provenance) == set(PROVENANCE_FIELDS)
        assert c.provenance == expected
        # the ten identifiers the task calls out explicitly
        assert c.provenance["doc_id"] == source.doc_id
        assert c.provenance["title"] == source.title
        assert c.provenance["source_url"] == source.source_url
        assert c.provenance["document_type"] == source.document_type
        assert c.provenance["publication_date"] == source.publication_date
        assert c.provenance["effective_date"] == source.effective_date
        assert c.provenance["reference_number"] == source.reference_number
        assert c.provenance["is_excerpt"] == source.is_excerpt
        assert c.provenance["is_current"] == source.is_current
        assert c.provenance["scope_note"] == source.scope_note


def test_effective_date_stays_none_and_is_not_invented(real_2014_document):
    for c in chunk_document(real_2014_document):
        assert c.provenance["effective_date"] is None


def test_chunks_contain_only_words_from_the_source_no_loss_no_invention(real_2014_document):
    source_words = real_2014_document.text.split()
    chunk_words = [w for c in chunk_document(real_2014_document) for w in c.text.split()]
    # No word added, removed, duplicated, or reordered (no overlap).
    assert chunk_words == source_words
    assert collections.Counter(chunk_words) == collections.Counter(source_words)


# --------------------------------------------------------------------------
# Multiple documents remain distinguishable
# --------------------------------------------------------------------------


def test_multiple_documents_stay_separable_by_doc_id():
    a = _doc(" ".join(["aaa"] * 50), doc_id="doc-a")
    b = _doc(" ".join(["bbb"] * 30), doc_id="doc-b")
    chunks = chunk_documents([a, b], chunk_size_words=20)

    by_doc = collections.defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)

    assert set(by_doc) == {"doc-a", "doc-b"}
    # each document re-indexed from 0
    assert [c.chunk_index for c in by_doc["doc-a"]] == [0, 1, 2]
    assert [c.chunk_index for c in by_doc["doc-b"]] == [0, 1]
    # text never bleeds between documents
    assert all(set(c.text.split()) == {"aaa"} for c in by_doc["doc-a"])
    assert all(set(c.text.split()) == {"bbb"} for c in by_doc["doc-b"])


def test_chunk_documents_preserves_order_of_documents():
    a = _doc("alpha one two", doc_id="first")
    b = _doc("bravo three four", doc_id="second")
    chunks = chunk_documents([a, b])
    assert [c.doc_id for c in chunks] == ["first", "second"]


# --------------------------------------------------------------------------
# Historical/excerpt vs. verified current distinction survives chunking
# --------------------------------------------------------------------------


def test_excerpt_and_current_provenance_survive_every_chunk():
    historical = _doc(
        " ".join(["h"] * 60),
        doc_id="historical-excerpt",
        is_excerpt=True,
        is_current=False,
        coverage_note="Only sections 1-2 of the document.",
        scope_note="Historical excerpt; not current or binding.",
    )
    current = _doc(
        " ".join(["c"] * 60),
        doc_id="verified-current",
        is_excerpt=False,
        is_current=True,
    )

    chunks = chunk_documents([historical, current], chunk_size_words=40)
    by_doc = {c.doc_id: [] for c in chunks}
    for c in chunks:
        by_doc[c.doc_id].append(c)

    for c in by_doc["historical-excerpt"]:
        assert c.source_metadata.is_excerpt is True
        assert c.source_metadata.is_current is False
        assert c.source_metadata.is_verified_current_regulation is False
        assert c.provenance["is_excerpt"] is True

    for c in by_doc["verified-current"]:
        assert c.source_metadata.is_verified_current_regulation is True
        assert c.provenance["is_current"] is True


def test_real_approved_source_chunks_are_all_marked_historical_excerpt(real_2014_document):
    for c in chunk_document(real_2014_document):
        assert c.source_metadata.is_excerpt is True
        assert c.source_metadata.is_verified_current_regulation is False
        assert "not be treated as current or binding" in c.provenance["scope_note"].lower()


# --------------------------------------------------------------------------
# Input validation / config
# --------------------------------------------------------------------------


def test_chunk_size_is_respected():
    chunks = chunk_document(_doc(" ".join(["w"] * 95)), chunk_size_words=10)
    sizes = [len(c.text.split()) for c in chunks]
    assert sizes == [10, 10, 10, 10, 10, 10, 10, 10, 10, 5]


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size_words must be >= 1"):
        chunk_document(_doc("a b c"), chunk_size_words=0)


def test_non_loadeddocument_input_is_rejected():
    with pytest.raises(TypeError):
        chunk_document("plain string")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        chunk_document({"text": "x"})  # type: ignore[arg-type]


def test_default_chunk_size_is_forty():
    assert DEFAULT_CHUNK_SIZE_WORDS == 40


# --------------------------------------------------------------------------
# Decoupled from the Phase 0 smoke test
# --------------------------------------------------------------------------


def test_chunking_does_not_import_the_phase0_smoke_test():
    import ast
    import inspect

    import app.rag.chunking as chunking_module
    import app.rag.smoke_test  # noqa: F401 -- unchanged, must still import

    tree = ast.parse(inspect.getsource(chunking_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert not any("smoke_test" in name for name in imported), (
        f"chunking must not import the Phase 0 smoke test; imports: {sorted(imported)}"
    )
