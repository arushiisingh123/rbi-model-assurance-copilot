"""Phase 3A, Task 4 tests: RBI chunk embeddings + ChromaDB vector store.

Owner: Nidhi. Covers only the embedding function and the storage/indexing
layer. The retrieval API (ranking, citations, no-evidence handling), the
LLM, and report generation are later tasks and are not exercised here.
Every store is created with ``ChunkVectorStore.in_memory()`` so tests are
isolated and offline. The Phase 0 smoke-test files are left untouched.
"""
import collections
import math
from pathlib import Path

import pytest

from app.rag.chunking import chunk_document, chunk_documents
from app.rag.corpus import RBICorpus, RBISourceMetadata
from app.rag.ingestion import LoadedDocument, load_by_id, load_corpus
from app.rag.vector_store import (
    ChunkVectorStore,
    DeterministicHashEmbedding,
    chunk_metadata,
)

APPROVED_2014_ID = "rbi-master-circular-irac-advances-2014-07-01"


def _doc(text: str, *, doc_id: str = "doc-x", **meta_overrides) -> LoadedDocument:
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
def real_2014_chunks():
    return chunk_document(load_by_id(APPROVED_2014_ID))


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ======================================================================
# DeterministicHashEmbedding
# ======================================================================


def test_embedding_generation_produces_one_vector_per_text():
    e = DeterministicHashEmbedding()
    vectors = e(["non performing asset", "overdue", "x y z"])
    assert len(vectors) == 3
    assert all(len(v) == e.dim for v in vectors)
    assert all(float(x) == x for v in vectors for x in v)  # every value is numeric
    # the plain-Python method returns a list of floats
    plain = e.embed_text("non performing asset")
    assert isinstance(plain, list)
    assert all(isinstance(x, float) for x in plain)


def test_embedding_dimension_is_consistent_and_configurable():
    default = DeterministicHashEmbedding()
    assert default.dim == 256
    assert all(len(v) == 256 for v in default(["short", "a much longer piece of text here"]))

    small = DeterministicHashEmbedding(dim=32)
    assert small.dim == 32
    assert all(len(v) == 32 for v in small(["alpha", "beta gamma delta"]))


def test_embedding_is_deterministic():
    e = DeterministicHashEmbedding()
    assert e.embed_text("non performing asset") == e.embed_text("non performing asset")
    # __call__ agrees with embed_text (compared as plain Python lists;
    # ChromaDB's EmbeddingFunction wrapper hands __call__ output back as
    # numpy arrays)
    texts = ["prudential norms", "asset classification"]
    via_call = [list(v) for v in e(texts)]
    via_method = [e.embed_text(t) for t in texts]
    assert via_call == via_method


def test_nonempty_text_is_l2_normalised_empty_text_is_zero():
    e = DeterministicHashEmbedding()
    vec = e.embed_text("income recognition asset classification")
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)
    assert all(x == 0.0 for x in e.embed_text("   \n\t "))


def test_embedding_tracks_lexical_overlap():
    e = DeterministicHashEmbedding()
    base = e.embed_text("non performing asset overdue ninety days")
    near = e.embed_text("non performing asset overdue")
    far = e.embed_text("completely different unrelated vocabulary entirely")
    assert _cosine(base, near) > _cosine(base, far)


def test_embedding_config_roundtrip_and_name():
    e = DeterministicHashEmbedding(dim=16)
    assert e.get_config() == {"dim": 16}
    rebuilt = DeterministicHashEmbedding.build_from_config(e.get_config())
    assert rebuilt.dim == 16
    assert isinstance(DeterministicHashEmbedding.name(), str)
    assert DeterministicHashEmbedding.name()


def test_embedding_rejects_bad_dim():
    for bad in (0, -4, 2.5, True, "8"):
        with pytest.raises(ValueError):
            DeterministicHashEmbedding(dim=bad)  # type: ignore[arg-type]


# ======================================================================
# ChunkVectorStore -- indexing
# ======================================================================


def test_chunks_can_be_indexed(real_2014_chunks):
    store = ChunkVectorStore.in_memory()
    assert store.count() == 0
    written = store.add_chunks(real_2014_chunks)
    assert written == len(real_2014_chunks)
    assert store.count() == len(real_2014_chunks)


def test_adding_an_empty_iterable_is_a_noop():
    store = ChunkVectorStore.in_memory()
    assert store.add_chunks([]) == 0
    assert store.count() == 0


def test_indexed_chunk_preserves_text_and_metadata(real_2014_chunks):
    store = ChunkVectorStore.in_memory()
    store.add_chunks(real_2014_chunks)

    chunk = real_2014_chunks[1]
    stored = store.get_chunk(chunk.chunk_id)
    assert stored is not None
    assert stored["text"] == chunk.text

    meta = stored["metadata"]
    assert meta["doc_id"] == APPROVED_2014_ID
    assert meta["chunk_index"] == chunk.chunk_index
    assert meta["chunk_id"] == chunk.chunk_id
    assert meta["title"] == chunk.source_metadata.title
    assert meta["source_url"] == chunk.source_metadata.source_url
    assert meta["document_type"] == "Master Circular"
    assert meta["publication_date"] == "2014-07-01"
    assert meta["reference_number"] == chunk.source_metadata.reference_number
    assert meta["is_excerpt"] is True
    assert meta["is_current"] is False


def test_none_provenance_fields_are_omitted_not_invented():
    # The 2014 excerpt has effective_date = None.
    chunk = chunk_document(load_by_id(APPROVED_2014_ID))[0]
    meta = chunk_metadata(chunk)
    assert "effective_date" not in meta
    assert meta["doc_id"] == APPROVED_2014_ID
    assert meta["chunk_index"] == 0

    # A synthetic source with several unset optional fields.
    sparse = chunk_document(_doc("alpha beta gamma", doc_id="sparse"))[0]
    sparse_meta = chunk_metadata(sparse)
    for absent in ("source_url", "reference_number", "publication_date", "effective_date"):
        assert absent not in sparse_meta
    assert sparse_meta["doc_id"] == "sparse"
    assert sparse_meta["chunk_index"] == 0
    assert sparse_meta["chunk_id"] == "sparse::chunk-0"


def test_repeated_indexing_is_idempotent_and_predictable(real_2014_chunks):
    store = ChunkVectorStore.in_memory()
    store.add_chunks(real_2014_chunks)
    first = store.count()
    store.add_chunks(real_2014_chunks)
    assert store.count() == first  # upsert by chunk_id, no growth

    # Re-index one chunk_id with different text -> content updated, count stable.
    edited = chunk_document(_doc("brand new replacement text", doc_id=APPROVED_2014_ID))[0]
    # force the id to collide with an existing stored chunk
    object.__setattr__(edited, "chunk_index", real_2014_chunks[0].chunk_index)
    store.add_chunks([edited])
    assert store.count() == first
    assert store.get_chunk(edited.chunk_id)["text"] == "brand new replacement text"


# ======================================================================
# ChunkVectorStore -- multiple documents in one index
# ======================================================================


def test_multiple_documents_coexist_in_one_index():
    store = ChunkVectorStore.in_memory()
    a = _doc(" ".join(["aaa"] * 50), doc_id="doc-a", is_excerpt=True, is_current=False,
             coverage_note="part 1 only")
    b = _doc(" ".join(["bbb"] * 30), doc_id="doc-b", is_excerpt=False, is_current=True)
    store.add_chunks(chunk_documents([a, b], chunk_size_words=20))

    assert store.count() == 3 + 2

    only_a = store.query("aaa", n_results=10, where={"doc_id": "doc-a"})
    assert only_a["ids"][0]
    assert all(m["doc_id"] == "doc-a" for m in only_a["metadatas"][0])
    assert all(m["is_excerpt"] is True for m in only_a["metadatas"][0])

    only_b = store.query("bbb", n_results=10, where={"doc_id": "doc-b"})
    assert all(m["doc_id"] == "doc-b" for m in only_b["metadatas"][0])
    assert all(m["is_current"] is True for m in only_b["metadatas"][0])


def test_excerpt_vs_verified_current_distinction_is_stored_per_chunk():
    store = ChunkVectorStore.in_memory()
    historical = _doc(" ".join(["h"] * 45), doc_id="historical", is_excerpt=True,
                      is_current=False, coverage_note="sections 1-2")
    current = _doc(" ".join(["c"] * 45), doc_id="current-full", is_excerpt=False, is_current=True)
    store.add_chunks(chunk_documents([historical, current], chunk_size_words=40))

    hist = store.query("h", n_results=10, where={"doc_id": "historical"})["metadatas"][0]
    assert hist and all(m["is_excerpt"] is True and m["is_current"] is False for m in hist)

    cur = store.query("c", n_results=10, where={"doc_id": "current-full"})["metadatas"][0]
    assert cur and all(m["is_current"] is True and m["is_excerpt"] is False for m in cur)


# ======================================================================
# ChunkVectorStore -- isolation
# ======================================================================


def test_in_memory_stores_are_isolated():
    one = ChunkVectorStore.in_memory()
    two = ChunkVectorStore.in_memory()
    assert one.collection_name != two.collection_name

    one.add_chunks(chunk_document(_doc("some text here for one", doc_id="d1")))
    assert one.count() == 1
    assert two.count() == 0


def test_clear_empties_the_store_but_keeps_it_usable(real_2014_chunks):
    store = ChunkVectorStore.in_memory()
    store.add_chunks(real_2014_chunks)
    assert store.count() > 0
    store.clear()
    assert store.count() == 0
    store.add_chunks(real_2014_chunks)
    assert store.count() == len(real_2014_chunks)


# ======================================================================
# ChunkVectorStore -- query passthrough
# ======================================================================


def test_query_returns_the_relevant_indexed_chunk(real_2014_chunks):
    store = ChunkVectorStore.in_memory()
    store.add_chunks(real_2014_chunks)
    result = store.query("what is a non performing asset", n_results=1)
    assert result["metadatas"][0][0]["doc_id"] == APPROVED_2014_ID
    assert "non performing" in result["documents"][0][0].lower()


def test_query_on_empty_store_returns_empty_shape():
    store = ChunkVectorStore.in_memory()
    result = store.query(["anything"], n_results=3)
    assert result["ids"] == [[]]
    assert result["documents"] == [[]]


# ======================================================================
# rebuild_from_corpus
# ======================================================================


def test_rebuild_from_the_approved_corpus():
    store = ChunkVectorStore.in_memory()
    summary = store.rebuild_from_corpus()
    assert summary["documents"] == 1
    assert summary["chunks"] > 1
    assert summary["doc_ids"] == [APPROVED_2014_ID]
    assert store.count() == summary["chunks"]

    sample = store.query("prudential norms advances", n_results=3)
    for meta in sample["metadatas"][0]:
        assert meta["doc_id"] == APPROVED_2014_ID
        assert meta["is_excerpt"] is True
        assert meta["is_current"] is False


def test_rebuild_is_repeatable_and_does_not_accumulate():
    store = ChunkVectorStore.in_memory()
    first = store.rebuild_from_corpus()
    count_after_first = store.count()
    second = store.rebuild_from_corpus()
    assert second == first
    assert store.count() == count_after_first


def test_rebuild_from_a_custom_multi_document_corpus(tmp_path):
    (tmp_path / "data" / "rbi_sources").mkdir(parents=True)
    (tmp_path / "data" / "rbi_sources" / "a.txt").write_text(
        " ".join(["alpha"] * 60), encoding="utf-8"
    )
    (tmp_path / "data" / "rbi_sources" / "b.txt").write_text(
        " ".join(["bravo"] * 45), encoding="utf-8"
    )
    corpus = RBICorpus(
        [
            RBISourceMetadata(
                doc_id="cust-a", title="Custom A",
                issuing_authority="Reserve Bank of India (RBI)",
                document_type="Master Direction",
                local_path="data/rbi_sources/a.txt",
            ),
            RBISourceMetadata(
                doc_id="cust-b", title="Custom B",
                issuing_authority="Reserve Bank of India (RBI)",
                document_type="Notification",
                local_path="data/rbi_sources/b.txt",
            ),
        ]
    )
    store = ChunkVectorStore.in_memory()
    summary = store.rebuild_from_corpus(corpus, repo_root=tmp_path, chunk_size_words=40)

    assert summary["documents"] == 2
    assert set(summary["doc_ids"]) == {"cust-a", "cust-b"}
    assert store.count() == summary["chunks"]

    a_meta = store.query("alpha", n_results=10, where={"doc_id": "cust-a"})["metadatas"][0]
    b_meta = store.query("bravo", n_results=10, where={"doc_id": "cust-b"})["metadatas"][0]
    assert a_meta and b_meta


def test_rebuilt_index_contains_only_ingested_words_no_invention():
    store = ChunkVectorStore.in_memory()
    summary = store.rebuild_from_corpus()

    ingested_words = load_corpus()[0].text.split()

    # Reassemble every stored chunk's text via its deterministic chunk_id.
    indexed_words: list[str] = []
    for i in range(summary["chunks"]):
        stored = store.get_chunk(f"{APPROVED_2014_ID}::chunk-{i}")
        assert stored is not None
        indexed_words.extend(stored["text"].split())

    assert collections.Counter(indexed_words) == collections.Counter(ingested_words)


# ======================================================================
# Offline / decoupling
# ======================================================================


def test_default_embedding_is_the_offline_deterministic_one():
    store = ChunkVectorStore.in_memory()
    assert isinstance(store.embedding, DeterministicHashEmbedding)
    # constructible and usable with zero external configuration
    assert len(DeterministicHashEmbedding().embed_text("offline")) == 256


def test_indexing_and_query_work_without_network(monkeypatch):
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("unexpected network connect() during offline RAG indexing")

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=True)

    store = ChunkVectorStore.in_memory()
    store.add_chunks(chunk_document(_doc("alpha beta gamma delta epsilon", doc_id="net-test")))
    assert store.count() == 1
    result = store.query("alpha", n_results=1)
    assert result["ids"][0]


def test_vector_store_does_not_import_the_phase0_smoke_test():
    import ast
    import inspect

    import app.rag.vector_store as vs_module
    import app.rag.smoke_test  # noqa: F401 -- unchanged, must still import

    tree = ast.parse(inspect.getsource(vs_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("smoke_test" in name for name in imported), (
        f"vector_store must not import the Phase 0 smoke test; imports: {sorted(imported)}"
    )
