"""Phase 3A, Task 5 tests: RBI vector retrieval.

Owner: Nidhi. Covers the relevance-gated retrieval layer only -- no LLM,
no report generation, no compliance/API/dashboard. Every retriever is
built on ``ChunkVectorStore.in_memory()`` with the deterministic offline
embedding, so tests are reproducible and need no network. The Phase 0
smoke-test files are left untouched.
"""
from pathlib import Path

import pytest

from app.rag.chunking import chunk_document, chunk_documents
from app.rag.corpus import APPROVED_CORPUS, RBISourceMetadata
from app.rag.ingestion import LoadedDocument, load_by_id
from app.rag.vector_store import ChunkVectorStore
from app.rag.retrieval import (
    DEFAULT_TOP_K,
    EVIDENCE_RETRIEVED,
    NO_VERIFIED_EVIDENCE,
    RBIRetriever,
    build_default_retriever,
)

APPROVED_2014_ID = "rbi-master-circular-irac-advances-2014-07-01"

# Contract minimum + the stable extras this implementation always returns.
REQUIRED_MIN_KEYS = {"retrieved_text", "source", "chunk_index"}
FULL_RESULT_KEYS = {
    "query", "evidence_status", "reason", "retrieved_text", "source",
    "distance", "provenance", "results",
    "doc_id", "chunk_id", "chunk_index", "source_url", "title",
    "publication_date", "document_type", "is_excerpt", "is_current",
}


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


def _retriever_over(*docs: LoadedDocument, **retriever_kwargs) -> RBIRetriever:
    store = ChunkVectorStore.in_memory()
    store.add_chunks(chunk_documents(list(docs)))
    return RBIRetriever(store, **retriever_kwargs)


@pytest.fixture(scope="module")
def real_2014_doc() -> LoadedDocument:
    return load_by_id(APPROVED_2014_ID)


@pytest.fixture(scope="module")
def default_retriever() -> RBIRetriever:
    return build_default_retriever(APPROVED_CORPUS)


# ======================================================================
# 1-2. A relevant query retrieves relevant RBI text with the right source
# ======================================================================


def test_relevant_query_retrieves_relevant_text(default_retriever):
    result = default_retriever.retrieve("non performing asset overdue ninety days")
    assert result["evidence_status"] == EVIDENCE_RETRIEVED
    assert result["retrieved_text"].strip()
    lowered = result["retrieved_text"].lower()
    assert "performing" in lowered and "asset" in lowered


def test_returned_source_is_the_real_document_title(default_retriever):
    result = default_retriever.retrieve("income recognition accrual basis")
    assert result["source"] == (
        "Master Circular - Prudential Norms on Income Recognition, Asset "
        "Classification and Provisioning pertaining to Advances"
    )
    assert result["source"] == result["title"]
    assert result["doc_id"] == APPROVED_2014_ID
    assert result["source_url"].startswith("https://www.rbi.org.in/")
    assert result["document_type"] == "Master Circular"
    assert result["publication_date"] == "2014-07-01"


# ======================================================================
# 3-5. chunk_index / chunk_id / doc_id / provenance are preserved exactly
# ======================================================================


def test_chunk_index_and_ids_match_the_stored_chunk(real_2014_doc):
    store = ChunkVectorStore.in_memory()
    store.add_chunks(chunk_document(real_2014_doc))
    retriever = RBIRetriever(store)

    result = retriever.retrieve("out of order status overdraft cash credit account")
    assert result["evidence_status"] == EVIDENCE_RETRIEVED

    stored = store.get_chunk(result["chunk_id"])
    assert stored is not None
    assert result["chunk_index"] == stored["metadata"]["chunk_index"]
    assert result["chunk_id"] == stored["metadata"]["chunk_id"]
    assert result["doc_id"] == stored["metadata"]["doc_id"]
    # exact retrieved text, not paraphrased
    assert result["retrieved_text"] == stored["text"]


def test_provenance_is_the_stored_chroma_metadata_not_reconstructed(default_retriever):
    result = default_retriever.retrieve("prudential norms advances classification")
    prov = result["provenance"]
    assert isinstance(prov, dict)
    assert prov["doc_id"] == APPROVED_2014_ID
    assert prov["chunk_id"] == result["chunk_id"]
    assert prov["chunk_index"] == result["chunk_index"]
    assert prov["title"] == result["title"]
    assert prov["is_excerpt"] is True
    # the 2014 source states no effective date -> it was never stored
    assert "effective_date" not in prov


def test_result_shape_is_stable_and_meets_the_minimum_contract(default_retriever):
    found = default_retriever.retrieve("non performing asset")
    missing = default_retriever.retrieve("completely unrelated nonsense vocabulary here")
    assert REQUIRED_MIN_KEYS <= set(found)
    assert set(found) == FULL_RESULT_KEYS
    assert set(missing) == FULL_RESULT_KEYS  # identical shape when nothing is found


# ======================================================================
# 6. Multiple documents coexist and stay distinguishable
# ======================================================================


def test_multiple_documents_are_retrieved_and_stay_distinguishable(real_2014_doc):
    synthetic_current = _doc(
        " ".join(["hypothetical widget governance framework clause"] * 15),
        doc_id="synthetic-current-direction",
        title="Hypothetical Widget Governance Master Direction",
        document_type="Master Direction",
        is_excerpt=False,
        is_current=True,
    )
    retriever = _retriever_over(real_2014_doc, synthetic_current)

    from_2014 = retriever.retrieve("non performing asset overdue ninety days")
    from_synthetic = retriever.retrieve("hypothetical widget governance framework clause")

    assert from_2014["doc_id"] == APPROVED_2014_ID
    assert from_2014["is_excerpt"] is True and from_2014["is_current"] is False

    assert from_synthetic["doc_id"] == "synthetic-current-direction"
    assert from_synthetic["is_excerpt"] is False and from_synthetic["is_current"] is True
    assert from_synthetic["title"] == "Hypothetical Widget Governance Master Direction"

    # a query about one document does not leak chunks from the other
    assert all(h["doc_id"] == APPROVED_2014_ID for h in from_2014["results"])
    assert all(h["doc_id"] == "synthetic-current-direction" for h in from_synthetic["results"])


# ======================================================================
# 7. top-k / configuration behaves as documented
# ======================================================================


def test_top_k_limits_the_number_of_results(default_retriever):
    one = default_retriever.retrieve("non performing asset overdue", top_k=1)
    many = default_retriever.retrieve("non performing asset overdue", top_k=5)
    assert len(one["results"]) == 1
    assert 1 <= len(many["results"]) <= 5
    # top-level fields always describe results[0]
    assert one["retrieved_text"] == one["results"][0]["text"]
    assert many["retrieved_text"] == many["results"][0]["text"]


def test_default_top_k_is_used_when_not_specified():
    assert DEFAULT_TOP_K == 3
    r1 = build_default_retriever(APPROVED_CORPUS, default_top_k=1)
    assert r1.default_top_k == 1
    assert len(r1(query="non performing asset overdue")["results"]) == 1

    r2 = build_default_retriever(APPROVED_CORPUS, default_top_k=2)
    assert len(r2.retrieve("non performing asset overdue account")["results"]) <= 2


def test_results_are_ordered_by_ascending_distance(default_retriever):
    result = default_retriever.retrieve("non performing asset overdue account", top_k=5)
    distances = [h["distance"] for h in result["results"]]
    assert distances == sorted(distances)


def test_invalid_top_k_is_rejected(default_retriever):
    with pytest.raises(ValueError):
        default_retriever.retrieve("non performing asset", top_k=0)
    with pytest.raises(ValueError):
        RBIRetriever(ChunkVectorStore.in_memory(), default_top_k=0)


def test_relevance_threshold_is_configurable():
    doc = load_by_id(APPROVED_2014_ID)

    strict_distance = _retriever_over(doc, max_distance=0.05)
    assert strict_distance.retrieve("non performing asset overdue")[
        "evidence_status"
    ] == NO_VERIFIED_EVIDENCE

    strict_overlap = _retriever_over(doc, min_lexical_overlap=6)
    assert strict_overlap.retrieve("non performing asset")[
        "evidence_status"
    ] == NO_VERIFIED_EVIDENCE

    permissive = _retriever_over(doc)  # defaults
    assert permissive.retrieve("non performing asset overdue")[
        "evidence_status"
    ] == EVIDENCE_RETRIEVED


# ======================================================================
# 8-10. Irrelevant / no-evidence behaviour is explicit and deterministic
# ======================================================================


def test_irrelevant_query_does_not_become_evidence(default_retriever):
    result = default_retriever.retrieve(
        "quantum chromodynamics photosynthesis basketball tournament schedule"
    )
    assert result["evidence_status"] == NO_VERIFIED_EVIDENCE
    assert result["retrieved_text"] == ""
    assert result["source"] is None
    assert result["chunk_index"] is None
    assert result["results"] == []


def test_stopword_only_query_is_not_evidence(default_retriever):
    # The offline embedding scores "the" as a close vector match; the
    # lexical-overlap gate must still reject it.
    result = default_retriever.retrieve("the of and to a")
    assert result["evidence_status"] == NO_VERIFIED_EVIDENCE
    assert result["reason"] == "no indexed chunk met the relevance threshold"


def test_empty_query_is_explicit_no_evidence(default_retriever):
    for blank in ("", "   ", "\n\t "):
        result = default_retriever.retrieve(blank)
        assert result["evidence_status"] == NO_VERIFIED_EVIDENCE
        assert result["reason"] == "query is empty"
        assert result["results"] == []


def test_empty_index_is_explicit_no_evidence():
    retriever = RBIRetriever(ChunkVectorStore.in_memory())
    result = retriever.retrieve("non performing asset")
    assert result["evidence_status"] == NO_VERIFIED_EVIDENCE
    assert result["reason"] == "the vector index is empty"


def test_no_evidence_result_carries_no_fabricated_attribution(default_retriever):
    result = default_retriever.retrieve("nothing here matches this query at all really")
    for key in (
        "source", "doc_id", "chunk_id", "chunk_index", "source_url", "title",
        "publication_date", "document_type", "is_excerpt", "is_current",
        "provenance", "distance",
    ):
        assert result[key] is None


def test_retrieval_is_deterministic(default_retriever):
    query = "non performing asset overdue account out of order"
    a = default_retriever.retrieve(query, top_k=3)
    b = default_retriever.retrieve(query, top_k=3)
    c = default_retriever.retrieve(query, top_k=3)
    assert a == b == c


def test_no_evidence_is_deterministic(default_retriever):
    a = default_retriever.retrieve("irrelevant gibberish tokens with no corpus overlap")
    b = default_retriever.retrieve("irrelevant gibberish tokens with no corpus overlap")
    assert a == b
    assert a["evidence_status"] == NO_VERIFIED_EVIDENCE


# ======================================================================
# 11. Historical/excerpt vs current provenance stays distinguishable
# ======================================================================


def test_retrieving_an_excerpt_does_not_mark_it_current(default_retriever):
    result = default_retriever.retrieve("non performing asset definition overdue")
    assert result["evidence_status"] == EVIDENCE_RETRIEVED
    # retrieved != verified current/binding
    assert result["is_excerpt"] is True
    assert result["is_current"] is False
    assert all(h["is_excerpt"] is True and h["is_current"] is False for h in result["results"])


def test_excerpt_and_current_flags_come_through_per_result(real_2014_doc):
    current_doc = _doc(
        " ".join(["distinct current regulatory obligation reporting requirement"] * 12),
        doc_id="current-direction",
        is_excerpt=False,
        is_current=True,
    )
    retriever = _retriever_over(real_2014_doc, current_doc)

    hist = retriever.retrieve("non performing asset overdue ninety days")
    curr = retriever.retrieve("distinct current regulatory obligation reporting requirement")

    assert (hist["is_excerpt"], hist["is_current"]) == (True, False)
    assert (curr["is_excerpt"], curr["is_current"]) == (False, True)


# ======================================================================
# 10 (contract). retrieval_fn(query=...) accepts the exact keyword
# ======================================================================


def test_retriever_is_callable_with_the_exact_query_keyword(default_retriever):
    result = default_retriever(query="non performing asset overdue ninety days")
    assert result["evidence_status"] == EVIDENCE_RETRIEVED
    assert REQUIRED_MIN_KEYS <= set(result)


def test_call_requires_the_query_keyword(default_retriever):
    with pytest.raises(TypeError):
        default_retriever("non performing asset")  # positional not allowed
    with pytest.raises(TypeError):
        default_retriever()  # query is required


def test_build_default_retriever_returns_a_working_retrieval_fn():
    fn = build_default_retriever(APPROVED_CORPUS)
    out = fn(query="what is a non performing asset")
    assert callable(fn)
    assert out["evidence_status"] in (EVIDENCE_RETRIEVED, NO_VERIFIED_EVIDENCE)
    assert "retrieved_text" in out and "source" in out and "chunk_index" in out


# ======================================================================
# Type checks + decoupling
# ======================================================================


def test_retriever_requires_a_vector_store():
    with pytest.raises(TypeError):
        RBIRetriever("not a store")


def test_retrieval_does_not_import_the_phase0_smoke_test():
    import ast
    import inspect

    import app.rag.retrieval as retrieval_module
    import app.rag.smoke_test  # noqa: F401 -- unchanged, must still import

    tree = ast.parse(inspect.getsource(retrieval_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("smoke_test" in name for name in imported), (
        f"retrieval must not import the Phase 0 smoke test; imports: {sorted(imported)}"
    )
