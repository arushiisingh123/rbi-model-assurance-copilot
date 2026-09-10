"""Phase 3A, Task 6 tests: RBI evidence records + source attribution.

Owner: Nidhi. Covers the evidence/attribution layer only -- it reshapes a
Task 5 retrieval result, it does not retrieve, interpret, or generate a
report. Everything runs on the deterministic offline retriever, so tests
need no network. The Phase 0 smoke-test files are left untouched.
"""
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from app.rag.chunking import chunk_documents
from app.rag.corpus import RBISourceMetadata
from app.rag.ingestion import LoadedDocument
from app.rag.retrieval import RBIRetriever, build_default_retriever
from app.rag.vector_store import ChunkVectorStore
from app.rag.evidence import (
    EVIDENCE_RETRIEVED,
    NO_EVIDENCE_MESSAGE,
    NO_VERIFIED_EVIDENCE,
    EvidenceConsistencyError,
    EvidenceError,
    RBIEvidence,
    build_evidence,
    is_no_evidence,
    no_evidence_reason,
)

APPROVED_2014_ID = "rbi-master-circular-irac-advances-2014-07-01"

_ALL_FIELDS = (
    "doc_id", "chunk_id", "chunk_index", "source", "source_url", "title",
    "publication_date", "document_type", "is_excerpt", "is_current",
)


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


def _retriever_over(*docs: LoadedDocument, **kw) -> RBIRetriever:
    store = ChunkVectorStore.in_memory()
    store.add_chunks(chunk_documents(list(docs)))
    return RBIRetriever(store, **kw)


@pytest.fixture(scope="module")
def real_retriever() -> RBIRetriever:
    return build_default_retriever()


@pytest.fixture(scope="module")
def real_result(real_retriever) -> dict:
    return real_retriever.retrieve("non performing asset overdue account out of order", top_k=3)


# ======================================================================
# 1-13. Successful retrieval -> evidence records, every field preserved
# ======================================================================


def test_successful_retrieval_converts_to_evidence(real_result):
    evidence = build_evidence(real_result)
    assert real_result["evidence_status"] == EVIDENCE_RETRIEVED
    assert len(evidence) == len(real_result["results"])
    assert all(isinstance(e, RBIEvidence) for e in evidence)


def test_every_field_is_copied_from_the_retrieval_result(real_result):
    evidence = build_evidence(real_result)
    for record, hit in zip(evidence, real_result["results"]):
        # 2. exact text -- not paraphrased
        assert record.text == hit["text"]
        # 3-11. attribution fields
        assert record.doc_id == hit["doc_id"]
        assert record.chunk_id == hit["chunk_id"]
        assert record.chunk_index == hit["chunk_index"]
        assert record.source == hit["title"]        # Task 5 convention
        assert record.title == hit["title"]
        assert record.source_url == hit["source_url"]
        assert record.publication_date == hit["publication_date"]
        assert record.document_type == hit["document_type"]
        assert record.is_excerpt == hit["is_excerpt"]
        assert record.is_current == hit["is_current"]
        # 12. provenance preserved verbatim
        assert record.provenance == hit["provenance"]


def test_real_2014_attribution_values_are_the_corpus_values(real_result):
    top = build_evidence(real_result)[0]
    assert top.doc_id == APPROVED_2014_ID
    assert top.title.startswith("Master Circular - Prudential Norms")
    assert top.source == top.title
    assert top.source_url.startswith("https://www.rbi.org.in/")
    assert top.document_type == "Master Circular"
    assert top.publication_date == "2014-07-01"
    assert top.chunk_id.startswith(APPROVED_2014_ID + "::chunk-")
    assert isinstance(top.chunk_index, int)


def test_provenance_carries_the_stored_chunk_metadata(real_result):
    top = build_evidence(real_result)[0]
    assert top.provenance["doc_id"] == top.doc_id
    assert top.provenance["chunk_id"] == top.chunk_id
    assert top.provenance["chunk_index"] == top.chunk_index
    # 2014 source states no separate effective date -> never stored, never invented
    assert "effective_date" not in top.provenance


def test_multiple_results_preserve_their_ranking_order(real_result):
    evidence = build_evidence(real_result)
    assert len(evidence) >= 2
    assert [e.chunk_id for e in evidence] == [h["chunk_id"] for h in real_result["results"]]
    assert [e.chunk_index for e in evidence] == [h["chunk_index"] for h in real_result["results"]]


def test_retrieved_text_is_unchanged_word_for_word(real_result):
    for record, hit in zip(build_evidence(real_result), real_result["results"]):
        assert record.text == hit["text"]
        assert record.text.split() == hit["text"].split()  # nothing trimmed/rewrapped


# ======================================================================
# 14. NO_VERIFIED_EVIDENCE -> no fabricated evidence
# ======================================================================


def test_no_verified_evidence_yields_an_empty_list(real_retriever):
    result = real_retriever.retrieve("quantum chromodynamics basketball tournament schedule")
    assert result["evidence_status"] == NO_VERIFIED_EVIDENCE
    assert build_evidence(result) == []


def test_empty_query_no_evidence_is_also_empty(real_retriever):
    assert build_evidence(real_retriever.retrieve("   ")) == []


def test_no_evidence_reason_is_preserved_not_invented(real_retriever):
    result = real_retriever.retrieve("")
    assert is_no_evidence(result) is True
    assert no_evidence_reason(result) == result["reason"] == "query is empty"

    found = real_retriever.retrieve("non performing asset overdue")
    assert is_no_evidence(found) is False
    assert no_evidence_reason(found) is None


def test_no_evidence_message_constant_is_available():
    assert "No verified RBI evidence" in NO_EVIDENCE_MESSAGE
    # falls back to the standard message only when retrieval gave no reason
    assert no_evidence_reason({"evidence_status": NO_VERIFIED_EVIDENCE}) == NO_EVIDENCE_MESSAGE


# ======================================================================
# 15. Missing optional metadata stays missing
# ======================================================================


def test_missing_optional_metadata_is_preserved_as_none():
    sparse = _doc(
        " ".join(["widget governance framework clause obligation"] * 12),
        doc_id="sparse-synthetic-doc",
        # title + document_type are required by the schema; leave the rest unset
    )
    retriever = _retriever_over(sparse)
    result = retriever.retrieve("widget governance framework clause obligation")
    record = build_evidence(result)[0]

    assert record.source_url is None
    assert record.publication_date is None
    assert record.doc_id == "sparse-synthetic-doc"
    # unset fields are simply absent from provenance -- not blanked, not faked
    assert "source_url" not in record.provenance
    assert "publication_date" not in record.provenance
    assert "reference_number" not in record.provenance
    assert "effective_date" not in record.provenance


# ======================================================================
# 16. Historical/excerpt vs current stays distinguishable
# ======================================================================


def test_real_2014_evidence_stays_a_historical_excerpt(real_result):
    for record in build_evidence(real_result):
        assert record.is_excerpt is True
        assert record.is_current is False
        assert record.provenance["is_excerpt"] is True
        assert record.provenance["is_current"] is False


def test_excerpt_and_current_evidence_are_distinguishable():
    historical = _doc(
        " ".join(["alpha historical excerpt provision clause"] * 12),
        doc_id="hist-doc", is_excerpt=True, is_current=False,
        coverage_note="only part one",
    )
    current = _doc(
        " ".join(["bravo current binding direction requirement"] * 12),
        doc_id="curr-doc", is_excerpt=False, is_current=True,
    )
    retriever = _retriever_over(historical, current)

    hist_ev = build_evidence(retriever.retrieve("alpha historical excerpt provision clause"))[0]
    curr_ev = build_evidence(retriever.retrieve("bravo current binding direction requirement"))[0]

    assert (hist_ev.doc_id, hist_ev.is_excerpt, hist_ev.is_current) == ("hist-doc", True, False)
    assert (curr_ev.doc_id, curr_ev.is_excerpt, curr_ev.is_current) == ("curr-doc", False, True)


# ======================================================================
# 17. Serialization preserves attribution
# ======================================================================


def test_to_dict_preserves_all_attribution_fields(real_result):
    record = build_evidence(real_result)[0]
    data = record.to_dict()

    assert data["text"] == record.text
    for field in _ALL_FIELDS:
        assert data[field] == getattr(record, field)
    assert data["provenance"] == record.provenance
    # downstream report generation serialises this
    assert json.loads(json.dumps(data))["chunk_id"] == record.chunk_id


def test_to_dict_provenance_is_a_copy(real_result):
    record = build_evidence(real_result)[0]
    data = record.to_dict()
    data["provenance"]["doc_id"] = "MUTATED"
    assert record.provenance["doc_id"] == APPROVED_2014_ID  # record untouched


# ======================================================================
# 18. Inconsistent attribution fails clearly (no silent correction)
# ======================================================================


def _one_hit_result(**hit_overrides) -> dict:
    prov = {
        "doc_id": "doc-a", "chunk_id": "doc-a::chunk-0", "chunk_index": 0,
        "title": "Doc A", "document_type": "Master Direction",
        "is_excerpt": True, "is_current": False,
    }
    hit = {
        "text": "some retrieved regulatory text",
        "doc_id": "doc-a", "chunk_id": "doc-a::chunk-0", "chunk_index": 0,
        "source_url": None, "title": "Doc A", "publication_date": None,
        "document_type": "Master Direction", "is_excerpt": True, "is_current": False,
        "distance": 0.5, "provenance": prov,
    }
    hit.update(hit_overrides)
    return {"evidence_status": EVIDENCE_RETRIEVED, "results": [hit]}


def test_doc_id_disagreeing_with_provenance_fails_clearly():
    bad = _one_hit_result(doc_id="doc-WRONG")
    with pytest.raises(EvidenceConsistencyError, match="doc_id"):
        build_evidence(bad)


def test_chunk_index_disagreeing_with_provenance_fails_clearly():
    bad = _one_hit_result(chunk_index=7)
    with pytest.raises(EvidenceConsistencyError, match="chunk_index"):
        build_evidence(bad)


def test_is_current_disagreeing_with_provenance_fails_clearly():
    # provenance says historical excerpt; the hit claims it is current
    bad = _one_hit_result(is_current=True)
    with pytest.raises(EvidenceConsistencyError, match="is_current"):
        build_evidence(bad)


def test_consistent_hand_built_result_is_accepted():
    evidence = build_evidence(_one_hit_result())
    assert len(evidence) == 1
    assert evidence[0].doc_id == "doc-a"
    assert evidence[0].is_excerpt is True and evidence[0].is_current is False


# ======================================================================
# Input validation
# ======================================================================


def test_non_dict_input_is_rejected():
    with pytest.raises(EvidenceError):
        build_evidence("not a result")
    with pytest.raises(EvidenceError):
        build_evidence(None)


def test_result_without_evidence_status_is_rejected():
    with pytest.raises(EvidenceError):
        build_evidence({"results": []})


def test_unknown_evidence_status_is_rejected():
    with pytest.raises(EvidenceError, match="unrecognised evidence_status"):
        build_evidence({"evidence_status": "MAYBE", "results": []})


def test_evidence_retrieved_without_results_is_rejected():
    with pytest.raises(EvidenceError, match="non-empty 'results'"):
        build_evidence({"evidence_status": EVIDENCE_RETRIEVED, "results": []})


def test_result_entry_missing_a_required_key_is_rejected():
    broken = {"evidence_status": EVIDENCE_RETRIEVED, "results": [{"text": "x"}]}
    with pytest.raises(EvidenceError, match="missing required key"):
        build_evidence(broken)


# ======================================================================
# Record is immutable / factual only
# ======================================================================


def test_evidence_record_is_frozen(real_result):
    record = build_evidence(real_result)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.text = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.is_current = True


def test_record_has_no_interpretation_fields(real_result):
    record = build_evidence(real_result)[0]
    field_names = {f.name for f in dataclasses.fields(record)}
    for banned in ("interpretation", "summary", "explanation", "conclusion",
                   "recommendation", "compliance_status", "verdict", "citation"):
        assert banned not in field_names


def test_construction_copies_the_caller_provenance_dict():
    prov = {
        "doc_id": "d", "chunk_id": "d::chunk-0", "chunk_index": 0,
        "title": "T", "is_excerpt": True, "is_current": False,
    }
    record = RBIEvidence(
        text="t", doc_id="d", chunk_id="d::chunk-0", chunk_index=0,
        source="T", source_url=None, title="T", publication_date=None,
        document_type=None, is_excerpt=True, is_current=False, provenance=prov,
    )
    prov["doc_id"] = "MUTATED"
    assert record.provenance["doc_id"] == "d"


# ======================================================================
# 19. Offline / decoupling
# ======================================================================


def test_full_flow_works_without_network(monkeypatch):
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("unexpected network connect() during evidence build")

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=True)

    retriever = _retriever_over(
        _doc(" ".join(["offline evidence attribution provenance record"] * 10), doc_id="offline-doc")
    )
    evidence = build_evidence(retriever.retrieve("offline evidence attribution provenance record"))
    assert evidence and evidence[0].doc_id == "offline-doc"


def test_evidence_does_not_import_the_phase0_smoke_test():
    import ast
    import inspect

    import app.rag.evidence as evidence_module
    import app.rag.smoke_test  # noqa: F401 -- unchanged, must still import

    tree = ast.parse(inspect.getsource(evidence_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("smoke_test" in name for name in imported), (
        f"evidence must not import the Phase 0 smoke test; imports: {sorted(imported)}"
    )


def test_build_evidence_does_not_retrieve(monkeypatch, real_result):
    # Passing an already-produced result must never trigger a new search.
    import app.rag.retrieval as retrieval_module

    def _boom(*args, **kwargs):
        raise AssertionError("build_evidence must not run retrieval")

    monkeypatch.setattr(retrieval_module.RBIRetriever, "retrieve", _boom)
    monkeypatch.setattr(retrieval_module.RBIRetriever, "__call__", _boom)

    frozen_copy = copy.deepcopy(real_result)
    evidence = build_evidence(frozen_copy)
    assert len(evidence) == len(real_result["results"])
