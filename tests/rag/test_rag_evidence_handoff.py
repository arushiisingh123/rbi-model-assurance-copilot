"""Phase 3A/3B, Task 7: the Nidhi-owned RAG -> evidence handoff contract.

Owner: Nidhi. These are INTEGRATION tests only -- no new production code
was needed. The handoff is the direct composition of two existing public
functions::

    retriever = build_default_retriever()      # Task 5 -- IS the retrieval_fn
    result    = retriever(query="...")          # Task 5 result dict
    evidence  = build_evidence(result)          # Task 6 -> list[RBIEvidence] (or [])

These tests prove that contract end to end -- corpus -> ingestion ->
chunking -> vector store -> retrieval -> evidence -> downstream
``retrieval_fn(query=...)`` -- is stable, deterministic, offline,
provenance-preserving, and never turns "no verified evidence" into a
fabricated citation.

Nothing here calls an LLM, generates a report, touches an API/dashboard,
or reads any RBI source other than the one approved local excerpt. The
Phase 0 smoke-test files are not modified.
"""
import ast
import collections
import inspect
import json
from pathlib import Path

import pytest

from app.rag.corpus import APPROVED_CORPUS, RBI_IRAC_ADVANCES_2014, get_source, list_sources
from app.rag.chunking import chunk_document
from app.rag.ingestion import load_by_id, load_corpus, provenance as source_provenance
from app.rag.vector_store import ChunkVectorStore
from app.rag.retrieval import (
    EVIDENCE_RETRIEVED,
    NO_VERIFIED_EVIDENCE,
    RBIRetriever,
    build_default_retriever,
)
from app.rag.evidence import (
    NO_EVIDENCE_MESSAGE,
    RBIEvidence,
    build_evidence,
    is_no_evidence,
    no_evidence_reason,
)

APPROVED_2014_ID = "rbi-master-circular-irac-advances-2014-07-01"

# A query with strong lexical overlap with the approved excerpt's NPA
# section -- deterministically clears the Task 5 relevance gate.
RELEVANT_QUERY = "non performing asset overdue account out of order status"
# No shared vocabulary with the corpus -> must not become evidence.
IRRELEVANT_QUERY = "quantum chromodynamics photosynthesis basketball tournament schedule"


@pytest.fixture(scope="module")
def default_retriever() -> RBIRetriever:
    return build_default_retriever()


@pytest.fixture(scope="module")
def handoff(default_retriever):
    """The end-to-end handoff output for a relevant query."""
    result = default_retriever(query=RELEVANT_QUERY)
    return result, build_evidence(result)


# ======================================================================
# 1. The approved real RBI corpus is loaded, chunked, indexed, queried,
#    attributed, and converted -- with every stage wired explicitly.
# ======================================================================


def test_full_pipeline_from_the_approved_corpus_to_evidence():
    # corpus -> ingestion
    document = load_by_id(APPROVED_2014_ID)
    assert document.doc_id == APPROVED_2014_ID
    assert document.text.strip()

    # ingestion -> chunking
    chunks = chunk_document(document)
    assert len(chunks) > 1

    # chunking -> vector store
    store = ChunkVectorStore.in_memory()
    assert store.add_chunks(chunks) == len(chunks)

    # vector store -> retrieval
    retriever = RBIRetriever(store)
    result = retriever.retrieve(RELEVANT_QUERY, top_k=3)
    assert result["evidence_status"] == EVIDENCE_RETRIEVED

    # retrieval -> evidence
    evidence = build_evidence(result)
    assert evidence and all(isinstance(e, RBIEvidence) for e in evidence)
    assert all(e.doc_id == APPROVED_2014_ID for e in evidence)


def test_build_default_retriever_is_the_same_pipeline_in_one_call(handoff):
    result, evidence = handoff
    assert result["evidence_status"] == EVIDENCE_RETRIEVED
    assert evidence
    assert [e.chunk_id for e in evidence] == [h["chunk_id"] for h in result["results"]]


# ======================================================================
# 2-3. Successful retrieval -> EVIDENCE_RETRIEVED -> build_evidence()
# ======================================================================


def test_successful_retrieval_status_and_conversion(handoff):
    result, evidence = handoff
    assert result["evidence_status"] == EVIDENCE_RETRIEVED
    assert result["reason"] is None
    assert len(evidence) == len(result["results"]) >= 1


# ======================================================================
# 4. Exact retrieved text survives -- no paraphrase, no summary, no LLM
# ======================================================================


def test_exact_retrieved_text_survives_end_to_end(handoff):
    result, evidence = handoff
    for record, hit in zip(evidence, result["results"]):
        assert record.text == hit["text"]

    ingested_words = load_corpus()[0].text.split()
    for record in evidence:
        # every word of the evidence text is a word from the ingested source
        assert collections.Counter(record.text.split()) <= collections.Counter(ingested_words)


# ======================================================================
# 5-14. Every attribution/provenance field survives corpus -> evidence
# ======================================================================


def test_attribution_fields_match_the_approved_corpus_metadata(handoff):
    _result, evidence = handoff
    source_meta = get_source(APPROVED_2014_ID)
    top = evidence[0]

    assert top.doc_id == source_meta.doc_id                      # 5
    assert top.chunk_id.startswith(source_meta.doc_id + "::chunk-")  # 6
    assert isinstance(top.chunk_index, int) and top.chunk_index >= 0  # 7
    assert top.title == source_meta.title                        # 8
    assert top.source == source_meta.title                       # 8 (source mirrors title)
    assert top.source_url == source_meta.source_url              # 9
    assert top.source_url.startswith("https://www.rbi.org.in/")
    assert top.publication_date == source_meta.publication_date  # 10
    assert top.publication_date == "2014-07-01"
    assert top.document_type == source_meta.document_type        # 11
    assert top.document_type == "Master Circular"
    assert top.is_excerpt == source_meta.is_excerpt is True      # 12
    assert top.is_current == source_meta.is_current is False     # 13


def test_provenance_survives_every_stage_without_loss(handoff):
    _result, evidence = handoff
    source_meta = get_source(APPROVED_2014_ID)
    expected_source_fields = source_provenance(source_meta)  # the 14 provenance fields

    for record in evidence:
        prov = record.provenance                              # 14
        # every provenance field that the source actually stated is present
        for name, value in expected_source_fields.items():
            if value is None:
                assert name not in prov, f"{name!r} was None on the source; must not appear"
            else:
                assert prov[name] == value, f"{name!r} changed between corpus and evidence"
        # chunk coordinates were added by chunking and carried through
        assert prov["chunk_id"] == record.chunk_id
        assert prov["chunk_index"] == record.chunk_index
        # nothing was invented
        assert "effective_date" not in prov


def test_traceability_ids_are_internally_consistent(handoff):
    _result, evidence = handoff
    for record in evidence:
        assert record.chunk_id == f"{record.doc_id}::chunk-{record.chunk_index}"
        assert record.provenance["doc_id"] == record.doc_id


# ======================================================================
# 8 / 15. Multiple evidence records preserve ranking + independent attribution
# ======================================================================


def test_multiple_results_preserve_ranking_and_independent_provenance(default_retriever):
    result = default_retriever.retrieve(RELEVANT_QUERY, top_k=3)
    evidence = build_evidence(result)
    assert len(evidence) >= 2

    # order preserved
    assert [e.chunk_id for e in evidence] == [h["chunk_id"] for h in result["results"]]
    # each record carries its own attribution + its own provenance object
    for record, hit in zip(evidence, result["results"]):
        assert record.chunk_index == hit["chunk_index"]
        assert record.provenance == hit["provenance"]
    assert len({id(e.provenance) for e in evidence}) == len(evidence)


# ======================================================================
# 16-18. No-evidence behaviour survives the whole pipeline
# ======================================================================


def test_no_evidence_query_produces_no_verified_evidence(default_retriever):
    result = default_retriever(query=IRRELEVANT_QUERY)
    assert result["evidence_status"] == NO_VERIFIED_EVIDENCE


def test_no_evidence_produces_zero_fabricated_evidence(default_retriever):
    result = default_retriever(query=IRRELEVANT_QUERY)
    evidence = build_evidence(result)
    assert evidence == []
    # no placeholder attribution leaked into the result either
    for key in ("retrieved_text", "source", "chunk_index", "doc_id", "chunk_id",
                "source_url", "title", "publication_date", "document_type",
                "is_excerpt", "is_current", "provenance"):
        assert result[key] in ("", None)


def test_no_evidence_reason_is_preserved_and_means_not_retrieved(default_retriever):
    result = default_retriever(query="")
    assert is_no_evidence(result)
    assert no_evidence_reason(result) == "query is empty"

    threshold_miss = default_retriever(query=IRRELEVANT_QUERY)
    reason = no_evidence_reason(threshold_miss)
    assert reason == "no indexed chunk met the relevance threshold"
    # the meaning is "nothing retrieved", never "no RBI rule exists"
    assert "rule" not in reason.lower()
    assert "No verified RBI evidence" in NO_EVIDENCE_MESSAGE


# ======================================================================
# 4 (again) / historical-excerpt distinction survives end to end
# ======================================================================


def test_historical_excerpt_status_is_never_upgraded(handoff):
    _result, evidence = handoff
    for record in evidence:
        assert record.is_excerpt is True
        assert record.is_current is False
        assert record.provenance["is_excerpt"] is True
        assert record.provenance["is_current"] is False
        # serialised form keeps the same flags
        data = record.to_dict()
        assert data["is_excerpt"] is True and data["is_current"] is False

    # the source itself is not verified current regulation
    assert get_source(APPROVED_2014_ID).is_verified_current_regulation is False


def test_scope_caveat_from_the_source_is_carried_through_unchanged(handoff):
    _result, evidence = handoff
    scope_note = evidence[0].provenance["scope_note"]
    # this is the source file's own wording, not an invented claim
    assert "must not be treated as current or binding" in scope_note.lower()


# ======================================================================
# 19. Determinism
# ======================================================================


def test_handoff_is_deterministic():
    def run():
        retriever = build_default_retriever()
        result = retriever.retrieve(RELEVANT_QUERY, top_k=3)
        return [e.to_dict() for e in build_evidence(result)]

    a, b, c = run(), run(), run()
    assert a == b == c
    assert a  # and it actually produced evidence


# ======================================================================
# 20. Offline
# ======================================================================


def test_whole_handoff_runs_without_network(monkeypatch):
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("unexpected network connect() during RAG handoff")

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=True)

    retriever = build_default_retriever()
    evidence = build_evidence(retriever(query=RELEVANT_QUERY))
    assert evidence and evidence[0].doc_id == APPROVED_2014_ID
    assert build_evidence(retriever(query=IRRELEVANT_QUERY)) == []


# ======================================================================
# 21. Downstream retrieval_fn(query=...) compatibility
# ======================================================================


def test_retriever_satisfies_the_retrieval_fn_signature(default_retriever):
    # exact keyword, no positional
    out = default_retriever(query=RELEVANT_QUERY)
    assert isinstance(out, dict)
    # the keys app/report/generate.py reads off the retrieval_fn result
    assert "retrieved_text" in out and "source" in out and "chunk_index" in out
    assert isinstance(out["retrieved_text"], str)

    empty = default_retriever(query=IRRELEVANT_QUERY)
    assert empty["retrieved_text"] == "" and empty["source"] is None


def test_drops_into_the_existing_report_retrieval_slot_unmodified(default_retriever):
    # Import the downstream consumer READ-ONLY to prove real compatibility.
    from app.report.generate import _retrieve_section_evidence

    got = _retrieve_section_evidence(
        "compliance", "what is a non performing asset overdue", retrieval_fn=default_retriever
    )
    assert got.evidence_status == "RETRIEVED"
    assert got.citations and got.citations[0].quote.strip()
    # the citation's source string is built from OUR `source` (the real title)
    assert "Master Circular" in got.citations[0].source

    missing = _retrieve_section_evidence(
        "compliance", IRRELEVANT_QUERY, retrieval_fn=default_retriever
    )
    assert missing.evidence_status == "NOT_FOUND"
    assert missing.citations == []


# ======================================================================
# 22-24. No LLM / no report generation / no external source
# ======================================================================


def _module_imports(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def test_rag_layer_imports_no_llm_or_report_module():
    import app.rag.retrieval as retrieval_module
    import app.rag.evidence as evidence_module

    for module in (retrieval_module, evidence_module):
        imports = _module_imports(module)
        assert not any(
            bad in name.lower()
            for name in imports
            for bad in ("groq", "openai", "llm", "app.report", "app.api", "dashboard")
        ), f"{module.__name__} imports something out of scope: {sorted(imports)}"


def test_handoff_never_triggers_report_generation(monkeypatch, default_retriever):
    import app.report.generate as generate_module

    def _boom(*args, **kwargs):
        raise AssertionError("the RAG handoff must not call the report/LLM layer")

    monkeypatch.setattr(generate_module, "generate_report", _boom)
    monkeypatch.setattr(generate_module, "_call_groq_llm", _boom)

    result = default_retriever(query=RELEVANT_QUERY)
    evidence = build_evidence(result)
    assert evidence  # produced entirely without the report layer


def test_only_the_one_approved_local_source_is_used():
    sources = list_sources()
    assert [s.doc_id for s in sources] == [APPROVED_2014_ID]
    assert list(APPROVED_CORPUS.doc_ids()) == [APPROVED_2014_ID]

    # The directory also holds rbi_demo_requirements.txt: a VERIFICATION
    # BACKLOG of requirement intents awaiting clause-level verification. It is
    # not RBI text, not a corpus document, and deliberately not registered.
    #
    # It is named here rather than filtered by a wildcard, so that any OTHER
    # file appearing in this directory still fails this test. The check that
    # actually matters is the one below it: being present on disk must not
    # make a file a retrievable source.
    BACKLOG = "rbi_demo_requirements.txt"
    files = sorted(p.name for p in Path("data/rbi_sources").glob("*") if p.is_file())
    assert files in (
        ["RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt"],
        sorted(["RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt", BACKLOG]),
    ), files

    # The corpus is an explicit registry, never a directory scan. Dropping a
    # file into this folder must not enrol it as evidence.
    registered = {s.doc_id for s in list_sources()} | set(APPROVED_CORPUS.doc_ids())
    assert not any(BACKLOG.rsplit(".", 1)[0] in doc_id for doc_id in registered)

    assert RBI_IRAC_ADVANCES_2014.resolved_path().is_file()


# ======================================================================
# 25. Phase 0 smoke test is untouched and still works
# ======================================================================


def test_phase0_smoke_test_module_is_unchanged_and_importable():
    from app.rag import smoke_test

    # its Phase 0 API still exists exactly as before
    assert hasattr(smoke_test, "run_smoke_test")
    assert hasattr(smoke_test, "chunk_text")
    assert smoke_test.DOCUMENT_PATH.name == "RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt"
    # the handoff layer does not import it
    assert not any("smoke_test" in name for name in _module_imports(__import__(
        "app.rag.retrieval", fromlist=["x"])))


# ======================================================================
# Serialisation carries enough for downstream citation/attribution
# ======================================================================


def test_evidence_serialisation_is_downstream_ready(handoff):
    _result, evidence = handoff
    payload = [e.to_dict() for e in evidence]
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped == payload
    for item in round_tripped:
        for key in ("text", "doc_id", "chunk_id", "chunk_index", "source",
                    "source_url", "title", "publication_date", "document_type",
                    "is_excerpt", "is_current", "provenance"):
            assert key in item
