"""Phase 5D: RAG evidence is model-agnostic and does not leak state
across two different models' report-generation contexts.

This does NOT add model identity to RAG (see docs/decisions.md --
RAG evidence is regulatory text, not model output, and correctly
carries no model_id). It proves the model-agnostic design is SAFE
under the actual Phase 5D scenario: two models' reports generated
in the same process, one after another.
"""

from app.rag.evidence import build_evidence
from app.rag.retrieval import (
    EVIDENCE_RETRIEVED,
    NO_VERIFIED_EVIDENCE,
    build_default_retriever,
)

RELEVANT_QUERY = "non performing asset overdue account out of order status"
IRRELEVANT_QUERY = (
    "quantum chromodynamics photosynthesis basketball tournament schedule"
)


def test_same_query_returns_identical_evidence_regardless_of_model_context():
    """Two independently-built retrievers (standing in for two separate
    model report-generation runs) must retrieve byte-identical RBI
    evidence for the same query -- proving there is no hidden
    per-model state anywhere in the retrieval path."""
    retriever_for_model_a = build_default_retriever()
    retriever_for_model_b = build_default_retriever()
    result_a = retriever_for_model_a(query=RELEVANT_QUERY)
    result_b = retriever_for_model_b(query=RELEVANT_QUERY)
    evidence_a = [e.to_dict() for e in build_evidence(result_a)]
    evidence_b = [e.to_dict() for e in build_evidence(result_b)]
    assert evidence_a == evidence_b


def test_two_sequential_retrievers_do_not_share_or_leak_state():
    """Building a second retriever must not be affected by, or affect,
    a first one already in use -- guards against a future shared
    cache/singleton regression."""
    first = build_default_retriever()
    first_hit = first(query=RELEVANT_QUERY)
    assert first_hit["evidence_status"] == EVIDENCE_RETRIEVED
    second = build_default_retriever()
    second_miss = second(query=IRRELEVANT_QUERY)
    assert second_miss["evidence_status"] == NO_VERIFIED_EVIDENCE
    # the first retriever's own state is unaffected by the second's existence
    repeat_first_hit = first(query=RELEVANT_QUERY)
    assert repeat_first_hit == first_hit


def test_no_evidence_stays_explicit_even_when_interleaved_with_a_hit():
    """NO_VERIFIED_EVIDENCE for one 'model's' query must not be
    upgraded or fabricated just because a different query in the same
    process succeeded moments before."""
    retriever = build_default_retriever()
    hit = retriever(query=RELEVANT_QUERY)
    miss = retriever(query=IRRELEVANT_QUERY)
    assert hit["evidence_status"] == EVIDENCE_RETRIEVED
    assert miss["evidence_status"] == NO_VERIFIED_EVIDENCE
    assert build_evidence(miss) == []
    assert miss["reason"] and "rule" not in miss["reason"].lower()
