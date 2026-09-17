"""Phase 5D orchestration wiring (owner: Nidhi + Khushi's follow-up).

Two related, additive changes to app/api/orchestration.py, tested together
because both are exercised through the same live build_assurance_result()
path:

1. RAG -> compliance evidence wiring: build_evidence_by_rule() retrieves
   real RBI evidence per rule (app/rag/) and threads it through
   compute_real_compliance() into evaluate_compliance()'s existing
   evidence_by_rule parameter (app/compliance/compliance.py -- unchanged).
2. Adapter-aware model metrics: compute_real_model_metrics() and
   build_assurance_result() accept an optional adapter so predictions,
   metrics, and identity all describe the same model.

Deliberately does not exercise app/rag/retrieval.py's real corpus scoring
for the isolation/ordering assertions below -- those use a hand-built
Task 5-shaped retrieval double (matching tests/rag/test_evidence_records.py
and tests/compliance/test_compliance_evidence_chunks.py's own pattern), so
these tests do not depend on which RBI documents happen to be indexed.
"""
from typing import Any, Dict, List

from app.api.orchestration import (
    build_assurance_result,
    build_evidence_by_rule,
    compute_real_compliance,
    compute_real_drift,
    compute_real_explainability,
    compute_real_fairness,
    compute_real_model,
    compute_real_model_metrics,
)
from app.compliance.compliance import evaluate_compliance
from app.compliance.mock_findings import MOCK_TECHNICAL_FINDINGS
from app.models.model import MODEL_ID, MODEL_VERSION, RandomForestAdapter, evaluate_current_model
from app.rag.retrieval import EVIDENCE_RETRIEVED, NO_VERIFIED_EVIDENCE
from app.rbi.rules import load_rules


# ---------------------------------------------------------------------------
# Test doubles: Task 5-shaped retrieval results (see app/rag/retrieval.py's
# RBIRetriever._evidence / ._no_evidence for the exact shape build_evidence()
# expects).
# ---------------------------------------------------------------------------

_ATTRIBUTION = {
    "source_url": "https://rbi.example/circular",
    "title": "Some RBI Title",
    "publication_date": "2014-07-01",
    "document_type": "Master Circular",
    "is_excerpt": True,
    "is_current": False,
}


def _hit(*, chunk_id: str, doc_id: str, chunk_index: int, text: str) -> Dict[str, Any]:
    hit = {"text": text, "doc_id": doc_id, "chunk_id": chunk_id, "chunk_index": chunk_index, "provenance": {}}
    hit.update(_ATTRIBUTION)
    return hit


def _evidence_result(query: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    top = hits[0]
    result = {
        "query": query,
        "evidence_status": EVIDENCE_RETRIEVED,
        "reason": None,
        "retrieved_text": top["text"],
        "source": top["title"],
        "distance": 0.1,
        "provenance": top["provenance"],
        "results": hits,
    }
    result.update(_ATTRIBUTION)
    result["doc_id"] = top["doc_id"]
    result["chunk_id"] = top["chunk_id"]
    result["chunk_index"] = top["chunk_index"]
    return result


def _no_evidence_result(query: str) -> Dict[str, Any]:
    result = {
        "query": query,
        "evidence_status": NO_VERIFIED_EVIDENCE,
        "reason": "no indexed chunk met the relevance threshold",
        "retrieved_text": "",
        "source": None,
        "distance": None,
        "provenance": None,
        "results": [],
    }
    for key in _ATTRIBUTION:
        result[key] = None
    result["doc_id"] = None
    result["chunk_id"] = None
    result["chunk_index"] = None
    return result


def _category_routed_retrieval_fn(*, category_hits: Dict[str, List[Dict[str, Any]]]):
    """A retrieval_fn(query=...) double that routes by which SECTION_QUERIES
    text the query matches, so distinct categories get distinct evidence and
    an omitted category gets NO_VERIFIED_EVIDENCE."""
    from app.report.generate import SECTION_QUERIES

    query_to_category = {v: k for k, v in SECTION_QUERIES.items()}

    def _fn(*, query: str) -> Dict[str, Any]:
        category = query_to_category.get(query)
        hits = category_hits.get(category)
        if not hits:
            return _no_evidence_result(query)
        return _evidence_result(query, hits)

    return _fn


# ---------------------------------------------------------------------------
# build_evidence_by_rule(): isolation, ordering, and no-evidence handling
# ---------------------------------------------------------------------------


def test_build_evidence_by_rule_isolates_evidence_by_category():
    """Two different categories' evidence must never cross into each
    other's rules (RBI-FAIR-* must never see RBI-DRIFT-*'s chunk, and
    vice versa)."""
    retrieval_fn = _category_routed_retrieval_fn(
        category_hits={
            "fairness": [_hit(chunk_id="chunk-fair", doc_id="doc-fair", chunk_index=0, text="fairness excerpt")],
            "drift": [_hit(chunk_id="chunk-drift", doc_id="doc-drift", chunk_index=0, text="drift excerpt")],
        }
    )
    evidence_by_rule = build_evidence_by_rule(retrieval_fn=retrieval_fn)

    assert {rule["rule_id"] for rule in load_rules()} == set(evidence_by_rule)

    fairness_rule_ids = [r["rule_id"] for r in load_rules() if r["category"] == "fairness"]
    drift_rule_ids = [r["rule_id"] for r in load_rules() if r["category"] == "drift"]
    other_rule_ids = [r["rule_id"] for r in load_rules() if r["category"] not in ("fairness", "drift")]

    for rid in fairness_rule_ids:
        assert [e.chunk_id for e in evidence_by_rule[rid]] == ["chunk-fair"]
    for rid in drift_rule_ids:
        assert [e.chunk_id for e in evidence_by_rule[rid]] == ["chunk-drift"]
    # No leakage: a rule outside a category with hits gets no evidence.
    for rid in other_rule_ids:
        assert evidence_by_rule[rid] == []


def test_build_evidence_by_rule_preserves_retrieval_rank_order():
    """Multiple hits for one rule's query stay in the caller's ranked
    order -- build_evidence() preserves rank order, and this must not be
    reordered or deduplicated on the way into the per-rule dict."""
    hits = [
        _hit(chunk_id="chunk-a", doc_id="doc-1", chunk_index=0, text="first"),
        _hit(chunk_id="chunk-b", doc_id="doc-1", chunk_index=1, text="second"),
        _hit(chunk_id="chunk-c", doc_id="doc-2", chunk_index=0, text="third"),
    ]
    retrieval_fn = _category_routed_retrieval_fn(category_hits={"model": hits})
    evidence_by_rule = build_evidence_by_rule(retrieval_fn=retrieval_fn)

    model_rule_ids = [r["rule_id"] for r in load_rules() if r["category"] == "model"]
    assert model_rule_ids, "expected at least one 'model' category rule"
    for rid in model_rule_ids:
        assert [e.chunk_id for e in evidence_by_rule[rid]] == ["chunk-a", "chunk-b", "chunk-c"]


def test_build_evidence_by_rule_no_verified_evidence_yields_empty_lists():
    """A retrieval double that never finds evidence must yield [] for
    every rule -- never a fabricated chunk."""
    retrieval_fn = _category_routed_retrieval_fn(category_hits={})
    evidence_by_rule = build_evidence_by_rule(retrieval_fn=retrieval_fn)

    assert evidence_by_rule
    for rule_id, records in evidence_by_rule.items():
        assert records == []


def test_build_evidence_by_rule_builds_real_default_retriever_when_omitted():
    """Omitting retrieval_fn builds a real app.rag retriever over the
    approved corpus (no crash, valid dict shape) rather than requiring a
    caller to always inject one."""
    evidence_by_rule = build_evidence_by_rule()
    assert set(evidence_by_rule) == {rule["rule_id"] for rule in load_rules()}
    for records in evidence_by_rule.values():
        assert isinstance(records, list)


# ---------------------------------------------------------------------------
# compute_real_compliance(): forwards evidence_by_rule; unaffected when absent
# ---------------------------------------------------------------------------


def test_compute_real_compliance_forwards_evidence_by_rule():
    raw_model = compute_real_model()
    raw_explain = compute_real_explainability(raw_model)
    raw_fairness = compute_real_fairness(raw_model)
    raw_drift = compute_real_drift(raw_model)

    retrieval_fn = _category_routed_retrieval_fn(
        category_hits={"fairness": [_hit(chunk_id="chunk-via-compute", doc_id="doc-x", chunk_index=0, text="t")]}
    )
    evidence_by_rule = build_evidence_by_rule(retrieval_fn=retrieval_fn)

    res = compute_real_compliance(
        raw_model, raw_explain, raw_fairness, raw_drift, evidence_by_rule=evidence_by_rule
    )
    findings = {f["rule_id"]: f for f in res["findings"]}
    fairness_rule_ids = [r["rule_id"] for r in load_rules() if r["category"] == "fairness"]
    for rid in fairness_rule_ids:
        assert findings[rid]["evidence_chunks"] == ["chunk-via-compute"]
    for rid in findings:
        if rid not in fairness_rule_ids:
            assert findings[rid]["evidence_chunks"] == []


def test_compute_real_compliance_existing_behavior_unchanged_when_evidence_absent():
    """Omitting evidence_by_rule reproduces the pre-Phase-5D baseline
    exactly: every finding's evidence_chunks stays []."""
    result = evaluate_compliance(MOCK_TECHNICAL_FINDINGS)
    assert all(f["evidence_chunks"] == [] for f in result["findings"])


# ---------------------------------------------------------------------------
# build_assurance_result(): live end-to-end wiring, not just compute_*
# functions tested in isolation
# ---------------------------------------------------------------------------


def test_build_assurance_result_live_path_forwards_evidence_by_rule(monkeypatch):
    """The live build_assurance_result() -> compute_real_compliance() ->
    evaluate_compliance() chain actually carries retrieved evidence
    through, not just the compute_* helpers exercised directly."""
    retrieval_fn = _category_routed_retrieval_fn(
        category_hits={"drift": [_hit(chunk_id="chunk-live-drift", doc_id="doc-y", chunk_index=0, text="t")]}
    )
    monkeypatch.setattr(
        "app.api.orchestration.build_default_retriever", lambda: retrieval_fn
    )

    res = build_assurance_result()
    findings = {f["rule_id"]: f for f in res["compliance"]["findings"]}
    drift_rule_ids = [r["rule_id"] for r in load_rules() if r["category"] == "drift"]
    for rid in drift_rule_ids:
        assert findings[rid]["evidence_chunks"] == ["chunk-live-drift"]
    for rid in findings:
        if rid not in drift_rule_ids:
            assert findings[rid]["evidence_chunks"] == []


def test_build_assurance_result_live_path_no_evidence_yields_empty_chunks(monkeypatch):
    """When RAG genuinely finds nothing, the live path's findings keep
    evidence_chunks == [] -- the documented NO_VERIFIED_EVIDENCE -> []
    behavior, exercised through the real build_assurance_result() call."""
    retrieval_fn = _category_routed_retrieval_fn(category_hits={})
    monkeypatch.setattr(
        "app.api.orchestration.build_default_retriever", lambda: retrieval_fn
    )

    res = build_assurance_result()
    assert all(f["evidence_chunks"] == [] for f in res["compliance"]["findings"])


# ---------------------------------------------------------------------------
# Adapter-aware model metrics
# ---------------------------------------------------------------------------


def test_evaluate_current_model_default_still_works():
    """Default (no adapter) evaluate_current_model() behavior is unchanged."""
    metrics_default = evaluate_current_model()
    metrics_explicit_none = evaluate_current_model(adapter=None)
    assert metrics_default == metrics_explicit_none
    assert metrics_default["is_mock"] is False
    assert set(metrics_default) == {
        "accuracy", "precision", "recall", "f1", "roc_auc", "n_test_samples", "is_mock",
    }


def test_evaluate_current_model_rf_adapter_differs_from_default():
    """An RF adapter's metrics are computed from the RF model, not the
    default LR model -- and on this dataset the two genuinely differ."""
    rf_adapter = RandomForestAdapter.load_default()
    lr_metrics = evaluate_current_model()
    rf_metrics = evaluate_current_model(adapter=rf_adapter)
    assert rf_metrics != lr_metrics
    assert rf_metrics["is_mock"] is False
    assert set(rf_metrics) == set(lr_metrics)


def test_compute_real_model_metrics_default_backward_compatible():
    """compute_real_model_metrics() with zero args is unchanged."""
    metrics = compute_real_model_metrics()
    assert metrics["roc_auc_status"] == "computed"
    assert metrics["is_mock"] is False


def test_compute_real_model_metrics_with_rf_adapter_matches_direct_call():
    rf_adapter = RandomForestAdapter.load_default()
    via_orchestration = compute_real_model_metrics(adapter=rf_adapter)
    direct = evaluate_current_model(adapter=rf_adapter)
    assert via_orchestration["accuracy"] == direct["accuracy"]
    assert via_orchestration["roc_auc"] == direct["roc_auc"]


def test_build_assurance_result_default_backward_compatible():
    """Omitted adapter: build_assurance_result() output is unchanged --
    same top-level keys, no model_id stamped onto compliance."""
    res = build_assurance_result()
    assert set(res.keys()) == {"model", "explainability", "fairness_drift", "compliance", "note"}
    assert res["model"]["model_metadata"]["version"] == MODEL_VERSION
    assert "model_id" not in res["compliance"]
    for finding in res["compliance"]["findings"]:
        assert "model_id" not in finding


def test_build_assurance_result_with_rf_adapter_uses_rf_throughout():
    """Supplying an RF adapter must use it consistently for predictions,
    metrics, and identity -- never a mix of LR and RF."""
    rf_adapter = RandomForestAdapter.load_default()
    res = build_assurance_result(adapter=rf_adapter)

    expected_model = compute_real_model(adapter=rf_adapter)
    expected_metrics = compute_real_model_metrics(adapter=rf_adapter)

    assert res["model"]["predictions"] == expected_model["predictions"]
    assert res["model"]["probabilities"] == expected_model["probabilities"]
    assert res["model"]["model_metrics"]["accuracy"] == expected_metrics["accuracy"]
    assert res["model"]["model_metrics"]["roc_auc"] == expected_metrics["roc_auc"]

    # Identity/version match the RF adapter, not the default LR model.
    assert res["model"]["model_metadata"]["version"] == rf_adapter.model_version
    assert res["model"]["model_metadata"]["model_type"] == rf_adapter.model_type
    assert res["compliance"]["model_id"] == rf_adapter.model_id
    for finding in res["compliance"]["findings"]:
        assert finding["model_id"] == rf_adapter.model_id

    # And it must NOT be the default LR model's own type/identity.
    assert res["model"]["model_metadata"]["model_type"] != "logistic_regression"
    assert res["compliance"]["model_id"] != MODEL_ID
