"""Phase 5D orchestration wiring (owner: Nidhi + Khushi's follow-up).

Three related, additive changes to app/api/orchestration.py:

1. RAG -> compliance evidence wiring: build_evidence_by_rule() retrieves
   real RBI evidence per rule (app/rag/) and threads it through
   compute_real_compliance() into evaluate_compliance()'s existing
   evidence_by_rule parameter (app/compliance/compliance.py -- unchanged).
   Exercised through both build_assurance_result() and GET /compliance
   (app/api/main.py) -- the latter previously the only compliance path NOT
   wired to build_evidence_by_rule(), despite being the one that supports
   model_id.
2. Adapter-aware model metrics: compute_real_model_metrics() and
   build_assurance_result() accept an optional adapter so predictions,
   metrics, and identity all describe the same model. Guarded in
   evaluate_current_model() (app/models/model.py) against a schema-
   mismatched adapter (e.g. the synthetic bank) -- raises ValueError
   instead of crashing confusingly, so GET /model degrades model_metrics
   to null rather than reporting a mismatched model's numbers.
3. Adapter-aware fairness/drift: compute_real_fairness()/compute_real_drift()
   accept an optional adapter. Fairness reads the protected attribute from
   adapter.protected_attribute instead of hardcoding
   "personal_status_and_sex" (None declared -> PENDING, not a KeyError).
   Drift uses adapter.background_data() as the reference distribution
   instead of always loading German Credit, when the adapter's schema
   differs -- closing a prior finding where a coincidental column-name
   overlap (both schemas have an "age" column) let drift silently fabricate
   a plausible-looking result comparing two unrelated populations. See
   tests/integration/test_synthetic_bank_end_to_end.py for the live,
   through-the-route proof of this fix.

Deliberately does not exercise app/rag/retrieval.py's real corpus scoring
for the isolation/ordering assertions below -- those use a hand-built
Task 5-shaped retrieval double (matching tests/rag/test_evidence_records.py
and tests/compliance/test_compliance_evidence_chunks.py's own pattern), so
these tests do not depend on which RBI documents happen to be indexed.
"""
from typing import Any, Dict, List

import pandas as pd
import pytest

from app.api.orchestration import (
    NO_PROTECTED_ATTRIBUTE_DECLARED,
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
from app.models.model import (
    FEATURE_COLUMNS,
    MODEL_ID,
    MODEL_VERSION,
    RandomForestAdapter,
    evaluate_current_model,
)
from app.models import get_default_registry
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
# GET /compliance (app/api/main.py, owner: Khushi): the standalone route --
# previously the only compliance path NOT wired to build_evidence_by_rule(),
# even though it (unlike build_assurance_result()) supports model_id.
# ---------------------------------------------------------------------------


def test_get_compliance_route_forwards_live_evidence_by_rule(monkeypatch):
    """GET /compliance must carry real retrieved evidence into
    evidence_chunks, the same way build_assurance_result() already does --
    proving the standalone route was actually wired, not just the
    aggregate result."""
    from fastapi.testclient import TestClient

    from app.api.main import app

    retrieval_fn = _category_routed_retrieval_fn(
        category_hits={"fairness": [_hit(chunk_id="chunk-via-route", doc_id="doc-z", chunk_index=0, text="t")]}
    )
    monkeypatch.setattr(
        "app.api.orchestration.build_default_retriever", lambda: retrieval_fn
    )

    client = TestClient(app)
    response = client.get("/compliance")
    assert response.status_code == 200
    findings = {f["rule_id"]: f for f in response.json()["findings"]}

    fairness_rule_ids = [r["rule_id"] for r in load_rules() if r["category"] == "fairness"]
    for rid in fairness_rule_ids:
        assert findings[rid]["evidence_chunks"] == ["chunk-via-route"]
    for rid in findings:
        if rid not in fairness_rule_ids:
            assert findings[rid]["evidence_chunks"] == []


def test_get_compliance_route_no_evidence_yields_empty_chunks(monkeypatch):
    """Same route, RAG genuinely finds nothing -- evidence_chunks stays []
    for every finding, never fabricated."""
    from fastapi.testclient import TestClient

    from app.api.main import app

    retrieval_fn = _category_routed_retrieval_fn(category_hits={})
    monkeypatch.setattr(
        "app.api.orchestration.build_default_retriever", lambda: retrieval_fn
    )

    client = TestClient(app)
    response = client.get("/compliance")
    assert response.status_code == 200
    assert all(f["evidence_chunks"] == [] for f in response.json()["findings"])


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
    assert {"model", "explainability", "fairness_drift", "compliance", "note"} <= set(res)
    assert res["model"]["model_metadata"]["version"] == MODEL_VERSION

    # No adapter was named, so no model identity is claimed for the run
    # or stamped onto compliance -- unchanged from before.
    assert res["model_id"] is None
    assert res["compliance"].get("model_id") is None

    # Monitoring is adapter-driven, so the default path reports it as
    # unavailable WITH A REASON rather than as "no drift found".
    assert res["monitoring"] is None
    assert "adapter" in res["monitoring_unavailable_reason"]
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


# ---------------------------------------------------------------------------
# Adapter-aware evaluate_current_model()/compute_real_model_metrics():
# schema-mismatch guard (Khushi's follow-up on top of Nidhi's PR #57)
# ---------------------------------------------------------------------------


def test_evaluate_current_model_raises_for_schema_mismatched_adapter():
    """A different-schema adapter (the synthetic bank) has no meaning under
    German Credit's held-out split -- this must raise a clear ValueError,
    not a confusing KeyError/NotImplementedError from deep inside column
    selection or load_fitted_model()."""
    bank_adapter = get_default_registry().get("synthetic-bank-credit-v1")
    with pytest.raises(ValueError, match="synthetic-bank-credit-v1"):
        evaluate_current_model(adapter=bank_adapter)


def test_compute_real_model_metrics_raises_for_schema_mismatched_adapter():
    """Same guard, reachable through the orchestration wrapper."""
    bank_adapter = get_default_registry().get("synthetic-bank-credit-v1")
    with pytest.raises(ValueError):
        compute_real_model_metrics(adapter=bank_adapter)


# ---------------------------------------------------------------------------
# Adapter-aware compute_real_fairness(): protected attribute comes from the
# adapter, not a hardcoded column name
# ---------------------------------------------------------------------------


def _fake_synthetic_bank_model_dict(n: int = 4) -> Dict[str, Any]:
    """A hand-built model_dict shaped like predict_batch()'s output for the
    synthetic bank -- avoids needing a live HTTP server for fairness/drift
    unit tests, which never touch the network themselves."""
    return {
        "predictions": [0, 1, 0, 1][:n],
        "feature_matrix": pd.DataFrame(
            {
                "employment_type": ["salaried", "unemployed", "salaried", "retired"][:n],
                "region": ["north", "south", "east", "west"][:n],
                "loan_purpose": ["auto", "personal", "home", "education"][:n],
                "age": [30, 40, 50, 60][:n],
                "annual_income": [50000, 20000, 80000, 30000][:n],
                "employment_years": [5, 1, 20, 2][:n],
                "existing_loans": [1, 3, 0, 2][:n],
                "credit_utilization_ratio": [0.2, 0.9, 0.1, 0.5][:n],
                "late_payments_12m": [0, 4, 0, 1][:n],
                "loan_amount": [10000, 15000, 20000, 5000][:n],
            }
        ),
        "model_metadata": {"label_semantics": {"favorable_outcome_label": 0}},
    }


def test_compute_real_fairness_default_unchanged():
    """Omitting adapter is byte-identical to before adapter existed as a
    parameter: hardcoded personal_status_and_sex, computed via the real
    default LR model."""
    raw_model = compute_real_model()
    result = compute_real_fairness(raw_model)
    assert result["protected_attribute"] == "personal_status_and_sex"
    assert result["status"] in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_compute_real_fairness_with_rf_adapter_uses_same_protected_attribute():
    """RF shares German Credit's protected attribute -- passing the adapter
    must not change which column is used."""
    rf_adapter = RandomForestAdapter.load_default()
    raw_model = compute_real_model(adapter=rf_adapter)
    result = compute_real_fairness(raw_model, adapter=rf_adapter)
    assert result["protected_attribute"] == "personal_status_and_sex"


def test_compute_real_fairness_with_no_protected_attribute_declared_is_pending():
    """An adapter declaring no protected attribute (the synthetic bank,
    today) gets a PENDING result -- never a KeyError, never a guessed or
    fabricated attribute name."""
    bank_adapter = get_default_registry().get("synthetic-bank-credit-v1")
    model_dict = _fake_synthetic_bank_model_dict()

    result = compute_real_fairness(model_dict, adapter=bank_adapter)
    assert result["protected_attribute"] == NO_PROTECTED_ATTRIBUTE_DECLARED
    assert result["status"] == "PENDING"
    assert result["demographic_parity_diff"] == 0.0
    assert result["disparate_impact_ratio"] == 1.0
    assert result["groups"] == []
    assert result["is_mock"] is False


# ---------------------------------------------------------------------------
# Adapter-aware compute_real_drift(): reference distribution comes from the
# adapter's own background_data() for a different-schema model
# ---------------------------------------------------------------------------


def test_compute_real_drift_default_unchanged():
    """Omitting adapter is byte-identical to before: German Credit's own
    training split is the reference."""
    raw_model = compute_real_model()
    result = compute_real_drift(raw_model)
    assert set(result["features_evaluated"]).issubset(set(FEATURE_COLUMNS))


def test_compute_real_drift_with_rf_adapter_still_uses_german_credit_reference():
    """RF shares German Credit's schema -- passing the adapter must not
    change the reference distribution used."""
    rf_adapter = RandomForestAdapter.load_default()
    raw_model = compute_real_model(adapter=rf_adapter)

    without_adapter = compute_real_drift(raw_model)
    with_adapter = compute_real_drift(raw_model, adapter=rf_adapter)
    assert with_adapter["features_evaluated"] == without_adapter["features_evaluated"]


def test_compute_real_drift_different_schema_adapter_uses_own_background_data():
    """The core fix: a different-schema adapter's drift reference comes from
    its own background_data(), so ALL of its features are evaluated -- not
    just a coincidental column-name overlap with German Credit (e.g. both
    schemas happen to have a column called "age")."""
    from app.synthetic_bank.data_generator import NUMERIC_FEATURES

    bank_adapter = get_default_registry().get("synthetic-bank-credit-v1")
    model_dict = _fake_synthetic_bank_model_dict()

    result = compute_real_drift(model_dict, adapter=bank_adapter)
    assert set(result["features_evaluated"]) == set(NUMERIC_FEATURES)
    # Never just the one coincidentally-overlapping column -- that was the bug.
    assert result["features_evaluated"] != ["age"]


def test_compute_real_drift_no_background_data_yields_pending_not_a_crash():
    """An adapter with a different schema and no background data at all
    (background_data() returns None) must degrade to drift_report()'s own
    documented PENDING result -- never raise, never silently compare against
    an empty-but-unchecked reference."""
    from app.models.rest_adapter import RESTAdapter

    no_background_adapter = RESTAdapter(
        model_id="no-bg-test-model",
        model_version="0.0.1",
        model_type="test",
        endpoint_url="http://127.0.0.1:9999",
        feature_names=["employment_type", "age"],
        input_schema={"employment_type": {"type": "categorical"}, "age": {"type": "numeric"}},
        capabilities={"predict_proba": False, "batch": True, "explainability": False},
        # background left as default None
    )
    model_dict = _fake_synthetic_bank_model_dict()

    result = compute_real_drift(model_dict, adapter=no_background_adapter)
    assert result["status"] == "PENDING"
    assert result["features_evaluated"] == []
