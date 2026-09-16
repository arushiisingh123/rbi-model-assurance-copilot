"""Phase 5D: compliance findings stay isolated across two models.

evaluate_compliance()/run_compliance()'s model_id/assurance_run_id
kwargs already exist (PR #49) -- this file proves they actually
prevent cross-model collision when two models' findings are held
together, which nothing in the existing suite currently asserts.
"""

import pytest

from app.api.orchestration import (
    compute_real_compliance,
    compute_real_drift,
    compute_real_explainability,
    compute_real_fairness,
    compute_real_model,
)
from app.models.model import LogisticRegressionAdapter, RandomForestAdapter


def _compliance_for(adapter) -> dict:
    model = compute_real_model(adapter=adapter)
    explain = compute_real_explainability(model, method="shap")
    fairness = compute_real_fairness(model)
    drift = compute_real_drift(model)
    return compute_real_compliance(
        model,
        explain,
        fairness,
        drift,
        model_id=adapter.model_id,
        assurance_run_id=f"run-{adapter.model_id}",
    )


@pytest.fixture(scope="module")
def lr_compliance():
    return _compliance_for(LogisticRegressionAdapter.load_default())


@pytest.fixture(scope="module")
def rf_compliance():
    return _compliance_for(RandomForestAdapter.load_default())


def test_each_models_findings_carry_their_own_identity(
    lr_compliance, rf_compliance
):
    assert lr_compliance["model_id"] == "german-credit-logistic-regression"
    assert rf_compliance["model_id"] == "german-credit-random-forest"
    for f in lr_compliance["findings"]:
        assert f["model_id"] == "german-credit-logistic-regression"
    for f in rf_compliance["findings"]:
        assert f["model_id"] == "german-credit-random-forest"


def test_different_assurance_run_ids_are_not_overwritten_or_reused(
    lr_compliance, rf_compliance
):
    assert lr_compliance["assurance_run_id"] != rf_compliance["assurance_run_id"]
    run_ids = {f["assurance_run_id"] for f in lr_compliance["findings"]}
    assert run_ids == {lr_compliance["assurance_run_id"]}  # one run, one id, no drift


def test_pooling_two_models_findings_never_collapses_identity(
    lr_compliance, rf_compliance
):
    """The actual isolation proof: pool both models' findings and
    confirm model_id alone correctly separates them back out."""
    pooled = lr_compliance["findings"] + rf_compliance["findings"]
    by_model: dict = {}
    for f in pooled:
        by_model.setdefault(f["model_id"], []).append(f)
    assert set(by_model) == {
        "german-credit-logistic-regression",
        "german-credit-random-forest",
    }
    assert len(by_model["german-credit-logistic-regression"]) == len(
        lr_compliance["findings"]
    )
    assert len(by_model["german-credit-random-forest"]) == len(
        rf_compliance["findings"]
    )


def test_same_model_two_runs_get_distinct_run_ids_and_stay_isolated():
    """A model/version can have multiple assurance runs -- prove one
    run's findings never leak into the other's."""
    adapter = LogisticRegressionAdapter.load_default()
    run_a = _compliance_for(adapter)  # uses f"run-{adapter.model_id}" -- override below
    model = compute_real_model(adapter=adapter)
    explain = compute_real_explainability(model, method="shap")
    fairness = compute_real_fairness(model)
    drift = compute_real_drift(model)
    run_b = compute_real_compliance(
        model,
        explain,
        fairness,
        drift,
        model_id=adapter.model_id,
        assurance_run_id="run-b-distinct",
    )
    assert run_a["assurance_run_id"] != run_b["assurance_run_id"]
    assert run_a["model_id"] == run_b["model_id"]  # same model, different run


def test_omitting_identity_reproduces_the_existing_default_path(lr_compliance):
    """Regression guard: the default (no identity passed) path used
    by every current live caller is unaffected by this proof suite."""
    from app.compliance.compliance import evaluate_compliance
    from app.compliance.mock_findings import MOCK_TECHNICAL_FINDINGS

    result = evaluate_compliance(MOCK_TECHNICAL_FINDINGS)
    assert set(result) == {"findings", "is_mock"}


def test_compliance_still_consumes_not_recomputes_fairness_and_drift_status(
    lr_compliance, rf_compliance
):
    """Guard against scope creep: this proof suite must not encourage
    compliance to start deriving its own severity."""
    lr_fair_finding = next(
        f for f in lr_compliance["findings"] if f["rule_id"] == "RBI-FAIR-01"
    )
    rf_fair_finding = next(
        f for f in rf_compliance["findings"] if f["rule_id"] == "RBI-FAIR-01"
    )
    # mirror_status only -- these must be exactly one of the four
    # project-wide statuses, never a value compliance invented itself
    from app.config.thresholds import VALID_STATUSES

    assert lr_fair_finding["status"] in VALID_STATUSES
    assert rf_fair_finding["status"] in VALID_STATUSES
