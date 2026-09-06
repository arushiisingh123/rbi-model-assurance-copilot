"""Phase 2 integration tests for the Nidhi-owned compliance chain.

Proves the complete chain that is Nidhi's Phase 2 responsibility:

    real model output       (app.models.model.predict_batch)
    real explainability      (app.explainability.explain)
    real fairness            (app.fairness.fairness_report)
    real drift               (app.drift.drift_report)
        -> build_technical_findings()
        -> evaluate_compliance()  /  run_compliance()
        -> RBI rule evaluation
        -> compliance findings

Nothing here recomputes a fairness/drift metric or status -- the compliance
layer consumes what the analytical modules reported (docs/decisions.md,
"Analytical threshold authority"). Evidence retrieval (RAG) is Phase 3, so
``evidence_chunks`` stays empty and the result stays ``is_mock: True``.

The real model artifact is gitignored; a session fixture provisions it once
via Namitha's own ``train()``, mirroring ``tests/explainability/conftest.py``.
"""
import os

import pytest

from app.compliance import (
    build_technical_findings,
    evaluate_compliance,
    run_compliance,
)
from app.compliance.engine import STATUSES
from app.config.thresholds import VALID_STATUSES
from app.rbi.metadata import RULE_SET_DISCLAIMER
from app.rbi.rules import load_rules

APPROVED_FINDING_KEYS = {
    "rule_id",
    "rule_description",
    "technical_finding_ref",
    "status",
    "evidence_chunks",
}


# --------------------------------------------------------------------------
# Real-module fixtures (session-scoped: the real model is trained once)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _real_model_artifact():
    """Ensure a trained model artifact exists before the real chain runs."""
    from app.models.model import DEFAULT_MODEL_ARTIFACT_PATH, train
    from app.models.preprocessing import DEFAULT_DATASET_PATH

    if not os.path.exists(DEFAULT_MODEL_ARTIFACT_PATH):
        train(
            dataset_path=DEFAULT_DATASET_PATH,
            save_path=DEFAULT_MODEL_ARTIFACT_PATH,
            random_state=42,
        )
    return DEFAULT_MODEL_ARTIFACT_PATH


@pytest.fixture(scope="session")
def real_model_output(_real_model_artifact) -> dict:
    """app.models.model.predict_batch() on the held-out test split."""
    from app.models.model import predict_batch

    return predict_batch()


@pytest.fixture(scope="session")
def real_explainability_output(_real_model_artifact, real_model_output) -> dict:
    """app.explainability.explain() (SHAP) for the real model."""
    from app.explainability import explain

    return explain(real_model_output, method="shap")


@pytest.fixture(scope="session")
def real_fairness_output(real_model_output) -> dict:
    """app.fairness.fairness_report() on the real predictions.

    The favourable credit outcome is label 0 (GOOD) for this model, so
    favorable_label is stated explicitly rather than left to a default.
    The sensitive feature is raw Attribute 9 (personal_status_and_sex),
    aligned row-for-row with the predictions via the returned feature
    matrix -- it is not a standalone gender column.
    """
    from app.fairness import fairness_report

    sensitive = real_model_output["feature_matrix"]["personal_status_and_sex"]
    return fairness_report(
        real_model_output["predictions"],
        sensitive,
        favorable_label=0,
    )


@pytest.fixture(scope="session")
def real_drift_output(real_model_output) -> dict:
    """app.drift.drift_report() on a deterministic synthetic shift.

    The 'current' frame is a controlled shift of the real feature matrix
    (docs/decisions.md, "MVP drift reference/current dataset strategy") --
    synthetic by construction, not observed drift.
    """
    from app.drift.drift import drift_report
    from app.drift.scenario import build_drift_scenario
    from app.models.preprocessing import NUMERIC_FEATURES

    reference = real_model_output["feature_matrix"]
    ref, cur = build_drift_scenario(
        reference, shift_features=NUMERIC_FEATURES[0], shift_amount=1.5
    )
    return drift_report(ref, cur)


@pytest.fixture(scope="session")
def real_technical_findings(
    real_model_output,
    real_explainability_output,
    real_fairness_output,
    real_drift_output,
) -> dict:
    return build_technical_findings(
        model=real_model_output,
        explainability=real_explainability_output,
        fairness=real_fairness_output,
        drift=real_drift_output,
    )


@pytest.fixture(scope="session")
def real_compliance_result(real_technical_findings) -> dict:
    return evaluate_compliance(real_technical_findings)


# --------------------------------------------------------------------------
# The real module outputs really do have the shape the engine expects
# --------------------------------------------------------------------------


def test_real_module_outputs_are_real_not_mock(
    real_model_output,
    real_explainability_output,
    real_fairness_output,
    real_drift_output,
):
    assert real_model_output["is_mock"] is False
    assert real_explainability_output["is_mock"] is False
    assert real_fairness_output["is_mock"] is False
    assert real_drift_output["is_mock"] is False


def test_build_technical_findings_assembles_all_four_sections(real_technical_findings):
    assert set(real_technical_findings) == {
        "model",
        "explainability",
        "fairness",
        "drift",
    }
    # Stored verbatim -- not copied, reshaped, or reclassified.
    assert "status" in real_technical_findings["fairness"]
    assert "status" in real_technical_findings["drift"]
    assert "global_importance" in real_technical_findings["explainability"]
    assert "model_metadata" in real_technical_findings["model"]


# --------------------------------------------------------------------------
# evaluate_compliance() accepts the real findings and covers every rule
# --------------------------------------------------------------------------


def test_evaluate_compliance_accepts_real_findings_without_error(real_compliance_result):
    assert set(real_compliance_result) == {"findings", "is_mock"}


def test_every_configured_rule_produces_one_finding(real_compliance_result):
    finding_ids = [f["rule_id"] for f in real_compliance_result["findings"]]
    rule_ids = [r["rule_id"] for r in load_rules()]
    assert finding_ids == rule_ids
    assert len(finding_ids) == len(set(finding_ids))


def test_findings_preserve_traceability_fields(real_compliance_result):
    rules_by_id = {r["rule_id"]: r for r in load_rules()}
    for finding in real_compliance_result["findings"]:
        assert set(finding) == APPROVED_FINDING_KEYS
        rule = rules_by_id[finding["rule_id"]]
        assert finding["rule_description"] == rule["rule_description"]
        assert finding["technical_finding_ref"] == rule["technical_finding_ref"]
        assert finding["rule_description"].strip()


def test_all_statuses_are_in_the_project_vocabulary(real_compliance_result):
    for finding in real_compliance_result["findings"]:
        assert finding["status"] in STATUSES
        assert finding["status"] in VALID_STATUSES  # PASS / WARNING / FAIL / PENDING


# --------------------------------------------------------------------------
# Each documented technical_finding_ref resolves against the real findings
# --------------------------------------------------------------------------


def _status_for(result: dict, rule_id: str) -> str:
    return next(f["status"] for f in result["findings"] if f["rule_id"] == rule_id)


def test_fairness_status_is_consumed_through_fairness_status(
    real_compliance_result, real_fairness_output
):
    # RBI-FAIR-01 mirrors fairness.status -- it must equal what the fairness
    # module itself reported, never a value compliance derived on its own.
    assert _status_for(real_compliance_result, "RBI-FAIR-01") == real_fairness_output[
        "status"
    ]


def test_drift_status_is_consumed_through_drift_status(
    real_compliance_result, real_drift_output
):
    assert _status_for(real_compliance_result, "RBI-DRIFT-01") == real_drift_output[
        "status"
    ]


def test_model_metadata_reference_resolves(real_compliance_result):
    assert _status_for(real_compliance_result, "RBI-MODEL-01") == "PASS"


def test_explainability_global_importance_reference_resolves(real_compliance_result):
    assert _status_for(real_compliance_result, "RBI-EXPL-01") == "PASS"


def test_demographic_parity_diff_reference_resolves(real_compliance_result):
    # presence-only rule (docs/thresholds.md sec 4.2): reported -> PASS.
    assert _status_for(real_compliance_result, "RBI-FAIR-02") == "PASS"


def test_ks_statistic_reference_resolves(real_compliance_result):
    assert _status_for(real_compliance_result, "RBI-DRIFT-02") == "PASS"


# --------------------------------------------------------------------------
# Missing sections -> PENDING, never an exception
# --------------------------------------------------------------------------


def test_missing_sections_are_pending_not_errors(real_model_output):
    # Only the model section is available.
    result = evaluate_compliance(build_technical_findings(model=real_model_output))
    statuses = {f["rule_id"]: f["status"] for f in result["findings"]}

    assert statuses["RBI-MODEL-01"] == "PASS"
    assert statuses["RBI-FAIR-01"] == "PENDING"
    assert statuses["RBI-FAIR-02"] == "PENDING"
    assert statuses["RBI-DRIFT-01"] == "PENDING"
    assert statuses["RBI-DRIFT-02"] == "PENDING"
    assert statuses["RBI-EXPL-01"] == "PENDING"


def test_no_sections_at_all_is_all_pending():
    result = run_compliance()
    assert result["is_mock"] is True
    assert result["findings"]
    assert all(f["status"] == "PENDING" for f in result["findings"])


def test_build_technical_findings_omits_none_sections(real_fairness_output):
    findings = build_technical_findings(fairness=real_fairness_output)
    assert set(findings) == {"fairness"}


# --------------------------------------------------------------------------
# Compliance does not invent or recalculate fairness/drift status
# --------------------------------------------------------------------------


def test_compliance_mirrors_reported_status_and_never_recomputes_it():
    # fairness.status says FAIL, but the raw ratio (0.99) would classify as
    # PASS. Compliance must report FAIL: it consumes the reported status and
    # does not re-derive severity from the metric.
    tampered = build_technical_findings(
        fairness={
            "protected_attribute": "personal_status_and_sex",
            "demographic_parity_diff": 0.01,
            "disparate_impact_ratio": 0.99,
            "status": "FAIL",
            "is_mock": False,
        },
        drift={
            "features_evaluated": ["age"],
            "psi": 0.9,  # would classify FAIL
            "ks_statistic": 0.5,
            "status": "PASS",  # ... but the drift module reported PASS
            "is_mock": False,
        },
    )
    result = evaluate_compliance(tampered)
    assert _status_for(result, "RBI-FAIR-01") == "FAIL"
    assert _status_for(result, "RBI-DRIFT-01") == "PASS"


def test_run_compliance_matches_manual_assemble_then_evaluate(
    real_model_output,
    real_explainability_output,
    real_fairness_output,
    real_drift_output,
):
    via_helper = run_compliance(
        model=real_model_output,
        explainability=real_explainability_output,
        fairness=real_fairness_output,
        drift=real_drift_output,
    )
    manual = evaluate_compliance(
        build_technical_findings(
            model=real_model_output,
            explainability=real_explainability_output,
            fairness=real_fairness_output,
            drift=real_drift_output,
        )
    )
    assert via_helper == manual


# --------------------------------------------------------------------------
# Phase 1 disclaimer is preserved (Phase 2 adds no verified regulatory claim)
# --------------------------------------------------------------------------


def test_illustrative_rule_disclaimer_is_preserved(real_compliance_result):
    assert real_compliance_result["is_mock"] is True
    assert all(f["evidence_chunks"] == [] for f in real_compliance_result["findings"])
    assert "ILLUSTRATIVE" in RULE_SET_DISCLAIMER.upper() or "illustrative" in RULE_SET_DISCLAIMER
    for rule in load_rules():
        assert rule["clause_reference"] is None
        assert "ILLUSTRATIVE" in rule["rbi_source"]
