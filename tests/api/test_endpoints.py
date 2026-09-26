"""Unit tests for FastAPI endpoints (Owner: Khushi)."""
import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    DriftAssuranceEnvelope,
    DriftComparisonResult,
    ExplainabilityResult,
    FairnessAssuranceEnvelope,
    FairnessDriftResult,
    ModelResult,
    ReportResult,
)
from app.models.preprocessing import DEFAULT_DATASET_PATH, FEATURE_COLUMNS

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_endpoint():
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert "instance_ids" in body
    parsed = ModelResult(**body)
    assert parsed.is_mock is False
    assert len(parsed.predictions) == 200
    assert len(parsed.probabilities) == 200
    assert len(parsed.instance_ids) == 200
    assert len(parsed.instance_ids) == len(parsed.predictions) == len(parsed.probabilities)
    assert all(isinstance(iid, str) and len(iid) > 0 for iid in parsed.instance_ids)
    assert parsed.model_metadata.feature_names == FEATURE_COLUMNS
    assert parsed.model_metadata.label_semantics is not None
    assert parsed.model_metadata.label_semantics.favorable_outcome_label == 0
    assert parsed.model_metrics is not None
    assert parsed.model_metrics.is_mock is False
    assert parsed.model_metrics.n_test_samples == 200
    assert 0.0 <= parsed.model_metrics.accuracy <= 1.0
    assert 0.0 <= parsed.model_metrics.precision <= 1.0
    assert 0.0 <= parsed.model_metrics.recall <= 1.0
    assert 0.0 <= parsed.model_metrics.f1 <= 1.0
    assert 0.0 <= parsed.model_metrics.roc_auc <= 1.0

    # Feature matrix round-trips via DataFrame
    df_recovered = pd.DataFrame(body["feature_matrix"])
    assert len(df_recovered) == 200
    assert list(df_recovered.columns) == FEATURE_COLUMNS


def test_explainability_default_endpoint():
    response = client.get("/explainability")
    assert response.status_code == 200
    body = response.json()
    parsed = ExplainabilityResult(**body)
    assert parsed.is_mock is False
    assert parsed.method == "shap"
    assert len(parsed.per_instance) > 0
    assert len(parsed.global_importance) == len(FEATURE_COLUMNS)


def test_explainability_lime_endpoint():
    response = client.get("/explainability?method=lime")
    assert response.status_code == 200
    body = response.json()
    parsed = ExplainabilityResult(**body)
    assert parsed.is_mock is False
    assert parsed.method == "lime"
    assert len(parsed.per_instance) <= 20
    assert len(parsed.global_importance) == len(FEATURE_COLUMNS)


def test_explainability_invalid_method_returns_4xx():
    response = client.get("/explainability?method=bogus")
    assert response.status_code == 400


def test_fairness_drift_endpoint():
    response = client.get("/fairness-drift")
    assert response.status_code == 200
    body = response.json()
    parsed = FairnessDriftResult(**body)
    assert parsed.fairness.is_mock is False
    assert parsed.fairness.protected_attribute == "personal_status_and_sex"
    assert parsed.fairness.status in {"PASS", "WARNING", "FAIL", "PENDING"}

    # Phase 4 per-group fairness detail
    assert "groups" in body["fairness"]
    assert len(parsed.fairness.groups) > 0
    for g in parsed.fairness.groups:
        assert isinstance(g.group, str)
        assert g.count > 0
        assert g.favorable_count >= 0
        assert 0.0 <= g.selection_rate <= 1.0

    # Drift is real detection on the development train/test split
    assert parsed.drift.is_mock is False
    assert parsed.drift.status in {"PASS", "WARNING", "FAIL", "PENDING"}
    assert parsed.drift.note is None

    # Phase 4 per-feature drift detail
    assert "per_feature" in body["drift"]
    assert len(parsed.drift.per_feature) == len(parsed.drift.features_evaluated)
    assert [pf.feature for pf in parsed.drift.per_feature] == parsed.drift.features_evaluated
    for pf in parsed.drift.per_feature:
        assert pf.psi >= 0.0
        assert pf.ks_statistic >= 0.0


def test_compliance_endpoint():
    response = client.get("/compliance")
    assert response.status_code == 200
    body = response.json()
    parsed = ComplianceResult(**body)
    assert parsed.is_mock is True
    assert len(parsed.findings) == 6
    # Active evaluated statuses (not all pending)
    statuses = {f.status for f in parsed.findings}
    assert statuses - {"PENDING"} != set(), "Compliance findings should evaluate real statuses, not all PENDING"


def test_assurance_result_endpoint():
    response = client.get("/assurance-result")
    assert response.status_code == 200
    body = response.json()
    assert "instance_ids" in body["model"]
    parsed = AssuranceResult(**body)
    # Each section independently reports is_mock
    assert parsed.model.is_mock is False
    assert parsed.model.model_metrics is not None
    assert parsed.model.model_metrics.is_mock is False
    assert parsed.model.model_metrics.n_test_samples == 200
    assert len(parsed.model.instance_ids) == 200
    assert len(parsed.model.instance_ids) == len(parsed.model.predictions) == len(parsed.model.probabilities)
    assert all(isinstance(iid, str) and len(iid) > 0 for iid in parsed.model.instance_ids)
    assert parsed.explainability.is_mock is False
    assert parsed.fairness_drift.fairness.is_mock is False
    assert parsed.fairness_drift.drift.is_mock is False
    assert parsed.compliance.is_mock is True

    # Phase 4 fairness groups and drift per-feature exposed through assurance result
    assert "groups" in body["fairness_drift"]["fairness"]
    assert len(parsed.fairness_drift.fairness.groups) > 0
    assert "per_feature" in body["fairness_drift"]["drift"]
    assert len(parsed.fairness_drift.drift.per_feature) == len(parsed.fairness_drift.drift.features_evaluated)

    # Drift uses the development train/test split and has no synthetic note
    assert parsed.fairness_drift.drift.note is None

    # Overall note documents per-section status
    assert parsed.note != ""
    assert "is_mock: False" in parsed.note
    assert "is_mock: True" in parsed.note


def test_mock_assurance_result_deprecated_endpoint():
    mock_resp = client.get("/mock-assurance-result")
    assert mock_resp.status_code == 200
    parsed = AssuranceResult(**mock_resp.json())
    assert parsed.model.is_mock is True
    assert parsed.model.instance_ids == ["gc-0000", "gc-0001", "gc-0002"]
    assert parsed.model.model_metrics is not None
    assert parsed.compliance.is_mock is True


def test_report_endpoint():
    """The report is produced from the live pipeline, not a stored fixture.

    The coverage figures are deliberately NOT hard-coded here. They are a
    measurement of the retrieval pipeline over the current RBI corpus, so
    pinning them to 1-of-5 (the pre-integration fixture's values) would make
    this test assert a stale claim and fail whenever the corpus is corrected.
    The invariant that matters is that the parts sum to the whole.
    """
    response = client.get("/report")
    assert response.status_code == 200
    body = response.json()
    parsed = ReportResult(**body)
    assert parsed.is_mock is False
    assert len(parsed.sections) == 5
    assert parsed.evidence_coverage.total == 5
    assert (
        parsed.evidence_coverage.retrieved + parsed.evidence_coverage.not_found
        == parsed.evidence_coverage.total
    )
    assert parsed.model_version == "0.1.0"
    assert len(parsed.disclaimers) > 0


def test_fairness_assurance_endpoint():
    response = client.get("/fairness-assurance")
    assert response.status_code == 200
    body = response.json()
    parsed = FairnessAssuranceEnvelope(**body)
    assert parsed.context.model_id == "german-credit-logistic-regression"
    assert parsed.context.model_version == "0.1.0"
    assert isinstance(parsed.context.assurance_run_id, str)
    assert len(parsed.context.assurance_run_id) > 0
    assert parsed.context.adapter_id is None
    assert parsed.result.protected_attribute == "personal_status_and_sex"
    assert parsed.result.is_mock is False
    assert len(parsed.result.groups) > 0


def test_assurance_result_top_level_keys_unchanged():
    response = client.get("/assurance-result")
    assert response.status_code == 200
    body = response.json()
    # UPDATED (final backend pass): the five original keys are unchanged;
    # run identity and the monitoring lane are additive.
    expected_top_keys = {
        "model",
        "explainability",
        "fairness_drift",
        "compliance",
        "note",
        "model_id",
        "assurance_run_id",
        "monitoring",
        "monitoring_unavailable_reason",
    }
    assert set(body.keys()) == expected_top_keys
    assert {"model", "explainability", "fairness_drift", "compliance", "note"} <= set(body)


def test_drift_assurance_endpoint():
    response = client.get("/drift-assurance")
    assert response.status_code == 200
    body = response.json()
    parsed = DriftAssuranceEnvelope(**body)
    assert parsed.context.model_id == "german-credit-logistic-regression"
    assert parsed.context.model_version == "0.1.0"
    assert isinstance(parsed.context.assurance_run_id, str)
    assert len(parsed.context.assurance_run_id) > 0
    assert parsed.dataset_id == DEFAULT_DATASET_PATH
    assert parsed.dataset_version is None
    assert len(parsed.feature_space) == 16
    assert parsed.result.is_mock is False
    assert len(parsed.result.features_evaluated) > 0


def test_drift_comparison_endpoint():
    response = client.get("/drift-comparison")
    assert response.status_code == 200
    body = response.json()
    parsed = DriftComparisonResult(**body)
    assert parsed.comparability == "COMPARABLE"
    assert parsed.drift_a.context.model_id == "german-credit-logistic-regression"
    assert parsed.drift_b.context.model_id == "german-credit-random-forest"
    assert parsed.drift_a.context.model_id != parsed.drift_b.context.model_id
    assert parsed.reason is not None



