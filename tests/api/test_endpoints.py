"""Unit tests for FastAPI endpoints (Owner: Khushi)."""
import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    ExplainabilityResult,
    FairnessDriftResult,
    ModelResult,
    ReportResult,
)
from app.models.preprocessing import FEATURE_COLUMNS

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

    # Drift is real detection on the development train/test split
    assert parsed.drift.is_mock is False
    assert parsed.drift.status in {"PASS", "WARNING", "FAIL", "PENDING"}
    assert parsed.drift.note is None


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
    assert len(parsed.model.instance_ids) == 200
    assert len(parsed.model.instance_ids) == len(parsed.model.predictions) == len(parsed.model.probabilities)
    assert all(isinstance(iid, str) and len(iid) > 0 for iid in parsed.model.instance_ids)
    assert parsed.explainability.is_mock is False
    assert parsed.fairness_drift.fairness.is_mock is False
    assert parsed.fairness_drift.drift.is_mock is False
    assert parsed.compliance.is_mock is True

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
    assert parsed.compliance.is_mock is True


def test_report_endpoint():
    response = client.get("/report")
    assert response.status_code == 200
    body = response.json()
    parsed = ReportResult(**body)
    assert parsed.is_mock is True
    assert len(parsed.sections) == 5
    assert parsed.evidence_coverage.total == 5
    assert parsed.evidence_coverage.retrieved == 1
    assert parsed.evidence_coverage.not_found == 4
    assert parsed.model_version == "0.1.0"
    assert len(parsed.disclaimers) > 0

