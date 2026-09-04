"""Unit tests for FastAPI endpoints (Owner: Khushi)."""
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    ExplainabilityResult,
    FairnessDriftResult,
    ModelResult,
)

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_endpoint():
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    parsed = ModelResult(**body)
    assert parsed.is_mock is True
    assert len(parsed.predictions) == 3
    assert parsed.model_metadata.feature_names == ["income", "age", "credit_history_len"]


def test_explainability_default_endpoint():
    response = client.get("/explainability")
    assert response.status_code == 200
    body = response.json()
    parsed = ExplainabilityResult(**body)
    assert parsed.is_mock is True
    assert parsed.method == "shap"


def test_explainability_lime_endpoint():
    response = client.get("/explainability?method=lime")
    assert response.status_code == 200
    body = response.json()
    parsed = ExplainabilityResult(**body)
    assert parsed.is_mock is True
    assert parsed.method == "lime"


def test_explainability_invalid_method_returns_4xx():
    response = client.get("/explainability?method=bogus")
    assert response.status_code >= 400 and response.status_code < 500


def test_fairness_drift_endpoint():
    response = client.get("/fairness-drift")
    assert response.status_code == 200
    body = response.json()
    parsed = FairnessDriftResult(**body)
    assert parsed.fairness.is_mock is True
    assert parsed.drift.is_mock is True
    assert parsed.fairness.status in {"PASS", "WARNING", "FAIL", "PENDING"}
    assert parsed.drift.status in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_compliance_endpoint():
    response = client.get("/compliance")
    assert response.status_code == 200
    body = response.json()
    parsed = ComplianceResult(**body)
    assert parsed.is_mock is True
    assert len(parsed.findings) >= 2


def test_assurance_result_endpoint():
    response = client.get("/assurance-result")
    assert response.status_code == 200
    body = response.json()
    parsed = AssuranceResult(**body)
    assert parsed.model.is_mock is True
    assert parsed.explainability.is_mock is True
    assert parsed.fairness_drift.fairness.is_mock is True
    assert parsed.fairness_drift.drift.is_mock is True
    assert parsed.compliance.is_mock is True
    assert parsed.note != ""


def test_mock_assurance_result_endpoint_matches_assurance_result():
    mock_resp = client.get("/mock-assurance-result")
    canon_resp = client.get("/assurance-result")
    assert mock_resp.status_code == 200
    assert canon_resp.status_code == 200
    assert mock_resp.json() == canon_resp.json()
    parsed = AssuranceResult(**mock_resp.json())
    assert parsed.model.is_mock is True
