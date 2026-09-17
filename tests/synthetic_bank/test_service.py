"""Unit tests for the synthetic bank FastAPI service (owner: Manas).

Uses FastAPI's TestClient directly against app.synthetic_bank.service.app --
no live socket/server needed for these; see
tests/integration/test_synthetic_bank_end_to_end.py for the real-HTTP proof.
"""
from fastapi.testclient import TestClient

from app.synthetic_bank.data_generator import FEATURE_COLUMNS
from app.synthetic_bank.model import MODEL_ID, MODEL_TYPE, MODEL_VERSION
from app.synthetic_bank.service import app

client = TestClient(app)

VALID_PAYLOAD = {
    "employment_type": "salaried",
    "region": "north",
    "loan_purpose": "auto",
    "age": 45,
    "annual_income": 90000,
    "employment_years": 15,
    "existing_loans": 0,
    "credit_utilization_ratio": 0.1,
    "late_payments_12m": 0,
    "loan_amount": 8000,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == MODEL_ID
    assert body["model_version"] == MODEL_VERSION
    assert body["model_type"] == MODEL_TYPE
    assert body["feature_names"] == FEATURE_COLUMNS


def test_score_valid_payload():
    response = client.post("/score", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == MODEL_VERSION


def test_score_missing_field_returns_422_not_500():
    payload = dict(VALID_PAYLOAD)
    del payload["age"]
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_extra_field_returns_422():
    payload = dict(VALID_PAYLOAD)
    payload["unexpected_field"] = "surprise"
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_wrong_type_returns_422():
    payload = dict(VALID_PAYLOAD)
    payload["age"] = "not-a-number"
    response = client.post("/score", json=payload)
    assert response.status_code == 422
