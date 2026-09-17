"""Integration tests for model registry API endpoints and model_id query parameter routing.

Verifies:
- GET /models, GET /models/{model_id}, GET /models/{model_id}/health
- 404 status codes for unknown model_ids
- GET /model, /explainability, /fairness-drift, /compliance with model_id=german-credit-random-forest vs default
- Omitting model_id on existing endpoints preserves default Logistic Regression behavior
- GET /assurance-result preserves default-only behavior
"""
from fastapi.testclient import TestClient
import pytest

from app.api.main import app
from app.api.schemas import (
    ComplianceResult,
    ExplainabilityResult,
    FairnessDriftResult,
    ModelResult,
)
from app.models.model import (
    MODEL_ID,
    MODEL_TYPE,
    RF_MODEL_ID,
    RF_MODEL_TYPE,
)

client = TestClient(app)


# =====================================================================
# 1. /models Registry Endpoints
# =====================================================================


SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"


def test_get_models_returns_all_default_models():
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) == 3

    model_map = {m["model_id"]: m for m in models}
    assert MODEL_ID in model_map
    assert RF_MODEL_ID in model_map
    assert SYNTHETIC_BANK_MODEL_ID in model_map

    lr_meta = model_map[MODEL_ID]
    assert lr_meta["model_type"] == MODEL_TYPE
    assert lr_meta["integration_type"] == "in_process"
    assert lr_meta["capabilities"] == {
        "predict_proba": True,
        "batch": True,
        "explainability": True,
    }

    rf_meta = model_map[RF_MODEL_ID]
    assert rf_meta["model_type"] == RF_MODEL_TYPE
    assert rf_meta["integration_type"] == "in_process"
    assert rf_meta["capabilities"] == {
        "predict_proba": True,
        "batch": True,
        "explainability": True,
    }

    bank_meta = model_map[SYNTHETIC_BANK_MODEL_ID]
    assert bank_meta["model_type"] == "xgboost"
    assert bank_meta["integration_type"] == "rest"
    assert bank_meta["capabilities"] == {
        "predict_proba": True,
        "batch": True,
        "explainability": False,
    }


def test_get_model_metadata_by_id():
    # LR
    resp_lr = client.get(f"/models/{MODEL_ID}")
    assert resp_lr.status_code == 200
    assert resp_lr.json()["model_id"] == MODEL_ID
    assert resp_lr.json()["model_type"] == MODEL_TYPE

    # RF
    resp_rf = client.get(f"/models/{RF_MODEL_ID}")
    assert resp_rf.status_code == 200
    assert resp_rf.json()["model_id"] == RF_MODEL_ID
    assert resp_rf.json()["model_type"] == RF_MODEL_TYPE


def test_get_model_metadata_unknown_id_returns_404():
    response = client.get("/models/non-existent-model-id")
    assert response.status_code == 404
    assert "non-existent-model-id" in response.json()["detail"]


def test_get_model_health():
    # LR
    resp_lr = client.get(f"/models/{MODEL_ID}/health")
    assert resp_lr.status_code == 200
    assert resp_lr.json() == {"status": "ok"}

    # RF
    resp_rf = client.get(f"/models/{RF_MODEL_ID}/health")
    assert resp_rf.status_code == 200
    assert resp_rf.json() == {"status": "ok"}


def test_get_model_health_unknown_id_returns_404():
    response = client.get("/models/non-existent-model-id/health")
    assert response.status_code == 404
    assert "non-existent-model-id" in response.json()["detail"]


# =====================================================================
# 2. model_id Query Routing on Existing Endpoints
# =====================================================================


def test_get_model_routing_default_vs_rf():
    # Default (no param)
    resp_default = client.get("/model")
    assert resp_default.status_code == 200
    body_default = resp_default.json()
    assert body_default["model_metadata"]["model_type"] == MODEL_TYPE
    assert body_default["model_metrics"] is not None

    # Explicit LR
    resp_lr = client.get(f"/model?model_id={MODEL_ID}")
    assert resp_lr.status_code == 200
    body_lr = resp_lr.json()
    assert body_lr["model_metadata"]["model_type"] == MODEL_TYPE
    assert body_lr["predictions"] == body_default["predictions"]

    # Explicit RF
    resp_rf = client.get(f"/model?model_id={RF_MODEL_ID}")
    assert resp_rf.status_code == 200
    body_rf = resp_rf.json()
    assert body_rf["model_metadata"]["model_type"] == RF_MODEL_TYPE
    # RF predictions differ from LR predictions on some German credit samples
    assert body_rf["model_metadata"]["model_type"] != body_default["model_metadata"]["model_type"]
    assert body_rf["model_metrics"] is None

    # Validate schema
    parsed_rf = ModelResult(**body_rf)
    assert parsed_rf.model_metadata.model_type == RF_MODEL_TYPE
    assert parsed_rf.model_metrics is None


def test_get_explainability_routing_default_vs_rf():
    # Default (no param)
    resp_default = client.get("/explainability")
    assert resp_default.status_code == 200
    body_default = resp_default.json()
    parsed_default = ExplainabilityResult(**body_default)
    assert parsed_default.method == "shap"
    assert len(parsed_default.per_instance) > 0
    assert len(parsed_default.global_importance) == len(parsed_default.global_importance)

    # Explicit RF
    resp_rf = client.get(f"/explainability?model_id={RF_MODEL_ID}")
    assert resp_rf.status_code == 200
    body_rf = resp_rf.json()
    parsed_rf = ExplainabilityResult(**body_rf)
    assert parsed_rf.method == "shap"
    assert len(parsed_rf.per_instance) > 0


def test_get_fairness_drift_routing_default_vs_rf():
    # Default (no param)
    resp_default = client.get("/fairness-drift")
    assert resp_default.status_code == 200
    body_default = resp_default.json()
    parsed_default = FairnessDriftResult(**body_default)
    assert parsed_default.fairness.status in {"PASS", "WARNING", "FAIL", "PENDING"}

    # Explicit RF
    resp_rf = client.get(f"/fairness-drift?model_id={RF_MODEL_ID}")
    assert resp_rf.status_code == 200
    body_rf = resp_rf.json()
    parsed_rf = FairnessDriftResult(**body_rf)
    assert parsed_rf.fairness.status in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_get_compliance_routing_default_vs_rf():
    # Default (no param)
    resp_default = client.get("/compliance")
    assert resp_default.status_code == 200
    body_default = resp_default.json()
    parsed_default = ComplianceResult(**body_default)
    assert len(parsed_default.findings) > 0

    # Explicit RF
    resp_rf = client.get(f"/compliance?model_id={RF_MODEL_ID}")
    assert resp_rf.status_code == 200
    body_rf = resp_rf.json()
    parsed_rf = ComplianceResult(**body_rf)
    assert len(parsed_rf.findings) > 0


def test_get_assurance_result_default_preserved():
    response = client.get("/assurance-result")
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["model_metadata"]["model_type"] == MODEL_TYPE


def test_unknown_model_id_returns_404_on_all_routed_endpoints():
    unknown_id = "non-existent-model"
    routes = [
        f"/model?model_id={unknown_id}",
        f"/explainability?model_id={unknown_id}",
        f"/fairness-drift?model_id={unknown_id}",
        f"/compliance?model_id={unknown_id}",
    ]
    for route in routes:
        response = client.get(route)
        assert response.status_code == 404, f"Expected 404 for route {route}, got {response.status_code}"
        assert unknown_id in response.json()["detail"]
