"""Regression: the API explains the exact model it was asked for (owner: Manas).

THE DEFECT THIS FILE GUARDS
---------------------------
Every route that supports ``?model_id=`` resolved an adapter and scored
predictions through it, then dropped that adapter before explainability. So
``explain()`` fell back to loading the DEFAULT Logistic Regression artifact,
and a caller could request Random Forest explanations, receive the Logistic
Regression's SHAP values, and have them stamped with the Random Forest's
``model_id`` -- confident, specific, and about the wrong model.

WHY THESE TESTS GO THROUGH THE ROUTE
------------------------------------
``tests/explainability/`` already proves ``explain(adapter=...)`` is correct,
and ``test_model_agnostic.py`` proves the orchestration function forwards an
adapter. Neither would have caught a route that resolves an adapter and then
forgets to pass it -- which is precisely what the defect was. These tests
therefore drive the real HTTP routes end to end.

Identity is checked TWO ways, because matching ``model_id`` alone proves
nothing: a mislabelled explanation has the right id by construction. So each
model's explanation is also asserted to differ from the others' and to carry
the explainer/scale its own structure implies.
"""

from __future__ import annotations

import threading
import time

import pytest
import requests
import uvicorn
from fastapi.testclient import TestClient

from app.api.main import app as main_app
from app.models.model import MODEL_ID, RF_MODEL_ID
from app.models.registry import get_default_registry, reset_default_registry
from app.synthetic_bank.data_generator import FEATURE_COLUMNS as BANK_FEATURES
from app.synthetic_bank.service import app as bank_app

# Distinct from 8100 (service default), 8199 (bank e2e suite) and 8207
# (adapter explainability suite) so suites can run in any order.
TEST_PORT = 8213
SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"

client = TestClient(main_app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def live_bank():
    """A real uvicorn server for the synthetic bank, on its own port."""
    server = uvicorn.Server(
        uvicorn.Config(bank_app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{TEST_PORT}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:  # pragma: no cover - only on a broken environment
        server.should_exit = True
        pytest.fail("synthetic bank service did not start within timeout")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def bank_registered(live_bank, monkeypatch):
    """Point the registry's bank entry at the live test server."""
    monkeypatch.setenv("SYNTHETIC_BANK_URL", live_bank)
    reset_default_registry()
    yield
    monkeypatch.delenv("SYNTHETIC_BANK_URL", raising=False)
    reset_default_registry()


def _explain(model_id=None, method="shap"):
    query = f"?method={method}"
    if model_id is not None:
        query += f"&model_id={model_id}"
    response = client.get(f"/explainability{query}")
    assert response.status_code == 200, response.text
    return response.json()


# ===========================================================================
# Requirement 10: each request gets ITS OWN model's explanation
# ===========================================================================


def test_lr_request_returns_the_lr_explanation():
    body = _explain(MODEL_ID)

    assert body["model_id"] == MODEL_ID
    assert body["model_type"] == "logistic_regression"
    assert body["available"] is True
    # Linear structure -> exact additive decomposition of the decision
    # function, so the contributions are LOG-ODDS.
    assert body["explainer"] == "LinearExplainer"
    assert body["fidelity"] == "exact"
    assert body["scale"] == "log_odds"
    assert len(body["global_importance"]) == 20


def test_rf_request_returns_the_rf_explanation():
    body = _explain(RF_MODEL_ID)

    assert body["model_id"] == RF_MODEL_ID
    assert body["model_type"] == "random_forest"
    assert body["available"] is True
    # Tree structure -> exact, but additive to predict_proba, so the
    # contributions are on the PROBABILITY scale, not log-odds.
    assert body["explainer"] == "TreeExplainer"
    assert body["fidelity"] == "exact"
    assert body["scale"] == "probability"
    assert len(body["global_importance"]) == 20


def test_synthetic_bank_request_returns_the_bank_explanation(bank_registered):
    body = _explain(SYNTHETIC_BANK_MODEL_ID)

    assert body["model_id"] == SYNTHETIC_BANK_MODEL_ID
    assert body["model_type"] == "xgboost"
    assert body["integration_type"] == "rest"
    assert body["available"] is True
    # No local artifact to inspect, so the only honest answer is an
    # approximate black-box estimate through the prediction interface.
    assert body["explainer"] == "KernelExplainer"
    assert body["fidelity"] == "approximate"
    assert body["scale"] == "probability"
    # Its OWN 10-feature space, not German Credit's 20.
    assert set(body["global_importance"]) == set(BANK_FEATURES)


def test_the_three_models_do_not_share_one_explanation(bank_registered):
    """The assertion that actually catches the defect.

    Matching model_ids prove nothing on their own -- a mislabelled
    explanation carries the right id by construction. Three genuinely
    different explanations is the evidence that three different models were
    explained.
    """
    lr = _explain(MODEL_ID)
    rf = _explain(RF_MODEL_ID)
    bank = _explain(SYNTHETIC_BANK_MODEL_ID)

    assert lr["global_importance"] != rf["global_importance"]
    assert lr["per_instance"] != rf["per_instance"]

    # The bank does not even share a feature space with the other two.
    assert set(bank["global_importance"]).isdisjoint(
        set(lr["global_importance"]) - {"age"}
    )

    assert len({lr["explainer"], rf["explainer"], bank["explainer"]}) == 3


def test_requested_model_matches_the_model_used_for_prediction(bank_registered):
    """Requirement 8: explanation and prediction must describe one model.

    /model and /explainability resolve the adapter independently, so this
    pins that they land on the same model rather than drifting apart.
    """
    for model_id, expected_type in (
        (MODEL_ID, "logistic_regression"),
        (RF_MODEL_ID, "random_forest"),
        (SYNTHETIC_BANK_MODEL_ID, "xgboost"),
    ):
        prediction = client.get(f"/model?model_id={model_id}")
        assert prediction.status_code == 200, prediction.text
        predicted_type = prediction.json()["model_metadata"]["model_type"]

        explanation = _explain(model_id)

        assert predicted_type == expected_type
        assert explanation["model_type"] == predicted_type, (
            f"{model_id}: prediction used {predicted_type} but the "
            f"explanation describes {explanation['model_type']}"
        )
        assert explanation["model_id"] == model_id


# ===========================================================================
# Truthful metadata and honest scale reporting
# ===========================================================================


def test_scale_differs_between_two_shap_explanations():
    """Both are method='shap'; their units are NOT the same.

    A consumer inferring the scale from the method name would label one of
    these wrongly. The scale therefore has to travel on the response, which
    is why ExplainabilityResult declares it -- an undeclared field is dropped
    silently by the response model.
    """
    lr = _explain(MODEL_ID)
    rf = _explain(RF_MODEL_ID)

    assert lr["method"] == rf["method"] == "shap"
    assert lr["scale"] != rf["scale"]
    assert {lr["scale"], rf["scale"]} == {"log_odds", "probability"}


def test_api_only_model_is_never_described_as_exact(bank_registered):
    """Requirement: an API-backed model must not claim model-internal SHAP.

    Its model_type is literally "xgboost", so a name-based router would have
    claimed exact TreeSHAP for a model whose internals are unreachable.
    """
    body = _explain(SYNTHETIC_BANK_MODEL_ID)

    assert body["fidelity"] != "exact"
    assert body["explainer"] not in ("TreeExplainer", "LinearExplainer")
    assert body["limitations"], "an approximate explanation must state why"
    joined = " ".join(body["limitations"]).lower()
    assert "approximation" in joined or "not exact" in joined


def test_lime_also_routes_per_model():
    """LIME needs only a probability callable, so it must distinguish models too."""
    lr = _explain(MODEL_ID, method="lime")
    rf = _explain(RF_MODEL_ID, method="lime")

    assert lr["model_id"] == MODEL_ID
    assert rf["model_id"] == RF_MODEL_ID
    assert lr["fidelity"] == rf["fidelity"] == "surrogate"
    assert lr["scale"] == rf["scale"] == "probability"
    assert lr["global_importance"] != rf["global_importance"]


# ===========================================================================
# Backward compatibility of the default (no model_id) request
# ===========================================================================


def test_no_model_id_keeps_the_original_response_shape():
    """The default request must be unchanged for existing consumers.

    The additive fields are present-but-null rather than absent (Pydantic
    serializes declared optionals), and critically no identity is claimed for
    a model the caller never named.
    """
    body = _explain()

    populated = {key for key, value in body.items() if value is not None}
    assert populated == {"method", "per_instance", "global_importance", "is_mock"}
    assert body["is_mock"] is False
    assert body["model_id"] is None
    assert body["scale"] is None
    assert len(body["global_importance"]) == 20


def test_unknown_model_id_is_a_404_not_a_default_explanation():
    """A typo must never silently return the default model's explanation."""
    response = client.get("/explainability?model_id=not-a-registered-model")

    assert response.status_code == 404


def test_registry_models_are_all_explainable_or_honestly_unavailable(bank_registered):
    """Every registered model must yield a usable answer, never a crash.

    Model-agnostic means an unfamiliar model is explained approximately or
    declared unavailable with a reason -- not rejected and not crashed.
    """
    for metadata in get_default_registry().list_models():
        body = _explain(metadata["model_id"])

        assert body["model_id"] == metadata["model_id"]
        if body["available"]:
            assert body["explainer"] and body["scale"] and body["fidelity"]
            assert body["global_importance"]
        else:
            assert body["global_importance"] == {}
            assert body["limitations"]
