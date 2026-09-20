"""Full-pipeline integration: one model, one run, one identity (final backend pass).

WHAT THIS COVERS
----------------
The complete backend chain for every supported model:

    registry -> adapter -> prediction -> explainability -> fairness
    -> feature/prediction drift (monitoring) -> evidence -> RBI rule engine
    -> report

The unit suites already prove each module in isolation. What only an
integration test can catch is a module being handed the WRONG model, or a
run's artefacts drifting apart so they can no longer be gathered as one run.
Both are failures that leave every individual number looking correct.

THE TWO PROPERTIES ASSERTED THROUGHOUT
--------------------------------------
1. MODEL ISOLATION -- a request for model X is answered by model X, and never
   by the default Logistic Regression. Matching ``model_id`` alone proves
   nothing (a mislabelled result carries the right id by construction), so
   results are also compared ACROSS models and required to differ.
2. RUN IDENTITY -- every artefact of one run carries the same
   ``assurance_run_id``, so a reviewer can collect one run's output without
   relying on request timing.

Unavailable capabilities are asserted to be STRUCTURED (a stated reason)
rather than absent, zero-valued, or a crash.
"""

from __future__ import annotations

import threading
import time

import pytest
import requests
import uvicorn
from fastapi.testclient import TestClient

from app.api.main import app as main_app
from app.api.orchestration import build_assurance_result
from app.models.model import MODEL_ID, RF_MODEL_ID
from app.models.registry import get_default_registry, reset_default_registry
from app.synthetic_bank.data_generator import FEATURE_COLUMNS as BANK_FEATURES
from app.synthetic_bank.service import app as bank_app

BANK_ID = "synthetic-bank-credit-v1"
# Distinct from every other suite's port (8100/8199/8207/8213).
TEST_PORT = 8221

client = TestClient(main_app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def live_bank():
    server = uvicorn.Server(
        uvicorn.Config(bank_app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{TEST_PORT}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:  # pragma: no cover
        server.should_exit = True
        pytest.fail("synthetic bank service did not start")

    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def bank_live(live_bank, monkeypatch):
    monkeypatch.setenv("SYNTHETIC_BANK_URL", live_bank)
    reset_default_registry()
    yield
    monkeypatch.delenv("SYNTHETIC_BANK_URL", raising=False)
    reset_default_registry()


def _assurance(model_id):
    adapter = get_default_registry().get(model_id)
    return build_assurance_result(adapter=adapter)


# ===========================================================================
# Per-model end-to-end assurance flows
# ===========================================================================


@pytest.mark.parametrize(
    "model_id, model_type, explainer, scale, fidelity",
    [
        (MODEL_ID, "logistic_regression", "LinearExplainer", "log_odds", "exact"),
        (RF_MODEL_ID, "random_forest", "TreeExplainer", "probability", "exact"),
    ],
    ids=["logistic_regression", "random_forest"],
)
def test_in_process_model_full_assurance_flow(
    model_id, model_type, explainer, scale, fidelity
):
    """Every domain of one run describes the same model, under one run id."""
    result = _assurance(model_id)

    # --- run identity -------------------------------------------------------
    run_id = result["assurance_run_id"]
    assert result["model_id"] == model_id
    assert run_id

    # --- prediction ---------------------------------------------------------
    assert result["model"]["is_mock"] is False
    assert result["model"]["model_metadata"]["model_type"] == model_type
    assert result["model"]["predictions"]
    assert len(result["model"]["predictions"]) == len(result["model"]["instance_ids"])

    # --- explainability: right model, right explainer, right SCALE ----------
    explanation = result["explainability"]
    assert explanation["model_id"] == model_id
    assert explanation["model_type"] == model_type
    assert explanation["explainer"] == explainer
    assert explanation["scale"] == scale
    assert explanation["fidelity"] == fidelity
    assert explanation["available"] is True

    # --- fairness -----------------------------------------------------------
    fairness = result["fairness_drift"]["fairness"]
    assert fairness["protected_attribute"]
    assert fairness["status"] in {"PASS", "WARNING", "FAIL", "PENDING"}

    # --- feature drift ------------------------------------------------------
    drift = result["fairness_drift"]["drift"]
    assert drift["status"] in {"PASS", "WARNING", "FAIL", "PENDING"}

    # --- monitoring (feature + prediction drift) ----------------------------
    monitoring = result["monitoring"]
    assert monitoring is not None, result["monitoring_unavailable_reason"]
    assert result["monitoring_unavailable_reason"] is None
    context = monitoring["result"]["context"]
    assert context["model_id"] == model_id
    assert context["assurance_run_id"] == run_id, (
        "monitoring ran under a different assurance_run_id, so its findings "
        "cannot be gathered with the rest of this run"
    )
    assert "prediction_drift" in monitoring["result"]

    # --- evidence -----------------------------------------------------------
    evidence = monitoring["evidence"]
    assert evidence
    for record in evidence:
        assert record["model_id"] == model_id
        assert record["assurance_run_id"] == run_id
        assert record["evidence_type"]

    # --- RBI rule engine ----------------------------------------------------
    compliance = result["compliance"]
    assert compliance["model_id"] == model_id
    assert compliance["assurance_run_id"] == run_id
    assert compliance["findings"]
    for finding in compliance["findings"]:
        assert finding["model_id"] == model_id
        assert finding["assurance_run_id"] == run_id
        assert finding["status"] in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_synthetic_bank_full_assurance_flow(bank_live):
    """The REST model end to end, over real HTTP.

    It has no local artifact, so explainability must be approximate and
    fairness must be honestly unavailable -- neither may be faked to make the
    flow complete.
    """
    result = _assurance(BANK_ID)
    run_id = result["assurance_run_id"]

    assert result["model_id"] == BANK_ID
    assert result["model"]["model_metadata"]["model_type"] == "xgboost"

    # Explainability: black-box, never exact, and in the bank's own space.
    explanation = result["explainability"]
    assert explanation["model_id"] == BANK_ID
    assert explanation["explainer"] == "KernelExplainer"
    assert explanation["fidelity"] == "approximate"
    assert explanation["scale"] == "probability"
    assert set(explanation["global_importance"]) == set(BANK_FEATURES)

    # Fairness: no protected attribute is declared for this model, so the
    # result is a STRUCTURED unavailable. Inventing an attribute to make the
    # demo pass would fabricate a regulatory finding.
    fairness = result["fairness_drift"]["fairness"]
    assert fairness["protected_attribute"] == "none_declared"
    assert fairness["status"] == "PENDING"

    # Drift: evaluated against the bank's OWN reference data.
    drift = result["fairness_drift"]["drift"]
    from app.synthetic_bank.data_generator import NUMERIC_FEATURES

    assert set(drift["features_evaluated"]) == set(NUMERIC_FEATURES)

    # Monitoring reached the remote model and ran under this run's id.
    monitoring = result["monitoring"]
    assert monitoring is not None, result["monitoring_unavailable_reason"]
    assert monitoring["result"]["context"]["model_id"] == BANK_ID
    assert monitoring["result"]["context"]["assurance_run_id"] == run_id
    for record in monitoring["evidence"]:
        assert record["model_id"] == BANK_ID
        assert record["assurance_run_id"] == run_id

    # Compliance completed rather than failing the whole run.
    assert result["compliance"]["model_id"] == BANK_ID
    assert result["compliance"]["findings"]


# ===========================================================================
# Model isolation -- the property that catches a wrong-model result
# ===========================================================================


def test_three_models_produce_three_distinct_assurance_runs(bank_live):
    """No component may silently fall back to the default German Credit LR."""
    runs = {mid: _assurance(mid) for mid in (MODEL_ID, RF_MODEL_ID, BANK_ID)}

    # Distinct identities and distinct run ids.
    assert {r["model_id"] for r in runs.values()} == {MODEL_ID, RF_MODEL_ID, BANK_ID}
    assert len({r["assurance_run_id"] for r in runs.values()}) == 3

    lr, rf, bank = runs[MODEL_ID], runs[RF_MODEL_ID], runs[BANK_ID]

    # Explanations genuinely differ -- not merely differently labelled.
    assert (
        lr["explainability"]["global_importance"]
        != rf["explainability"]["global_importance"]
    )
    assert len(
        {
            lr["explainability"]["explainer"],
            rf["explainability"]["explainer"],
            bank["explainability"]["explainer"],
        }
    ) == 3

    # The bank does not even share a feature space with the other two.
    assert set(bank["explainability"]["global_importance"]) != set(
        lr["explainability"]["global_importance"]
    )

    # Predictions differ between LR and RF on the same rows.
    assert lr["model"]["predictions"] != rf["model"]["predictions"] or (
        lr["model"]["probabilities"] != rf["model"]["probabilities"]
    )


def test_evidence_from_two_models_stays_separable(bank_live):
    """Pooled monitoring evidence must remain attributable per model."""
    from app.report.generate import _route_evidence_records

    lr = _assurance(MODEL_ID)
    rf = _assurance(RF_MODEL_ID)

    pooled = lr["monitoring"]["evidence"] + rf["monitoring"]["evidence"]
    routed = _route_evidence_records(pooled)

    model_ids = {model_id for _section, model_id in routed}
    assert model_ids == {MODEL_ID, RF_MODEL_ID}
    assert (None not in model_ids), "an unattributed bucket means identity was lost"

    # Grouping recovers exactly the two original sets.
    counts = {}
    for (_section, model_id), records in routed.items():
        counts[model_id] = counts.get(model_id, 0) + len(records)
    assert counts[MODEL_ID] == len(lr["monitoring"]["evidence"])
    assert counts[RF_MODEL_ID] == len(rf["monitoring"]["evidence"])


# ===========================================================================
# Route-level: /assurance-result and /report select a model
# ===========================================================================


@pytest.mark.parametrize("model_id", [MODEL_ID, RF_MODEL_ID])
def test_assurance_result_route_selects_the_model(model_id):
    response = client.get(f"/assurance-result?model_id={model_id}")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["model_id"] == model_id
    assert body["assurance_run_id"]
    assert body["explainability"]["model_id"] == model_id
    assert body["compliance"]["model_id"] == model_id
    assert body["monitoring"]["result"]["context"]["model_id"] == model_id


def test_assurance_result_route_default_is_backward_compatible():
    """No model_id: the five original domains still present, no identity claimed."""
    response = client.get("/assurance-result")
    assert response.status_code == 200
    body = response.json()

    assert {"model", "explainability", "fairness_drift", "compliance", "note"} <= set(body)
    assert body["model_id"] is None
    # Monitoring is adapter-driven, so it is structurally unavailable here --
    # with a reason, never presented as "no drift".
    assert body["monitoring"] is None
    assert body["monitoring_unavailable_reason"]


def test_assurance_result_route_rejects_an_unknown_model():
    assert client.get("/assurance-result?model_id=nope").status_code == 404


def test_report_route_accepts_a_model_id():
    """/report must select a model and never 500.

    Without GROQ_API_KEY it returns the clearly-disclaimered mock fallback;
    the contract asserted here is that the parameter exists, is validated,
    and the route stays available either way.
    """
    response = client.get(f"/report?model_id={RF_MODEL_ID}")
    assert response.status_code == 200, response.text
    body = response.json()

    assert "report_id" in body
    assert "sections" in body
    # Identity fields exist on the contract whether or not the LLM ran.
    assert "model_id" in body
    assert "assurance_run_id" in body


def test_report_route_rejects_an_unknown_model():
    assert client.get("/report?model_id=nope").status_code == 404


# ===========================================================================
# Structured errors, not unexplained 500s
# ===========================================================================


def test_unreachable_model_service_returns_a_structured_gateway_error(
    unreachable_synthetic_bank,
):
    """A remote model that is down is an upstream failure, stated as one.

    That must surface as a 502 naming the model -- not an unexplained 500,
    and not mock numbers standing in for a real failure.

    The fixture points the registry at a port CONFIRMED to have nothing
    listening, so the real RESTAdapter makes a real connection attempt that
    really fails. This previously relied on the service's default port
    happening to be free on the developer's machine, which made the test fail
    whenever the bank was running for a demo -- while the application was
    behaving correctly.
    """
    response = client.get(f"/explainability?model_id={BANK_ID}")

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert BANK_ID in detail
    assert "could not be reached" in detail


def test_foreign_schema_metrics_are_null_not_a_failed_run(bank_live):
    """An unavailable capability must not take down the assurance run.

    German Credit's held-out split has no meaning for the bank's schema, so
    ``evaluate_current_model()`` raises for it. Before this pass that
    exception propagated out of ``build_assurance_result()`` and the entire
    run failed -- including the explainability, fairness, drift and compliance
    findings that had already computed successfully.

    The honest outcome is null metrics: the capability is absent, and it is
    never filled with another model's numbers.
    """
    result = _assurance(BANK_ID)

    assert result["model"]["model_metrics"] is None
    # Everything else still computed.
    assert result["explainability"]["available"] is True
    assert result["compliance"]["findings"]
    assert result["model"]["predictions"]


def test_monitoring_degrades_without_failing_the_whole_run(
    unreachable_synthetic_bank,
):
    """An unreachable monitoring lane must not delete the other findings.

    Monitoring cannot score its windows against an unreachable service -- but
    explainability/fairness/compliance for the models that DID compute must
    still be returned, and the monitoring gap must be stated.

    The fixture guarantees the service is unreachable (confirmed-closed port)
    rather than assuming the default port is free.
    """
    from app.api.orchestration import _monitoring_for_assurance

    adapter = get_default_registry().get(BANK_ID)
    monitoring, reason = _monitoring_for_assurance(
        adapter, {}, model_id=BANK_ID, assurance_run_id="run-degrade-1"
    )

    assert monitoring is None
    assert reason and BANK_ID in reason
    # Crucially: it says no drift was MEASURED, not that there is no drift.
    assert "NOT a statement that the model is stable" in reason
