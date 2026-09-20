"""Live end-to-end proof: the synthetic bank model, served over real HTTP by
its own process/thread, works through the full ModelRegistry + RESTAdapter +
model_id routing stack (owner: Manas).

This is deliberately NOT mocked -- a real uvicorn server runs the synthetic
bank's FastAPI app on a background thread, and app.api.main's TestClient
talks to it over genuine HTTP via RESTAdapter, proving the adapter
architecture works against a model this codebase never trained in-process.

UPDATE: fairness and drift schema-hardcoding is now fixed (see
app.api.orchestration.compute_real_fairness()/compute_real_drift(), and
ModelAdapter.protected_attribute in app/models/model.py /
app/models/rest_adapter.py). Both functions now accept an optional
``adapter`` parameter:

- compute_real_fairness(model_dict, adapter=...): reads the protected
  attribute from adapter.protected_attribute instead of hardcoding
  "personal_status_and_sex". An adapter with none declared (the synthetic
  bank, today) gets a PENDING result -- never a KeyError, never a guessed
  attribute.
- compute_real_drift(model_dict, adapter=...): when the adapter's schema
  differs from German Credit's, the reference distribution comes from
  ``adapter.background_data()`` instead of always loading German Credit --
  so drift is always a same-schema comparison, never a coincidental
  column-name overlap (the previous finding: both schemas happen to have a
  column literally called "age", and drift_report() doesn't know or care
  that the two "age" columns describe unrelated populations).

Both fixes are ADAPTER-GATED: omitting ``adapter`` keeps each function's
pre-fix default behavior (German Credit reference / hardcoded protected
attribute) for full backward compatibility. All live routes that support
model_id (`/fairness-drift`, `/compliance`, and `build_assurance_result()`)
now pass ``adapter`` through, so the fix is reachable end-to-end -- but a
future caller of the bare compute_real_* functions that forgets to pass
adapter will still get the old, schema-blind default. See
test_drift_without_adapter_falls_back_to_german_credit_by_design below,
which documents that this is an intentional default-argument contract, not
a live bug.

Explainability remains a genuine, separate, NOT-yet-fixed limitation
(Step 6, Manas's ownership) -- /explainability and /compliance still fail
for the synthetic bank model, just for a different reason now (a clean
schema-mismatch error from explain(), not a fairness KeyError).
"""
from __future__ import annotations

import os
import threading
import time

import pytest
import requests
import uvicorn
from fastapi.testclient import TestClient

from app.api.main import app as main_app
from app.models.registry import reset_default_registry
from app.synthetic_bank.service import app as bank_app

TEST_PORT = 8199
SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"

client = TestClient(main_app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def live_synthetic_bank():
    """Run the synthetic bank's real FastAPI app via a genuine uvicorn server
    on a background thread, bound to a dedicated test port distinct from the
    service's own default (8100) and anything else in this repo."""
    config = uvicorn.Config(bank_app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{TEST_PORT}"
    deadline = time.time() + 10
    ready = False
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=0.5)
            if resp.status_code == 200:
                ready = True
                break
        except requests.RequestException:
            time.sleep(0.1)
    if not ready:
        pytest.fail("synthetic bank service did not start within timeout")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def synthetic_bank_registered(live_synthetic_bank):
    """Point the default model registry's synthetic bank entry at the live
    test server, then restore it afterward -- in explicit set/reset order
    (not monkeypatch) so reset_default_registry() runs strictly after the
    environment variable is restored, never before."""
    original = os.environ.get("SYNTHETIC_BANK_URL")
    os.environ["SYNTHETIC_BANK_URL"] = live_synthetic_bank
    reset_default_registry()
    try:
        yield live_synthetic_bank
    finally:
        if original is None:
            os.environ.pop("SYNTHETIC_BANK_URL", None)
        else:
            os.environ["SYNTHETIC_BANK_URL"] = original
        reset_default_registry()


# =====================================================================
# What genuinely works today
# =====================================================================


def test_model_route_works_end_to_end_against_live_synthetic_bank(synthetic_bank_registered):
    response = client.get(f"/model?model_id={SYNTHETIC_BANK_MODEL_ID}")
    assert response.status_code == 200
    body = response.json()

    assert body["model_metadata"]["model_type"] == "xgboost"
    assert len(body["predictions"]) > 0
    assert len(body["predictions"]) == len(body["probabilities"]) == len(body["feature_matrix"])
    assert set(body["predictions"]) <= {0, 1}
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])
    # No adapter-aware metrics path exists yet (see the /model model_metrics fix).
    assert body["model_metrics"] is None


def test_models_listing_includes_synthetic_bank_regardless_of_liveness():
    # Does NOT need synthetic_bank_registered -- list_models()/metadata() make
    # no network call, so this must work even if nothing is running.
    response = client.get("/models")
    assert response.status_code == 200
    model_ids = {m["model_id"] for m in response.json()}
    assert SYNTHETIC_BANK_MODEL_ID in model_ids


def test_model_health_reports_ok_when_service_is_actually_live(synthetic_bank_registered):
    response = client.get(f"/models/{SYNTHETIC_BANK_MODEL_ID}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_health_reports_unreachable_when_service_is_not_running(
    unreachable_synthetic_bank,
):
    # Deliberately without synthetic_bank_registered. The fixture points the
    # registry at a port CONFIRMED to have nothing listening, so health() makes
    # a real request that really fails. It previously assumed the service's
    # default address was free, which is false whenever the bank is running.
    response = client.get(f"/models/{SYNTHETIC_BANK_MODEL_ID}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unreachable"


# =====================================================================
# Previously-documented limitations, now CLOSED.
#
# These two tests asserted hard failures (400 / 500) caused by
# compute_real_explainability() dropping the resolved adapter, so explain()
# fell back to the German Credit schema gate. The adapter is now forwarded,
# and both routes work for this model over real HTTP.
# =====================================================================


def test_explainability_now_works_for_synthetic_bank_model(synthetic_bank_registered):
    """FIXED: was a 400 from the German Credit schema gate.

    The adapter resolved from model_id now reaches explain(), so the bank's
    own 10-feature schema and reference data are used. It has no local
    artifact, so the honest result is an APPROXIMATE black-box explanation --
    never exact TreeSHAP, despite model_type being literally "xgboost".
    """
    response = client.get(f"/explainability?model_id={SYNTHETIC_BANK_MODEL_ID}")
    assert response.status_code == 200
    body = response.json()

    assert body["model_id"] == SYNTHETIC_BANK_MODEL_ID
    assert body["model_type"] == "xgboost"
    assert body["integration_type"] == "rest"
    assert body["available"] is True

    # Black-box, because there are no model internals to inspect.
    assert body["explainer"] == "KernelExplainer"
    assert body["fidelity"] == "approximate"
    assert body["scale"] == "probability"
    assert body["fidelity"] != "exact"
    assert body["explainer"] != "TreeExplainer"

    # The bank's OWN feature space -- not German Credit's, and no silent
    # fallback to the default LR model's 20 columns.
    from app.synthetic_bank.data_generator import FEATURE_COLUMNS as BANK_FEATURES

    assert set(body["global_importance"]) == set(BANK_FEATURES)
    assert "status_checking_account" not in body["global_importance"]
    assert body["limitations"]


def test_fairness_drift_now_works_correctly_for_synthetic_bank_model(synthetic_bank_registered):
    """The fix: /fairness-drift no longer KeyErrors, and no longer fabricates
    a drift result off a coincidental column-name overlap -- it returns a
    real, honest 200 through the actual live route."""
    response = client.get(f"/fairness-drift?model_id={SYNTHETIC_BANK_MODEL_ID}")
    assert response.status_code == 200
    body = response.json()

    # Fairness: no protected attribute declared for this model -> PENDING,
    # not a crash and not a guessed/fabricated attribute.
    assert body["fairness"]["protected_attribute"] == "none_declared"
    assert body["fairness"]["status"] == "PENDING"

    # Drift: evaluated against the synthetic bank's OWN background data, so
    # ALL of its numeric features are compared -- not just a coincidental
    # "age" overlap with German Credit's unrelated population.
    from app.synthetic_bank.data_generator import NUMERIC_FEATURES

    assert set(body["drift"]["features_evaluated"]) == set(NUMERIC_FEATURES)


def test_drift_without_adapter_falls_back_to_german_credit_by_design(synthetic_bank_registered):
    """Documents an intentional default-argument contract, not a live bug:
    compute_real_drift(model_dict) with NO adapter cannot know which model
    produced model_dict, so it keeps its pre-fix default (German Credit's
    reference) for full backward compatibility with every other existing
    caller. This is why every live route that supports model_id
    (/fairness-drift, /compliance, build_assurance_result()) now explicitly
    passes adapter=adapter through -- a future caller of the bare
    compute_real_drift() that forgets to do the same will silently get this
    degraded, coincidental-overlap-prone path back, not an error.
    """
    from app.api.orchestration import compute_real_drift, compute_real_model
    from app.models import get_default_registry

    adapter = get_default_registry().get(SYNTHETIC_BANK_MODEL_ID)
    raw_model = compute_real_model(adapter=adapter)

    # Omitting adapter: old behavior, only the coincidentally-overlapping
    # "age" column is evaluated.
    without_adapter = compute_real_drift(raw_model)
    assert without_adapter["features_evaluated"] == ["age"]

    # Passing adapter: fixed behavior, the model's own full feature set.
    from app.synthetic_bank.data_generator import NUMERIC_FEATURES

    with_adapter = compute_real_drift(raw_model, adapter=adapter)
    assert set(with_adapter["features_evaluated"]) == set(NUMERIC_FEATURES)


def test_compliance_now_works_for_synthetic_bank_model(synthetic_bank_registered):
    """FIXED: was a 500 caused solely by explainability.

    /compliance was the last route still failing for this model, and the
    cause was explainability's dropped adapter -- fairness and drift had
    already been fixed. All three now receive the same adapter, so the route
    completes.
    """
    response = client.get(f"/compliance?model_id={SYNTHETIC_BANK_MODEL_ID}")
    assert response.status_code == 200

    # Isolate each contributor directly: none of the three raises now, and
    # explainability describes THIS model rather than the default LR one.
    from app.api.orchestration import (
        compute_real_explainability,
        compute_real_drift,
        compute_real_fairness,
        compute_real_model,
    )
    from app.models import get_default_registry

    adapter = get_default_registry().get(SYNTHETIC_BANK_MODEL_ID)
    raw_model = compute_real_model(adapter=adapter)

    compute_real_fairness(raw_model, adapter=adapter)  # must not raise
    compute_real_drift(raw_model, adapter=adapter)  # must not raise
    explain_res = compute_real_explainability(
        raw_model, method="shap", adapter=adapter
    )
    assert explain_res["model_id"] == SYNTHETIC_BANK_MODEL_ID
    assert explain_res["fidelity"] == "approximate"


def test_compliance_without_the_adapter_still_refuses_this_models_schema(
    synthetic_bank_registered,
):
    """The adapter is what makes it work -- omitting it must still refuse.

    Pins that the fix is the forwarded adapter and not a loosened schema
    gate: called bare, compute_real_explainability() cannot know which model
    produced model_dict, so it keeps the default German Credit contract and
    rejects this frame rather than explaining the wrong model.
    """
    from app.api.orchestration import compute_real_explainability, compute_real_model
    from app.models import get_default_registry

    adapter = get_default_registry().get(SYNTHETIC_BANK_MODEL_ID)
    raw_model = compute_real_model(adapter=adapter)

    with pytest.raises(ValueError, match="Incompatible feature schema"):
        compute_real_explainability(raw_model, method="shap")
