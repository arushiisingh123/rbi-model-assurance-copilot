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


def test_model_health_reports_unreachable_when_service_is_not_running():
    # Deliberately without synthetic_bank_registered: SYNTHETIC_BANK_URL is
    # unset here, so the registry points at the default (unreachable) address.
    reset_default_registry()
    response = client.get(f"/models/{SYNTHETIC_BANK_MODEL_ID}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unreachable"


# =====================================================================
# Documented current limitations -- NOT fixed in this task, asserted so
# they're a known boundary rather than a silent surprise later.
# =====================================================================


def test_explainability_currently_fails_for_synthetic_bank_model(synthetic_bank_registered):
    # explain() validates the incoming feature schema against German Credit's
    # raw schema before it ever reaches load_fitted_model()/background data,
    # so this is a clean 400 (schema mismatch), not a 500 -- still a hard
    # failure, just caught earlier and more informatively than expected.
    response = client.get(f"/explainability?model_id={SYNTHETIC_BANK_MODEL_ID}")
    assert response.status_code == 400


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


def test_compliance_still_fails_for_synthetic_bank_model_but_now_via_explainability(
    synthetic_bank_registered,
):
    """/compliance still fails for the synthetic bank model -- but now
    ONLY because of explainability's genuine, separate, not-yet-fixed
    schema mismatch (Step 6, Manas's ownership), not because of fairness's
    now-fixed KeyError. Confirms the fairness/drift fix didn't just move
    the failure around without checking why it still fails."""
    response = client.get(f"/compliance?model_id={SYNTHETIC_BANK_MODEL_ID}")
    assert response.status_code == 500

    # Isolate the cause directly: fairness and drift no longer raise for
    # this model; explainability still does.
    from app.api.orchestration import (
        compute_real_explainability,
        compute_real_fairness,
        compute_real_model,
    )
    from app.models import get_default_registry

    adapter = get_default_registry().get(SYNTHETIC_BANK_MODEL_ID)
    raw_model = compute_real_model(adapter=adapter)

    compute_real_fairness(raw_model, adapter=adapter)  # must not raise
    with pytest.raises(ValueError):
        compute_real_explainability(raw_model, method="shap")
