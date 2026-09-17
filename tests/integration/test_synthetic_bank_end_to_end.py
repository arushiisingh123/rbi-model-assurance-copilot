"""Live end-to-end proof: the synthetic bank model, served over real HTTP by
its own process/thread, works through the full ModelRegistry + RESTAdapter +
model_id routing stack (owner: Manas).

This is deliberately NOT mocked -- a real uvicorn server runs the synthetic
bank's FastAPI app on a background thread, and app.api.main's TestClient
talks to it over genuine HTTP via RESTAdapter, proving the adapter
architecture works against a model this codebase never trained in-process.

Known, EXPECTED current limitation (tracked separately, not fixed here):
app.api.orchestration.py's fairness/drift/explainability functions are
hardcoded to the German Credit schema and dataset, so /explainability,
/fairness-drift, and /compliance do not work correctly for a model with a
different schema yet. This file explicitly asserts the current failure mode
for each route rather than silently leaving it undiscovered -- it is a
documented boundary, not a bug this task fixes.

IMPORTANT, more severe sub-finding: compute_real_drift() does NOT always
fail loudly for a different schema -- it silently "succeeds" whenever the
current model's schema happens to share a column NAME with German Credit's
(here: both schemas have a column literally called "age", by coincidence).
It then reports a PSI/KS "drift" result comparing German Credit's age
distribution against the synthetic bank's age distribution as if they
measured the same population -- a fabricated-looking but real-shaped
finding, not a crash. This is currently masked in the live /fairness-drift
route only because compute_real_fairness() raises first and aborts the
request before compute_real_drift() ever runs (see
test_fairness_drift_currently_fails_for_synthetic_bank_model below) --
fixing fairness's hardcoding WITHOUT also fixing drift's would silently
expose this. test_drift_silently_fabricates_result_on_coincidental_column_overlap
proves this directly, bypassing the route, so it cannot be masked by ordering.
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


def test_fairness_drift_currently_fails_for_synthetic_bank_model(synthetic_bank_registered):
    # Fails via compute_real_fairness()'s hardcoded "personal_status_and_sex"
    # KeyError, which raises before compute_real_drift() ever runs in this
    # route -- see test_drift_silently_fabricates_result_on_coincidental_column_overlap
    # for what would happen if only the fairness half were fixed.
    response = client.get(f"/fairness-drift?model_id={SYNTHETIC_BANK_MODEL_ID}")
    assert response.status_code == 500


def test_drift_silently_fabricates_result_on_coincidental_column_overlap(synthetic_bank_registered):
    """Tripwire: compute_real_drift() must not be considered "safe" for a
    different-schema model just because it doesn't crash. Called directly
    (bypassing /fairness-drift, whose fairness-first ordering currently masks
    this) it silently compares German Credit's "age" column against the
    synthetic bank's "age" column -- two unrelated populations that happen to
    share a column name -- and reports a real-shaped PSI/KS/status result as
    if it were a genuine finding. If this test ever starts failing because
    compute_real_drift() now raises instead, that is the fix landing
    correctly (tracked separately) -- update this test then, don't just
    delete it.
    """
    from app.api.orchestration import compute_real_drift, compute_real_model
    from app.models import get_default_registry

    adapter = get_default_registry().get(SYNTHETIC_BANK_MODEL_ID)
    raw_model = compute_real_model(adapter=adapter)
    drift_result = compute_real_drift(raw_model)

    # It "succeeded" -- but only on the one coincidentally-overlapping column.
    assert drift_result["features_evaluated"] == ["age"]
    assert drift_result["status"] in {"PASS", "WARNING", "FAIL"}
    assert isinstance(drift_result["psi"], float)


def test_compliance_currently_fails_for_synthetic_bank_model(synthetic_bank_registered):
    response = client.get(f"/compliance?model_id={SYNTHETIC_BANK_MODEL_ID}")
    assert response.status_code == 500
