"""HTTP contract for the monitoring router (owner: Arushi).

Exercises ``GET``/``POST /monitoring`` through the real FastAPI app and the real
model registry. Nothing is mocked: a request here runs a genuine monitoring
pass over real adapters, which is the point -- the router's job is to make the
monitoring lane reachable, and a mocked test would not prove that.

The analytical numbers are not re-asserted; their own suites own them. What is
asserted is the contract: identity survives the HTTP boundary, window metadata
survives it, an undeclared protected attribute stays undeclared, and bad input
produces a meaningful status code rather than a 500.
"""
from fastapi.testclient import TestClient

from app.api.main import app
from app.config.thresholds import VALID_STATUSES
from app.models.model import MODEL_ID, RF_MODEL_ID
from app.models.preprocessing import (
    DEFAULT_DATASET_PATH,
    load_dataset,
    preprocess,
    split_data,
)
from app.monitoring.evidence import MONITORING_EVIDENCE_TYPES

client = TestClient(app, raise_server_exceptions=False)


def _rows(n: int = 120, *, tail: bool = False):
    """Real raw feature rows, used only as a caller-supplied window payload."""
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    train, test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    frame = test if tail else train
    return frame.head(n).to_dict("records")


# ---------------------------------------------------------------------------
# GET /monitoring
# ---------------------------------------------------------------------------


def test_the_monitoring_endpoint_exists_and_returns_a_result():
    response = client.get("/monitoring")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"result", "evidence", "protected_attribute", "guidance"}
    assert body["result"]["monitoring_status"] in VALID_STATUSES


def test_the_response_carries_the_requested_models_identity():
    response = client.get(f"/monitoring?model_id={RF_MODEL_ID}")

    assert response.status_code == 200
    context = response.json()["result"]["context"]
    assert context["model_id"] == RF_MODEL_ID
    assert context["model_id"] != MODEL_ID
    assert context["assurance_run_id"]


def test_two_models_return_different_runs_through_the_same_route():
    """A routed model_id must change what is measured, not just the label."""
    lr = client.get(f"/monitoring?model_id={MODEL_ID}").json()
    rf = client.get(f"/monitoring?model_id={RF_MODEL_ID}").json()

    assert lr["result"]["context"]["model_id"] != rf["result"]["context"]["model_id"]
    # Prediction drift is the model-specific channel: the two models' own
    # outputs differ, so their prediction drift must differ too.
    assert (
        lr["result"]["prediction_drift"]["label_psi"]
        != rf["result"]["prediction_drift"]["label_psi"]
    )


def test_each_request_mints_its_own_assurance_run_id():
    first = client.get("/monitoring").json()["result"]["context"]["assurance_run_id"]
    second = client.get("/monitoring").json()["result"]["context"]["assurance_run_id"]

    assert first != second


def test_all_three_channels_and_window_metadata_are_exposed():
    body = client.get("/monitoring").json()
    result = body["result"]

    assert set(result["channel_status"]) == {
        "feature_drift",
        "prediction_drift",
        "fairness",
    }
    for side in ("reference", "current"):
        window = result["windows"][side]
        assert window["window_id"]
        # Unstated provenance must survive as null, never as "observed".
        assert window["provenance"] is None
        assert window["record_count"] > 0


def test_every_monitoring_evidence_type_is_exposed():
    body = client.get("/monitoring").json()

    types = [record["evidence_type"] for record in body["evidence"]]
    assert types == list(MONITORING_EVIDENCE_TYPES)
    for record in body["evidence"]:
        assert record["model_id"]
        assert record["assurance_run_id"] == body["result"]["context"][
            "assurance_run_id"
        ]


def test_score_availability_is_reported_explicitly():
    prediction_drift = client.get("/monitoring").json()["result"]["prediction_drift"]

    assert prediction_drift["score_availability"] in {
        "computed",
        "unavailable_no_scores",
    }


def test_an_unknown_model_id_is_404():
    response = client.get("/monitoring?model_id=no-such-model")

    assert response.status_code == 404


def test_an_explicit_protected_attribute_override_is_honoured():
    response = client.get(f"/monitoring?model_id={MODEL_ID}&protected_attribute=housing")

    assert response.status_code == 200
    body = response.json()
    assert body["protected_attribute"] == "housing"
    assert body["result"]["fairness"]["protected_attribute"] == "housing"


def test_an_attribute_absent_from_the_feature_space_is_pending_not_a_crash():
    response = client.get("/monitoring?protected_attribute=not_a_column")

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["fairness"] is None
    assert body["result"]["channel_status"]["fairness"] == "PENDING"


# ---------------------------------------------------------------------------
# POST /monitoring -- caller-supplied windows
# ---------------------------------------------------------------------------


def test_caller_supplied_windows_are_monitored():
    response = client.post(
        "/monitoring",
        json={
            "model_id": MODEL_ID,
            "reference": {"records": _rows(120), "window_id": "2026-Q2"},
            "current": {"records": _rows(120, tail=True), "window_id": "2026-Q3"},
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["reference_window_id"] == "2026-Q2"
    assert result["current_window_id"] == "2026-Q3"
    assert result["windows"]["reference"]["record_count"] == 120
    assert result["windows"]["current"]["record_count"] == 120


def test_declared_provenance_and_period_survive_the_http_boundary():
    response = client.post(
        "/monitoring",
        json={
            "model_id": MODEL_ID,
            "current": {
                "records": _rows(80, tail=True),
                "provenance": "observed",
                "window_start": "2026-07-01T00:00:00Z",
                "window_end": "2026-09-30T00:00:00Z",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    current = body["result"]["windows"]["current"]
    assert current["provenance"] == "observed"
    assert current["window_start"].startswith("2026-07-01")
    assert current["window_end"].startswith("2026-09-30")
    # And it reaches the evidence, not just the result.
    assert body["evidence"][0]["current_window"]["provenance"] == "observed"


def test_omitting_a_window_falls_back_to_the_models_own_default():
    response = client.post(
        "/monitoring",
        json={"model_id": MODEL_ID, "current": {"records": _rows(60, tail=True)}},
    )

    assert response.status_code == 200
    windows = response.json()["result"]["windows"]
    assert windows["current"]["record_count"] == 60
    # Reference omitted -> the adapter's own background data, not a guess.
    assert windows["reference"]["record_count"] > 0


def test_an_unknown_provenance_is_422_with_the_allowed_values():
    response = client.post(
        "/monitoring",
        json={"model_id": MODEL_ID, "current": {"provenance": "real"}},
    )

    assert response.status_code == 422
    assert "observed" in response.json()["detail"]


def test_reversed_window_bounds_are_422_not_500():
    response = client.post(
        "/monitoring",
        json={
            "model_id": MODEL_ID,
            "current": {
                "window_start": "2026-09-30T00:00:00Z",
                "window_end": "2026-07-01T00:00:00Z",
            },
        },
    )

    assert response.status_code == 422


def test_an_empty_record_list_is_422():
    """Distinguished from omitting the key, which legitimately means 'default'."""
    response = client.post(
        "/monitoring", json={"model_id": MODEL_ID, "current": {"records": []}}
    )

    assert response.status_code == 422


def test_an_unparseable_window_bound_is_422():
    response = client.post(
        "/monitoring",
        json={"model_id": MODEL_ID, "current": {"window_start": "not-a-date"}},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# No German Credit assumptions in the router itself
# ---------------------------------------------------------------------------


def test_the_router_names_no_dataset_feature_or_protected_attribute():
    """A monitoring route must carry no model-specific knowledge of its own."""
    from pathlib import Path

    source = Path("app/api/monitoring.py").read_text(encoding="utf-8")
    for forbidden in (
        "german_credit",
        "personal_status_and_sex",
        "DEFAULT_DATASET_PATH",
        "LABEL_SEMANTICS",
        "FEATURE_COLUMNS",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# External-model failure semantics
# ---------------------------------------------------------------------------


def test_an_unreachable_external_model_is_502_not_500(monkeypatch):
    """A model living in another process being down is not OUR server error.

    Every other model-facing route already returns 502 for this; /monitoring
    returned a bare 500, which tells an operator to go read our traceback
    instead of starting the model service.

    Caught by TYPE: RESTAdapterError specifically, so a genuine programming
    error still surfaces as a 500 rather than being relabelled as somebody
    else's outage.
    """
    from app.models.rest_adapter import RESTAdapterError

    def _unreachable(*args, **kwargs):
        raise RESTAdapterError("HTTP request error scoring row 0: connection refused")

    monkeypatch.setattr("app.api.monitoring.run_monitoring", _unreachable)

    response = client.get("/monitoring?model_id=synthetic-bank-credit-v1")

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "could not be reached" in detail
    assert "synthetic-bank-credit-v1" in detail


def test_an_internal_error_is_not_disguised_as_an_external_outage(monkeypatch):
    """The guard must not become a catch-all that hides our own defects."""

    def _boom(*args, **kwargs):
        raise TypeError("a genuine programming error")

    monkeypatch.setattr("app.api.monitoring.run_monitoring", _boom)

    response = client.get("/monitoring")

    assert response.status_code != 502


# ---------------------------------------------------------------------------
# Investigation guidance (surfaced in the UI as "What to look into")
# ---------------------------------------------------------------------------


def test_guidance_is_returned_for_channels_needing_attention():
    """The response says what to investigate, not just that something moved.

    Guidance is a lookup on statuses the analytical modules already assigned
    (app/report/guidance.py). It is fixed text, so it can never disagree with
    the result it accompanies.
    """
    body = client.get("/monitoring").json()
    channel_status = body["result"]["channel_status"]

    actionable = {
        channel
        for channel, status in channel_status.items()
        if status in ("WARNING", "FAIL")
    }
    guided = {entry["channel"] for entry in body["guidance"]}

    assert guided == actionable
    for entry in body["guidance"]:
        # It echoes the channel's existing status -- never its own verdict.
        assert entry["status"] == channel_status[entry["channel"]]
        assert entry["means"]
        assert entry["investigate"]


def test_guidance_never_covers_a_passing_or_unmeasured_channel():
    """A PASS needs no action, and a PENDING channel was never measured."""
    body = client.get("/monitoring").json()
    channel_status = body["result"]["channel_status"]

    for entry in body["guidance"]:
        assert channel_status[entry["channel"]] not in ("PASS", "PENDING")


def test_guidance_names_no_cause_and_recommends_no_retraining():
    """It proposes what to check. It must never assert why, or blame anyone."""
    body = client.get("/monitoring").json()
    text = " ".join(
        entry["means"] + " " + " ".join(entry["investigate"])
        for entry in body["guidance"]
    ).lower()

    for forbidden in ("retrain", "root cause is", "caused by", "employee error"):
        assert forbidden not in text
