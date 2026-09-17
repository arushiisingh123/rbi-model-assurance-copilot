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


# ===========================================================================
# POST /score-batch (Phase 6)
#
# The endpoint exists so perturbation-based explainability is feasible against
# this model over HTTP. Its contract is positional: predictions[i] and
# probabilities[i] describe instances[i]. Everything below pins that.
# ===========================================================================


def _batch_rows(n=6, seed=7):
    from app.synthetic_bank.data_generator import FEATURE_COLUMNS, generate_customers

    return generate_customers(n=n, random_state=seed)[FEATURE_COLUMNS].to_dict(
        "records"
    )


def test_score_batch_returns_one_result_per_row():
    rows = _batch_rows(6)
    response = client.post("/score-batch", json={"instances": rows})

    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == 6
    assert len(body["predictions"]) == 6
    assert len(body["probabilities"]) == 6
    assert all(p in (0, 1) for p in body["predictions"])
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])


def test_score_batch_equals_repeated_single_scoring():
    """Requirement 30: batching is an optimisation, not a different answer.

    Compared exactly, not approximately -- the same model scores both ways,
    so any difference would mean the batch path reshaped the input.
    """
    rows = _batch_rows(6)

    batch = client.post("/score-batch", json={"instances": rows}).json()
    singles = [client.post("/score", json=row).json() for row in rows]

    assert batch["predictions"] == [s["prediction"] for s in singles]
    assert batch["probabilities"] == [s["probability"] for s in singles]


def test_score_batch_preserves_the_submitted_order():
    """Requirement 31: results are joined by position, so order is the key.

    Scoring the same rows reversed must reverse the results. If the endpoint
    sorted, grouped or deduplicated internally, this would not hold -- and
    every applicant's score would silently belong to someone else.
    """
    rows = _batch_rows(6)

    forward = client.post("/score-batch", json={"instances": rows}).json()
    backward = client.post("/score-batch", json={"instances": rows[::-1]}).json()

    assert backward["probabilities"] == forward["probabilities"][::-1]
    assert backward["predictions"] == forward["predictions"][::-1]


def test_score_batch_does_not_deduplicate_identical_rows():
    """Two identical applicants are two results, not one.

    Deduplication would shorten the response and break the positional join.
    """
    row = _batch_rows(2)[0]
    response = client.post("/score-batch", json={"instances": [row, row, row]})

    body = response.json()
    assert body["n_rows"] == 3
    assert len(body["probabilities"]) == 3
    assert len(set(body["probabilities"])) == 1


def test_score_batch_rejects_an_empty_batch():
    """An empty batch is a caller error, not an empty result."""
    assert client.post("/score-batch", json={"instances": []}).status_code == 422


def test_score_batch_rejects_an_unknown_field():
    rows = _batch_rows(2)
    rows[0]["scraped_postcode"] = "560001"

    assert client.post("/score-batch", json={"instances": rows}).status_code == 422


def test_score_batch_rejects_a_missing_feature():
    rows = _batch_rows(2)
    del rows[1]["annual_income"]

    assert client.post("/score-batch", json={"instances": rows}).status_code == 422


def test_score_batch_rejects_a_blank_categorical_and_names_the_row():
    """One bad row fails the whole request rather than being dropped."""
    rows = _batch_rows(3)
    rows[2]["employment_type"] = ""

    response = client.post("/score-batch", json={"instances": rows})

    assert response.status_code == 422
    assert "instances[2]" in str(response.json()["detail"])
    assert "employment_type" in str(response.json()["detail"])


def test_score_batch_reports_the_model_version():
    rows = _batch_rows(2)
    body = client.post("/score-batch", json={"instances": rows}).json()

    assert body["model_version"] == client.get("/metadata").json()["model_version"]
