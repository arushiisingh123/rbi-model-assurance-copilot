"""Unit tests for RESTAdapter and RESTAdapterError (owner: Manas).

Verifies:
- predict() on a multi-row DataFrame makes sequential POST calls and returns 1-D int numpy array
- predict_proba() returns 1-D float numpy array when supported
- predict_proba() raises ProbabilityCapabilityUnavailable when unsupported without making HTTP requests
- predict() and predict_proba() error handling for timeouts, connection errors, HTTP errors, and malformed payloads
- health() check resilience (returns dict on failure without raising)
- load_fitted_model() raises NotImplementedError
- background_data() behavior
- Custom input_schema and capabilities overriding base class derivations
- metadata() serialization
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from app.models import (
    ModelAdapter,
    RESTAdapter,
    RESTAdapterError,
)
from app.models.model import ProbabilityCapabilityUnavailable


@pytest.fixture
def sample_features():
    return ["feature_a", "feature_b", "feature_c"]


@pytest.fixture
def sample_schema():
    return {
        "feature_a": {"type": "numeric"},
        "feature_b": {"type": "categorical"},
        "feature_c": {"type": "numeric"},
    }


@pytest.fixture
def sample_capabilities():
    return {
        "predict_proba": True,
        "batch": True,
        "explainability": False,
    }


@pytest.fixture
def rest_adapter(sample_features, sample_schema, sample_capabilities):
    return RESTAdapter(
        model_id="remote-credit-model",
        model_version="1.2.0",
        model_type="external_xgboost",
        endpoint_url="http://mock-bank.local/api/v1/",
        feature_names=sample_features,
        input_schema=sample_schema,
        capabilities=sample_capabilities,
        timeout=5.0,
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"feature_a": 10.5, "feature_b": "category_1", "feature_c": 100},
        {"feature_a": 20.0, "feature_b": "category_2", "feature_c": 200},
        {"feature_a": 30.5, "feature_b": "category_1", "feature_c": 300},
    ])


# =====================================================================
# 1. Prediction Tests (predict & predict_proba)
# =====================================================================


@patch("requests.post")
def test_predict_multi_row_success(mock_post, rest_adapter, sample_df):
    responses = [
        MagicMock(status_code=200, json=lambda: {"prediction": 0, "probability": 0.15}),
        MagicMock(status_code=200, json=lambda: {"prediction": 1, "probability": 0.85}),
        MagicMock(status_code=200, json=lambda: {"prediction": 0, "probability": 0.30}),
    ]
    mock_post.side_effect = responses

    preds = rest_adapter.predict(sample_df)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)
    assert list(preds) == [0, 1, 0]
    assert mock_post.call_count == 3

    # Check POST payload for each row
    expected_records = sample_df.to_dict("records")
    for i, call in enumerate(mock_post.call_args_list):
        args, kwargs = call
        assert args[0] == "http://mock-bank.local/api/v1/score"
        assert kwargs["json"] == expected_records[i]
        assert kwargs["timeout"] == 5.0


@patch("requests.post")
def test_predict_proba_multi_row_success(mock_post, rest_adapter, sample_df):
    responses = [
        MagicMock(status_code=200, json=lambda: {"prediction": 0, "probability": 0.15}),
        MagicMock(status_code=200, json=lambda: {"prediction": 1, "probability": 0.85}),
        MagicMock(status_code=200, json=lambda: {"prediction": 0, "probability": 0.30}),
    ]
    mock_post.side_effect = responses

    probs = rest_adapter.predict_proba(sample_df)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (3,)
    assert np.allclose(probs, [0.15, 0.85, 0.30])
    assert mock_post.call_count == 3


@patch("requests.post")
def test_predict_proba_unsupported_raises_probability_capability_unavailable(
    mock_post, sample_features, sample_schema, sample_df
):
    adapter_no_proba = RESTAdapter(
        model_id="remote-no-proba",
        model_version="1.0.0",
        model_type="external_linear",
        endpoint_url="http://mock-bank.local",
        feature_names=sample_features,
        input_schema=sample_schema,
        capabilities={"predict_proba": False, "batch": True, "explainability": False},
    )

    assert adapter_no_proba.supports_probability is False

    with pytest.raises(ProbabilityCapabilityUnavailable) as exc_info:
        adapter_no_proba.predict_proba(sample_df)

    assert "remote-no-proba" in str(exc_info.value)
    assert mock_post.call_count == 0


# =====================================================================
# 2. Error Handling Tests
# =====================================================================


@patch("requests.post")
def test_predict_connection_error_raises_rest_adapter_error(mock_post, rest_adapter, sample_df):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    with pytest.raises(RESTAdapterError) as exc_info:
        rest_adapter.predict(sample_df)

    assert "row 0" in str(exc_info.value).lower()
    assert "Connection refused" in str(exc_info.value)


@patch("requests.post")
def test_predict_timeout_raises_rest_adapter_error(mock_post, rest_adapter, sample_df):
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

    with pytest.raises(RESTAdapterError) as exc_info:
        rest_adapter.predict(sample_df)

    assert "row 0" in str(exc_info.value).lower()
    assert "timed out" in str(exc_info.value)


@patch("requests.post")
def test_predict_non_2xx_status_raises_rest_adapter_error(mock_post, rest_adapter, sample_df):
    mock_resp = MagicMock(status_code=500, text="Internal Server Error")
    mock_post.return_value = mock_resp

    with pytest.raises(RESTAdapterError) as exc_info:
        rest_adapter.predict(sample_df)

    assert "row 0" in str(exc_info.value).lower()
    assert "500" in str(exc_info.value)


@patch("requests.post")
def test_predict_missing_prediction_field_raises_rest_adapter_error(mock_post, rest_adapter, sample_df):
    mock_resp = MagicMock(status_code=200, json=lambda: {"probability": 0.85})
    mock_post.return_value = mock_resp

    with pytest.raises(RESTAdapterError) as exc_info:
        rest_adapter.predict(sample_df)

    assert "row 0" in str(exc_info.value).lower()
    assert "missing 'prediction' field" in str(exc_info.value)


@patch("requests.post")
def test_predict_proba_missing_probability_field_raises_rest_adapter_error(mock_post, rest_adapter, sample_df):
    mock_resp = MagicMock(status_code=200, json=lambda: {"prediction": 1})
    mock_post.return_value = mock_resp

    with pytest.raises(RESTAdapterError) as exc_info:
        rest_adapter.predict_proba(sample_df)

    assert "row 0" in str(exc_info.value).lower()
    assert "missing 'probability' field" in str(exc_info.value)


# =====================================================================
# 3. Health Check Tests
# =====================================================================


@patch("requests.get")
def test_health_success(mock_get, rest_adapter):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok", "service": "bank-scoring"})

    health_info = rest_adapter.health()

    assert health_info == {"status": "ok", "service": "bank-scoring"}
    mock_get.assert_called_once_with("http://mock-bank.local/api/v1/health", timeout=5.0)


@patch("requests.get")
def test_health_connection_error_returns_unreachable(mock_get, rest_adapter):
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection timed out")

    health_info = rest_adapter.health()

    assert isinstance(health_info, dict)
    assert health_info["status"] == "unreachable"
    assert "Connection timed out" in health_info["error"]


@patch("requests.get")
def test_health_non_2xx_returns_unreachable(mock_get, rest_adapter):
    mock_get.return_value = MagicMock(status_code=503, text="Service Unavailable")

    health_info = rest_adapter.health()

    assert isinstance(health_info, dict)
    assert health_info["status"] == "unreachable"
    assert "503" in health_info["error"]


# =====================================================================
# 4. Contract, Schema, and Metadata Tests
# =====================================================================


def test_rest_adapter_is_model_adapter(rest_adapter):
    assert isinstance(rest_adapter, ModelAdapter)
    assert rest_adapter.integration_type == "rest"


def test_load_fitted_model_raises_not_implemented(rest_adapter):
    with pytest.raises(NotImplementedError) as exc_info:
        rest_adapter.load_fitted_model()
    assert "RESTAdapter has no local fitted model artifact" in str(exc_info.value)


def test_background_data_behavior(sample_features, sample_schema, sample_capabilities, sample_df):
    adapter_no_bg = RESTAdapter(
        model_id="test",
        model_version="1.0.0",
        model_type="xgb",
        endpoint_url="http://test.local",
        feature_names=sample_features,
        input_schema=sample_schema,
        capabilities=sample_capabilities,
    )
    assert adapter_no_bg.background_data() is None

    adapter_with_bg = RESTAdapter(
        model_id="test",
        model_version="1.0.0",
        model_type="xgb",
        endpoint_url="http://test.local",
        feature_names=sample_features,
        input_schema=sample_schema,
        capabilities=sample_capabilities,
        background=sample_df,
    )
    assert adapter_with_bg.background_data() is sample_df


def test_input_schema_and_capabilities_not_derived_from_base_class(sample_schema, sample_capabilities):
    # Construct an adapter with arbitrary feature names not in German Credit
    custom_features = ["external_score", "custom_segment", "annual_turnover"]
    custom_schema = {
        "external_score": {"type": "numeric"},
        "custom_segment": {"type": "categorical"},
        "annual_turnover": {"type": "numeric"},
    }
    custom_caps = {
        "predict_proba": False,
        "batch": False,
        "explainability": False,
    }

    adapter = RESTAdapter(
        model_id="external-model",
        model_version="2.0.0",
        model_type="external_nn",
        endpoint_url="http://external.local",
        feature_names=custom_features,
        input_schema=custom_schema,
        capabilities=custom_caps,
    )

    # Prove it returns exact supplied schema (e.g. custom_segment is categorical, even though not in CATEGORICAL_FEATURES)
    assert adapter.input_schema == custom_schema
    assert adapter.input_schema["custom_segment"]["type"] == "categorical"
    assert adapter.capabilities == custom_caps


def test_metadata_returns_rest_metadata(rest_adapter, sample_capabilities):
    meta = rest_adapter.metadata()
    assert meta == {
        "model_id": "remote-credit-model",
        "model_version": "1.2.0",
        "model_type": "external_xgboost",
        "integration_type": "rest",
        "capabilities": sample_capabilities,
    }


# ===========================================================================
# Batch scoring (Phase 6)
#
# POST /score-batch exists so perturbation-based explainability is feasible
# against a remote model. The properties that matter are that the batch path
# returns EXACTLY what the per-row path would, in the SAME order, using far
# FEWER requests -- and that the capability is only advertised when real.
# ===========================================================================


@pytest.fixture
def batch_capabilities():
    return {
        "predict_proba": True,
        "batch": True,
        "batch_scoring": True,
        "explainability": False,
    }


@pytest.fixture
def batch_adapter(sample_features, sample_schema, batch_capabilities):
    return RESTAdapter(
        model_id="remote-batched-model",
        model_version="1.0.0",
        model_type="external_xgboost",
        endpoint_url="http://external.local",
        feature_names=sample_features,
        input_schema=sample_schema,
        capabilities=batch_capabilities,
    )


@pytest.fixture
def four_rows(sample_features):
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": ["p", "q", "p", "q"],
            "feature_c": [10.0, 20.0, 30.0, 40.0],
        }
    )[sample_features]


PREDICTIONS = [0, 1, 1, 0]
PROBABILITIES = [0.10, 0.80, 0.65, 0.20]


def _batch_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "predictions": list(PREDICTIONS),
        "probabilities": list(PROBABILITIES),
        "model_version": "1.0.0",
        "n_rows": 4,
    }
    return response


def _single_responses():
    responses = []
    for prediction, probability in zip(PREDICTIONS, PROBABILITIES):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "prediction": prediction,
            "probability": probability,
            "model_version": "1.0.0",
        }
        responses.append(response)
    return responses


def test_supports_batch_scoring_defaults_to_false(rest_adapter):
    """Absence of a declaration must never read as support.

    The fallback path issues one HTTP request PER ROW, so assuming batch
    support would let a perturbation explainer emit thousands of requests per
    explained row.
    """
    assert "batch_scoring" not in rest_adapter.capabilities
    assert rest_adapter.supports_batch_scoring is False


def test_batch_scoring_is_distinct_from_the_weaker_batch_flag(rest_adapter):
    """The weak flag means "accepts a multi-row frame" -- a per-row loop does.

    batch_scoring is the stronger claim: ONE call for the whole batch.
    Conflating them is what would license an unbounded request storm.
    """
    assert rest_adapter.capabilities["batch"] is True
    assert rest_adapter.supports_batch_scoring is False


def test_batch_predict_proba_uses_one_request(batch_adapter, four_rows):
    with patch(
        "app.models.rest_adapter.requests.post", return_value=_batch_response()
    ) as post:
        result = batch_adapter.predict_proba(four_rows)

    assert post.call_count == 1, "batch scoring must not fall back to per-row calls"
    assert post.call_args[0][0].endswith("/score-batch")
    np.testing.assert_allclose(result, PROBABILITIES)


def test_batch_output_equals_repeated_single_row_scoring(
    batch_adapter, rest_adapter, four_rows
):
    """Requirement 30: batching must be an optimisation, not a change.

    The same rows are scored both ways against the same stubbed backend, and
    the two probability vectors must be identical -- not merely close.
    """
    with patch("app.models.rest_adapter.requests.post", return_value=_batch_response()):
        batched = batch_adapter.predict_proba(four_rows)

    with patch("app.models.rest_adapter.requests.post", side_effect=_single_responses()):
        per_row = rest_adapter.predict_proba(four_rows)

    np.testing.assert_array_equal(batched, per_row)


def test_batch_reduces_the_request_count(batch_adapter, rest_adapter, four_rows):
    """Requirement 32: the whole reason the endpoint exists."""
    with patch(
        "app.models.rest_adapter.requests.post", return_value=_batch_response()
    ) as batched_post:
        batch_adapter.predict_proba(four_rows)

    with patch(
        "app.models.rest_adapter.requests.post", side_effect=_single_responses()
    ) as per_row_post:
        rest_adapter.predict_proba(four_rows)

    assert batched_post.call_count == 1
    assert per_row_post.call_count == len(four_rows) == 4
    assert batched_post.call_count < per_row_post.call_count


def test_batch_preserves_row_order(batch_adapter, four_rows):
    """Requirement 31: results are positional, so order IS the join key."""
    with patch(
        "app.models.rest_adapter.requests.post", return_value=_batch_response()
    ) as post:
        result = batch_adapter.predict_proba(four_rows)

    sent = post.call_args.kwargs["json"]["instances"]
    assert [row["feature_a"] for row in sent] == [1.0, 2.0, 3.0, 4.0]
    assert list(result) == PROBABILITIES
    # Not merely a permutation -- the exact sequence.
    assert list(result) != sorted(PROBABILITIES)


def test_batch_predict_returns_hard_labels_in_order(batch_adapter, four_rows):
    with patch(
        "app.models.rest_adapter.requests.post", return_value=_batch_response()
    ) as post:
        result = batch_adapter.predict(four_rows)

    assert post.call_count == 1
    np.testing.assert_array_equal(result, PREDICTIONS)


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"probabilities": PROBABILITIES}, "missing 'predictions'"),
        ({"predictions": PREDICTIONS}, "missing 'probabilities'"),
        (
            {"predictions": [0, 1], "probabilities": [0.1, 0.2]},
            "2 entries for 4 input row",
        ),
        (
            {"predictions": PREDICTIONS, "probabilities": "not-a-list"},
            "must be a list",
        ),
    ],
    ids=["no_predictions", "no_probabilities", "short_batch", "wrong_type"],
)
def test_malformed_batch_response_raises_rather_than_misaligning(
    batch_adapter, four_rows, payload, match
):
    """A short or malformed batch must never be padded or truncated.

    Silently accepting a 2-entry response for 4 rows would shift every
    subsequent applicant's score by one position -- a wrong answer that looks
    completely normal.
    """
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload

    with patch("app.models.rest_adapter.requests.post", return_value=response):
        with pytest.raises(RESTAdapterError, match=match):
            batch_adapter.predict(four_rows)


def test_batch_http_error_is_explicit(batch_adapter, four_rows):
    response = MagicMock()
    response.status_code = 503
    response.text = "service unavailable"

    with patch("app.models.rest_adapter.requests.post", return_value=response):
        with pytest.raises(RESTAdapterError, match="Batch scoring failed"):
            batch_adapter.predict_proba(four_rows)


def test_batch_request_error_is_explicit(batch_adapter, four_rows):
    with patch(
        "app.models.rest_adapter.requests.post",
        side_effect=requests.ConnectionError("refused"),
    ):
        with pytest.raises(RESTAdapterError, match="HTTP request error batch-scoring"):
            batch_adapter.predict_proba(four_rows)


def test_batch_path_still_respects_probability_capability(
    sample_features, sample_schema, four_rows
):
    """No probability capability means no request at all, batched or not."""
    adapter = RESTAdapter(
        model_id="labels-only",
        model_version="1.0.0",
        model_type="external",
        endpoint_url="http://external.local",
        feature_names=sample_features,
        input_schema=sample_schema,
        capabilities={"predict_proba": False, "batch": True, "batch_scoring": True},
    )

    with patch("app.models.rest_adapter.requests.post") as post:
        with pytest.raises(ProbabilityCapabilityUnavailable):
            adapter.predict_proba(four_rows)

    assert post.call_count == 0
