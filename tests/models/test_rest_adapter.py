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
