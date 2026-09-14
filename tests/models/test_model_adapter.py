"""Tests for the Phase 5A/5B model adapter boundary (owner: Namitha).

Verifies:
- ``LogisticRegressionAdapter`` exposes ``model_id`` / ``model_version`` /
  ``model_type`` / ``feature_names`` / ``supports_probability`` as the
  ``ModelAdapter`` boundary requires.
- ``predict()`` / ``predict_proba()`` match the existing LR pipeline
  directly, with ``predict_proba()`` returning the 1-D
  ``P(class == 1) == P(BAD)`` vector, not sklearn's raw 2-column
  ``predict_proba`` matrix.
- ``background_data()`` is the deterministic canonical training split.
- ``load_fitted_model()`` returns the underlying ``Pipeline``.
- ``predict_batch(adapter=...)`` is fully equivalent to ``predict_batch()``
  (no adapter) for the same default model: same predictions, probabilities,
  instance_ids, feature_matrix, and ``model_metadata`` -- ``model_id`` never
  leaks into ``model_metadata`` via either path.
- ``ProbabilityCapabilityUnavailable`` propagates through ``predict_batch()``
  when the supplied adapter cannot provide probabilities.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.models import LogisticRegressionAdapter, ModelAdapter, predict_batch
from app.models.model import (
    MODEL_ID,
    MODEL_TYPE,
    MODEL_VERSION,
    ProbabilityCapabilityUnavailable,
    _get_or_train_default_model,
)
from app.models.preprocessing import FEATURE_COLUMNS


# =====================================================================
# 1. Adapter identity + capability
# =====================================================================


@pytest.fixture(scope="module")
def adapter():
    return LogisticRegressionAdapter.load_default()


@pytest.fixture(scope="module")
def sample_rows():
    return predict_batch()["feature_matrix"].head(15).copy()


def test_adapter_is_a_model_adapter(adapter):
    assert isinstance(adapter, ModelAdapter)


def test_adapter_identity_fields(adapter):
    assert adapter.model_id == MODEL_ID == "german-credit-logistic-regression"
    assert adapter.model_version == MODEL_VERSION == "0.1.0"
    assert adapter.model_type == MODEL_TYPE == "logistic_regression"


def test_adapter_feature_names_match_raw_schema(adapter):
    assert adapter.feature_names == FEATURE_COLUMNS
    assert len(adapter.feature_names) == 20


def test_adapter_supports_probability_is_true_for_lr(adapter):
    assert adapter.supports_probability is True


# =====================================================================
# 2. predict() / predict_proba() match the underlying pipeline directly
# =====================================================================


def test_predict_matches_direct_pipeline_call(adapter, sample_rows):
    direct_model = _get_or_train_default_model()
    expected = direct_model.predict(sample_rows)
    actual = adapter.predict(sample_rows)
    assert list(actual) == list(expected)


def test_predict_proba_is_1d_p_of_bad_vector(adapter, sample_rows):
    proba = adapter.predict_proba(sample_rows)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(sample_rows),)  # 1-D, not a 2-column matrix
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_predict_proba_matches_direct_positive_class_column(adapter, sample_rows):
    direct_model = _get_or_train_default_model()
    classes = list(direct_model.classes_)
    expected = direct_model.predict_proba(sample_rows)[:, classes.index(1)]
    actual = adapter.predict_proba(sample_rows)
    assert np.allclose(actual, expected)


def test_predict_and_predict_proba_agree_on_decision_boundary(adapter, sample_rows):
    preds = adapter.predict(sample_rows)
    probs = adapter.predict_proba(sample_rows)
    for pred, prob in zip(preds, probs):
        assert (int(pred) == 1) == (prob >= 0.5)


# =====================================================================
# 3. Internal (non-serialized) accessors
# =====================================================================


def test_load_fitted_model_returns_the_pipeline(adapter):
    fitted = adapter.load_fitted_model()
    assert isinstance(fitted, Pipeline)


def test_background_data_is_the_canonical_training_split(adapter):
    background = adapter.background_data()
    assert isinstance(background, pd.DataFrame)
    assert list(background.columns) == FEATURE_COLUMNS
    assert len(background) == 800  # canonical 80% train split


def test_background_data_is_deterministic():
    a = LogisticRegressionAdapter.load_default().background_data()
    b = LogisticRegressionAdapter.load_default().background_data()
    pd.testing.assert_frame_equal(a, b)


# =====================================================================
# 4. predict_batch(adapter=...) equivalence to the existing LR path
# =====================================================================


def test_predict_batch_adapter_path_matches_default_path_exactly(adapter):
    default_result = predict_batch()
    adapter_result = predict_batch(adapter=adapter)

    assert adapter_result["predictions"] == default_result["predictions"]
    assert adapter_result["probabilities"] == default_result["probabilities"]
    assert adapter_result["instance_ids"] == default_result["instance_ids"]
    pd.testing.assert_frame_equal(
        adapter_result["feature_matrix"], default_result["feature_matrix"]
    )
    assert adapter_result["model_metadata"] == default_result["model_metadata"]
    assert adapter_result["is_mock"] is False
    assert default_result["is_mock"] is False


def test_predict_batch_output_keys_unchanged_with_and_without_adapter(adapter):
    default_result = predict_batch()
    adapter_result = predict_batch(adapter=adapter)

    expected_keys = {
        "predictions",
        "probabilities",
        "instance_ids",
        "feature_matrix",
        "model_metadata",
        "is_mock",
    }
    assert set(default_result.keys()) == expected_keys
    assert set(adapter_result.keys()) == expected_keys


def test_model_metadata_key_shape_unchanged_with_and_without_adapter(adapter):
    default_result = predict_batch()
    adapter_result = predict_batch(adapter=adapter)

    expected_metadata_keys = {
        "model_type",
        "version",
        "trained_on",
        "feature_names",
        "label_semantics",
    }
    assert set(default_result["model_metadata"].keys()) == expected_metadata_keys
    assert set(adapter_result["model_metadata"].keys()) == expected_metadata_keys
    assert "model_id" not in default_result["model_metadata"]
    assert "model_id" not in adapter_result["model_metadata"]


def test_predict_batch_adapter_path_with_custom_feature_matrix(adapter, sample_rows):
    result = predict_batch(feature_matrix=sample_rows, adapter=adapter)
    assert len(result["predictions"]) == len(sample_rows)
    assert len(result["probabilities"]) == len(sample_rows)
    assert result["model_metadata"]["feature_names"] == FEATURE_COLUMNS
    assert "model_id" not in result["model_metadata"]


# =====================================================================
# 5. ProbabilityCapabilityUnavailable propagates through predict_batch()
# =====================================================================


class _NoProbabilityAdapter(ModelAdapter):
    """Stub adapter for testing the probability-capability guard only."""

    model_id = "stub-no-probability"
    model_version = "0.0.0"
    model_type = "stub"

    def __init__(self, fitted_model):
        self.feature_names = list(FEATURE_COLUMNS)
        self._fitted_model = fitted_model

    @property
    def supports_probability(self) -> bool:
        return False

    def predict(self, X):
        return self._fitted_model.predict(X)

    def predict_proba(self, X):
        raise ProbabilityCapabilityUnavailable(
            "This model does not support probability predictions."
        )

    def load_fitted_model(self):
        return self._fitted_model

    def background_data(self):
        return None


def test_predict_batch_raises_when_adapter_lacks_probability_capability(sample_rows):
    stub = _NoProbabilityAdapter(_get_or_train_default_model())
    with pytest.raises(ProbabilityCapabilityUnavailable):
        predict_batch(feature_matrix=sample_rows, adapter=stub)
