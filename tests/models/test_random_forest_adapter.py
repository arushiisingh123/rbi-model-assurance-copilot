"""Tests for the Phase 5C Random Forest model/adapter (owner: Namitha).

Mirrors tests/models/test_model_adapter.py for the second supported model.
Verifies:
- ``RandomForestAdapter`` exposes ``model_id`` / ``model_version`` /
  ``model_type`` / ``feature_names`` / ``supports_probability`` as the
  ``ModelAdapter`` boundary requires, with the stable
  ``german-credit-random-forest`` identity.
- The RF feature schema matches the existing 20 raw features exactly.
- ``predict()`` works and ``predict_proba()`` returns a 1-D
  ``P(class == 1) == P(BAD)`` vector in ``[0, 1]``, consistent with the
  decision boundary.
- ``load_fitted_model()`` and ``background_data()`` work and
  ``background_data()`` is deterministic, using the same canonical
  training split source as ``LogisticRegressionAdapter``.
- ``predict_batch(adapter=rf_adapter)`` preserves the exact existing output
  key shape and never leaks ``model_id`` into ``model_metadata``, while
  ``model_metadata`` reports RF type/version.
- ``predict_batch()`` with no adapter is unaffected by RF's existence and
  still reports the Logistic Regression model.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.models import ModelAdapter, RandomForestAdapter, predict_batch
from app.models.model import (
    MODEL_TYPE,
    MODEL_VERSION,
    RF_MODEL_ID,
    RF_MODEL_TYPE,
    RF_MODEL_VERSION,
    _get_or_train_default_rf_model,
)
from app.models.preprocessing import FEATURE_COLUMNS


# =====================================================================
# 1. Adapter identity + capability
# =====================================================================


@pytest.fixture(scope="module")
def rf_adapter():
    return RandomForestAdapter.load_default()


@pytest.fixture(scope="module")
def sample_rows():
    return predict_batch()["feature_matrix"].head(15).copy()


def test_rf_adapter_is_a_model_adapter(rf_adapter):
    assert isinstance(rf_adapter, ModelAdapter)


def test_rf_adapter_identity_fields(rf_adapter):
    assert rf_adapter.model_id == RF_MODEL_ID == "german-credit-random-forest"
    assert rf_adapter.model_version == RF_MODEL_VERSION == "0.1.0"
    assert rf_adapter.model_type == RF_MODEL_TYPE == "random_forest"
    # Distinct identity from the Logistic Regression model.
    assert rf_adapter.model_id != "german-credit-logistic-regression"
    assert rf_adapter.model_type != MODEL_TYPE


def test_rf_adapter_feature_names_match_raw_schema(rf_adapter):
    assert rf_adapter.feature_names == FEATURE_COLUMNS
    assert len(rf_adapter.feature_names) == 20


def test_rf_adapter_supports_probability_is_true(rf_adapter):
    assert rf_adapter.supports_probability is True


# =====================================================================
# 2. predict() / predict_proba() match the underlying pipeline directly
# =====================================================================


def test_rf_predict_matches_direct_pipeline_call(rf_adapter, sample_rows):
    direct_model = _get_or_train_default_rf_model()
    expected = direct_model.predict(sample_rows)
    actual = rf_adapter.predict(sample_rows)
    assert list(actual) == list(expected)


def test_rf_predict_proba_is_1d_p_of_bad_vector(rf_adapter, sample_rows):
    proba = rf_adapter.predict_proba(sample_rows)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(sample_rows),)  # 1-D, not a 2-column matrix
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_rf_predict_proba_matches_direct_positive_class_column(rf_adapter, sample_rows):
    direct_model = _get_or_train_default_rf_model()
    classes = list(direct_model.classes_)
    expected = direct_model.predict_proba(sample_rows)[:, classes.index(1)]
    actual = rf_adapter.predict_proba(sample_rows)
    assert np.allclose(actual, expected)


def test_rf_predict_and_predict_proba_agree_on_decision_boundary(rf_adapter, sample_rows):
    preds = rf_adapter.predict(sample_rows)
    probs = rf_adapter.predict_proba(sample_rows)
    for pred, prob in zip(preds, probs):
        assert (int(pred) == 1) == (prob >= 0.5)


# =====================================================================
# 3. Internal (non-serialized) accessors
# =====================================================================


def test_rf_load_fitted_model_returns_a_pipeline_with_rf_classifier(rf_adapter):
    fitted = rf_adapter.load_fitted_model()
    assert isinstance(fitted, Pipeline)
    assert isinstance(fitted.named_steps["classifier"], RandomForestClassifier)


def test_rf_background_data_is_the_canonical_training_split(rf_adapter):
    background = rf_adapter.background_data()
    assert isinstance(background, pd.DataFrame)
    assert list(background.columns) == FEATURE_COLUMNS
    assert len(background) == 800  # canonical 80% train split


def test_rf_background_data_is_deterministic():
    a = RandomForestAdapter.load_default().background_data()
    b = RandomForestAdapter.load_default().background_data()
    pd.testing.assert_frame_equal(a, b)


def test_rf_background_data_matches_lr_background_source(rf_adapter):
    """RF and LR adapters must draw background data from the same
    deterministic split (same dataset, same random_state=42 split), not a
    separately derived sample.
    """
    from app.models import LogisticRegressionAdapter

    lr_background = LogisticRegressionAdapter.load_default().background_data()
    rf_background = rf_adapter.background_data()
    pd.testing.assert_frame_equal(lr_background, rf_background)


# =====================================================================
# 4. predict_batch(adapter=rf_adapter) shape + identity rules
# =====================================================================


def test_rf_predict_batch_output_keys_unchanged(rf_adapter):
    result = predict_batch(adapter=rf_adapter)
    expected_keys = {
        "predictions",
        "probabilities",
        "instance_ids",
        "feature_matrix",
        "model_metadata",
        "is_mock",
    }
    assert set(result.keys()) == expected_keys
    assert result["is_mock"] is False


def test_rf_model_metadata_reports_rf_type_and_version_without_model_id(rf_adapter):
    result = predict_batch(adapter=rf_adapter)
    metadata = result["model_metadata"]

    expected_metadata_keys = {
        "model_type",
        "version",
        "trained_on",
        "feature_names",
        "label_semantics",
    }
    assert set(metadata.keys()) == expected_metadata_keys
    assert metadata["model_type"] == RF_MODEL_TYPE == "random_forest"
    assert metadata["version"] == RF_MODEL_VERSION
    assert metadata["feature_names"] == FEATURE_COLUMNS
    assert "model_id" not in metadata


def test_rf_predict_batch_probability_is_p_of_bad(rf_adapter):
    result = predict_batch(adapter=rf_adapter)
    for pred, prob in zip(result["predictions"], result["probabilities"]):
        if pred == 1:
            assert prob >= 0.5
        else:
            assert prob <= 0.5


def test_rf_predict_batch_with_custom_feature_matrix(rf_adapter, sample_rows):
    result = predict_batch(feature_matrix=sample_rows, adapter=rf_adapter)
    assert len(result["predictions"]) == len(sample_rows)
    assert len(result["probabilities"]) == len(sample_rows)
    assert result["model_metadata"]["feature_names"] == FEATURE_COLUMNS
    assert "model_id" not in result["model_metadata"]


# =====================================================================
# 5. No-adapter predict_batch() is unaffected by RF's existence
# =====================================================================


def test_predict_batch_without_adapter_still_defaults_to_logistic_regression():
    result = predict_batch()
    assert result["model_metadata"]["model_type"] == MODEL_TYPE == "logistic_regression"
    assert result["model_metadata"]["version"] == MODEL_VERSION
    assert "model_id" not in result["model_metadata"]
