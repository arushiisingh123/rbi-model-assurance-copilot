"""Focused tests for evaluate_current_model() (owner: Namitha).

Phase 4, D1(a): expose the existing evaluate() held-out model performance
metrics as a property of the current trained model artifact. These tests
verify the new function delegates to evaluate() rather than recalculating
metrics, and that it introduces no regression to the existing Phase 3
predict_batch() contract.
"""

import pandas as pd
import pytest

from app.models import (
    evaluate,
    evaluate_current_model,
    load_dataset,
    predict_batch,
    preprocess,
    split_data,
)
from app.models.model import DEFAULT_DATASET_PATH, _get_or_train_default_model
from app.models.preprocessing import FEATURE_COLUMNS, INSTANCE_ID_COLUMN

REQUIRED_METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "n_test_samples",
    "is_mock",
}
NUMERIC_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


@pytest.fixture(scope="module")
def metrics():
    """evaluate_current_model() computed once for this test module."""
    return evaluate_current_model()


# =====================================================================
# Shape and value contract
# =====================================================================


def test_returns_exact_required_keys(metrics):
    """evaluate_current_model() returns exactly evaluate()'s key set."""
    assert set(metrics.keys()) == REQUIRED_METRIC_KEYS


def test_metric_values_are_numeric(metrics):
    """Each of the five metrics is a plain numeric (int/float) value."""
    for key in NUMERIC_METRIC_KEYS:
        assert isinstance(metrics[key], (int, float)), f"{key} is not numeric"


def test_metric_values_in_unit_interval(metrics):
    """accuracy/precision/recall/f1/roc_auc all fall within [0, 1]."""
    for key in NUMERIC_METRIC_KEYS:
        val = metrics[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} outside [0, 1]"


def test_n_test_samples_matches_canonical_held_out_split(metrics):
    """n_test_samples matches the canonical 20% held-out split size."""
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    assert metrics["n_test_samples"] == len(X_test)
    assert isinstance(metrics["n_test_samples"], int)


def test_is_mock_is_false(metrics):
    """Real held-out evaluation is never reported as mock data."""
    assert metrics["is_mock"] is False


# =====================================================================
# Delegation to evaluate() -- no reimplementation, no recalculation
# =====================================================================


def test_delegates_to_evaluate_on_canonical_split(metrics):
    """evaluate_current_model() must equal a direct evaluate() call on the
    same model and the same canonical held-out split -- proving it delegates
    rather than recalculating accuracy/precision/recall/f1/roc_auc itself.
    """
    model = _get_or_train_default_model()
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, y_test = split_data(X, y, test_size=0.2, random_state=42)
    expected = evaluate(model, X_test, y_test)

    assert metrics == expected


def test_repeated_calls_are_deterministic():
    """Calling evaluate_current_model() twice yields identical results
    (deterministic split + deterministic model artifact, no randomness).
    """
    first = evaluate_current_model()
    second = evaluate_current_model()
    assert first == second


# =====================================================================
# No regression to the existing Phase 3 predict_batch() contract
# =====================================================================


def test_predict_batch_contract_unchanged_by_this_addition():
    """predict_batch()'s output keys, instance_id exclusion, feature count,
    and probability semantics are exactly as before -- evaluate_current_model()
    is additive and does not touch predict_batch() at all.
    """
    result = predict_batch()

    assert set(result.keys()) == {
        "predictions",
        "probabilities",
        "instance_ids",
        "feature_matrix",
        "model_metadata",
        "is_mock",
    }
    assert "model_metrics" not in result  # not added to predict_batch() per decision

    # instance_id remains identity metadata, never a model feature
    assert len(FEATURE_COLUMNS) == 20
    assert INSTANCE_ID_COLUMN not in FEATURE_COLUMNS
    assert isinstance(result["feature_matrix"], pd.DataFrame)
    assert list(result["feature_matrix"].columns) == FEATURE_COLUMNS

    # instance_ids stay batch-aligned with predictions/probabilities
    n = len(result["predictions"])
    assert len(result["probabilities"]) == n
    assert len(result["instance_ids"]) == n
    assert len(result["feature_matrix"]) == n

    # label / probability semantics unchanged
    label_sem = result["model_metadata"]["label_semantics"]
    assert label_sem["favorable_outcome_label"] == 0
    assert label_sem["positive_class"] == 1
    assert label_sem["probabilities_represent"] == (
        "P(class == 1) = P(BAD / high credit risk)"
    )
    assert result["is_mock"] is False
