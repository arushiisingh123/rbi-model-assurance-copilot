"""Phase 2 integration tests: real German Credit data -> drift (owner: Arushi).

Proves the end-to-end path on real data:

    data/german_credit/german_credit.csv
        -> app.models.preprocessing (load + preprocess + split)
        -> reference = X_train, current = X_test   (pandas DataFrames)
        -> app.drift.drift_report()

WHAT THE TRAIN/TEST SETUP IS, AND IS NOT
    The train/test split is used here as a controlled reference/current pair to
    validate the integration path on real data. It measures distribution
    differences between the development training split and the held-out test
    split of one static dataset.

    It is **not** production monitoring data and must never be reported as
    evidence of production drift. Nothing in this repository observes a live
    lending population.

    ``build_drift_scenario()`` remains the tool for controlled, explicitly
    synthetic drift demonstrations; it is exercised separately below.

Contract references: ``docs/module-interfaces.md`` (Arushi's section) and
``docs/decisions.md``, "Phase 2 drift integration reference/current data".
"""
import math

import pandas as pd
import pytest

from app.config.thresholds import STATUS_PASS, VALID_STATUSES, classify_psi
from app.drift import drift_report
from app.drift.scenario import build_drift_scenario

DRIFT_KEYS = {"features_evaluated", "psi", "ks_statistic", "status", "is_mock"}

# The 7 continuous features of the German Credit schema. The other 13 columns
# are categorical and are deliberately excluded from PSI/KS.
EXPECTED_NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "present_residence",
    "age",
    "existing_credits",
    "num_dependents",
]


@pytest.fixture(scope="module")
def train_test_drift(german_credit_splits) -> dict:
    """Drift between the real train split (reference) and test split (current)."""
    X_train, X_test = german_credit_splits
    return drift_report(X_train, X_test)


# --------------------------------------------------------------------------
# The integration path itself
# --------------------------------------------------------------------------


def test_splits_are_dataframes_of_the_expected_shape(german_credit_splits):
    X_train, X_test = german_credit_splits

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert list(X_train.columns) == list(X_test.columns)
    assert X_train.shape[1] == 20


def test_train_test_drift_report_contract(train_test_drift: dict):
    assert set(train_test_drift.keys()) == DRIFT_KEYS
    assert train_test_drift["status"] in VALID_STATUSES
    assert train_test_drift["is_mock"] is False


def test_train_test_drift_metrics_are_valid_numbers(train_test_drift: dict):
    psi = train_test_drift["psi"]
    ks = train_test_drift["ks_statistic"]

    for value in (psi, ks):
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value >= 0.0

    # KS is a distance between two ECDFs, so it is bounded by 1.
    assert ks <= 1.0


def test_features_evaluated_are_the_numeric_german_credit_columns(
    train_test_drift: dict, german_credit_splits
):
    """``features_evaluated`` must describe exactly what was measured."""
    X_train, _ = german_credit_splits
    evaluated = train_test_drift["features_evaluated"]

    assert evaluated == EXPECTED_NUMERIC_FEATURES
    assert all(col in X_train.columns for col in evaluated)


def test_categorical_columns_including_attribute_9_are_not_covered(
    train_test_drift: dict,
):
    """Drift covers 7 of the 20 features; the other 13 are categorical.

    Attribute 9 is one of them, so this drift result carries no signal about
    the protected attribute. Pinned so the coverage limit stays visible.
    """
    evaluated = train_test_drift["features_evaluated"]

    assert "personal_status_and_sex" not in evaluated
    for categorical in ("status_checking_account", "purpose", "housing", "job"):
        assert categorical not in evaluated
    assert len(evaluated) == 7


def test_status_follows_the_central_psi_thresholds(train_test_drift: dict):
    """Status comes from app/config/thresholds.py, never from a local rule."""
    assert train_test_drift["status"] == classify_psi(train_test_drift["psi"])


def test_train_test_drift_is_deterministic(german_credit_splits, train_test_drift: dict):
    X_train, X_test = german_credit_splits
    assert drift_report(X_train, X_test) == train_test_drift


# --------------------------------------------------------------------------
# DataFrames are the internal representation (team contract #5)
# --------------------------------------------------------------------------


def test_drift_requires_dataframes_not_serialized_records(german_credit_splits):
    """Serialized records belong at the API boundary, not in the analytics.

    Orchestration must pass DataFrames in-process. Handing drift the API's
    ``list[dict]`` form fails loudly rather than silently computing nothing.
    """
    X_train, X_test = german_credit_splits

    with pytest.raises(ValueError, match="must be pandas DataFrames"):
        drift_report(X_train.to_dict("records"), X_test.to_dict("records"))


# --------------------------------------------------------------------------
# The model-output path agrees with the split path
# --------------------------------------------------------------------------


def test_model_feature_matrix_is_the_held_out_test_split(
    real_model_output: dict, german_credit_splits
):
    """``predict_batch()`` scores exactly the test split this suite uses.

    This is what lets the Phase 2 pipeline feed drift either from the model
    output or from the split directly without the two disagreeing.
    """
    _, X_test = german_credit_splits
    feature_matrix = real_model_output["feature_matrix"]

    assert isinstance(feature_matrix, pd.DataFrame)
    assert feature_matrix.equals(X_test.reset_index(drop=True))


def test_drift_from_model_output_matches_drift_from_split(
    real_model_output: dict, german_credit_splits, train_test_drift: dict
):
    X_train, _ = german_credit_splits
    from_model = drift_report(X_train, real_model_output["feature_matrix"])

    assert from_model == train_test_drift


# --------------------------------------------------------------------------
# Controlled SYNTHETIC drift demonstration
# --------------------------------------------------------------------------


def test_synthetic_shift_on_real_reference_is_detected(german_credit_splits):
    """SYNTHETIC test data -- not observed drift, not production evidence.

    A deliberately introduced shift on a real reference distribution must be
    detected. This demonstrates that the detector works; it says nothing about
    any real lending population.
    """
    X_train, _ = german_credit_splits

    reference, current = build_drift_scenario(
        X_train, shift_features="credit_amount", shift_amount=1.5
    )
    result = drift_report(reference, current)

    assert result["status"] != STATUS_PASS
    assert result["status"] == classify_psi(result["psi"])
    assert result["psi"] > 0.25
    assert result["is_mock"] is False
    assert "credit_amount" in result["features_evaluated"]


def test_synthetic_scenario_leaves_the_real_reference_untouched(german_credit_splits):
    """The scenario builder must not mutate the real split it is handed."""
    X_train, _ = german_credit_splits
    before = X_train.copy(deep=True)

    build_drift_scenario(X_train, shift_features="credit_amount", shift_amount=1.5)

    pd.testing.assert_frame_equal(X_train, before)


def test_synthetic_drift_is_reproducible(german_credit_splits):
    X_train, _ = german_credit_splits

    first = drift_report(
        *build_drift_scenario(X_train, shift_features="age", shift_amount=1.0)
    )
    second = drift_report(
        *build_drift_scenario(X_train, shift_features="age", shift_amount=1.0)
    )

    assert first == second


def test_synthetic_drift_is_larger_than_train_test_drift(
    german_credit_splits, train_test_drift: dict
):
    """The deliberate shift is far larger than the split's natural variation.

    Guards the interpretation: a PASS on train/test is not a claim that the
    detector is insensitive, and the synthetic FAIL is not a real-world finding.
    """
    X_train, _ = german_credit_splits
    synthetic = drift_report(
        *build_drift_scenario(X_train, shift_features="credit_amount", shift_amount=1.5)
    )

    assert synthetic["psi"] > train_test_drift["psi"]
