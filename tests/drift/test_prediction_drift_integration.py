"""Prediction drift on the REAL models, over real monitoring windows (owner: Arushi).

THE ARCHITECTURAL PROPERTY UNDER TEST
    Feature drift is model-independent; prediction drift is model-specific.
    Both statements are tested here against the same two windows, so the
    contrast is visible in one file:

        reference window = German Credit train split
        current window   = German Credit held-out test split

    Logistic Regression and Random Forest score those identical rows. The
    feature matrices are byte-identical -- so feature drift must be identical --
    while the models' own outputs differ, so prediction drift must differ.

    Nothing here is fabricated to manufacture that difference. Both adapters are
    the real registered ones loading real persisted artifacts, and the windows
    are the project's canonical deterministic split. The divergence below is the
    models genuinely disagreeing: on this data LR's BAD-flag rate RISES from
    0.2137 to 0.2400 while RF's FALLS from 0.2350 to 0.1450.

    These are development splits used as monitoring windows for demonstration.
    They are NOT observed production monitoring data
    (docs/decisions.md, "Phase 2 drift integration reference/current data").

Deliberately NOT duplicated here: the categorical-vs-quantile PSI argument and
every input-validation case (``test_prediction_drift.py``), the feature-drift
arithmetic (``test_drift.py``), the real train/test feature-drift pins
(``test_drift_integration.py``), and the adapter contract itself
(``tests/models/test_model_adapter.py``, ``test_random_forest_adapter.py``).
"""
import pytest

from app.config.thresholds import STATUS_PENDING
from app.drift import drift_report, prediction_drift_report
from app.drift.prediction_drift import (
    SCORE_AVAILABILITY_COMPUTED,
    SCORE_AVAILABILITY_UNAVAILABLE,
)
from app.models.model import (
    LogisticRegressionAdapter,
    RandomForestAdapter,
    predict_batch,
)


@pytest.fixture(scope="module")
def lr_adapter(real_model_artifact: str) -> LogisticRegressionAdapter:
    return LogisticRegressionAdapter.load_default()


@pytest.fixture(scope="module")
def rf_adapter(real_model_artifact: str) -> RandomForestAdapter:
    return RandomForestAdapter.load_default()


def _windows(adapter, german_credit_splits) -> tuple:
    """Score the canonical train/current split pair with one adapter."""
    X_train, X_test = german_credit_splits
    reference = predict_batch(
        feature_matrix=X_train.reset_index(drop=True), adapter=adapter
    )
    current = predict_batch(
        feature_matrix=X_test.reset_index(drop=True), adapter=adapter
    )
    return reference, current


@pytest.fixture(scope="module")
def lr_windows(lr_adapter, german_credit_splits) -> tuple:
    return _windows(lr_adapter, german_credit_splits)


@pytest.fixture(scope="module")
def rf_windows(rf_adapter, german_credit_splits) -> tuple:
    return _windows(rf_adapter, german_credit_splits)


def _prediction_drift(windows: tuple) -> dict:
    reference, current = windows
    return prediction_drift_report(
        reference["predictions"],
        current["predictions"],
        reference_scores=reference["probabilities"],
        current_scores=current["probabilities"],
    )


# ---------------------------------------------------------------------------
# Each model's prediction drift can actually be calculated
# ---------------------------------------------------------------------------


def test_lr_prediction_drift_is_calculated_on_real_output(lr_windows):
    """Logistic Regression: both channels measured, no PENDING."""
    result = _prediction_drift(lr_windows)

    assert result["is_mock"] is False
    # Set, not sequence: classes_evaluated is in first-observed order, which is
    # a property of the row order in the window rather than a contract.
    assert set(result["classes_evaluated"]) == {0, 1}
    assert result["label_status"] != STATUS_PENDING
    assert result["score_availability"] == SCORE_AVAILABILITY_COMPUTED
    assert result["score_psi"] is not None
    assert result["score_ks_statistic"] is not None


def test_rf_prediction_drift_is_calculated_on_real_output(rf_windows):
    """Random Forest: same contract, no per-model implementation."""
    result = _prediction_drift(rf_windows)

    assert result["is_mock"] is False
    assert set(result["classes_evaluated"]) == {0, 1}
    assert result["label_status"] != STATUS_PENDING
    assert result["score_availability"] == SCORE_AVAILABILITY_COMPUTED
    assert result["score_psi"] is not None


def test_both_models_return_the_same_result_contract(lr_windows, rf_windows):
    """The output shape is a property of the module, not of the model."""
    lr = _prediction_drift(lr_windows)
    rf = _prediction_drift(rf_windows)

    assert set(lr.keys()) == set(rf.keys())


# ---------------------------------------------------------------------------
# THE CONTRAST: same input rows, same feature drift, different output drift
# ---------------------------------------------------------------------------


def test_both_adapters_scored_the_identical_feature_matrix(lr_windows, rf_windows):
    """The premise for everything below.

    If the two models had been given different rows, a difference in prediction
    drift would prove nothing about the models.
    """
    lr_ref, lr_cur = lr_windows
    rf_ref, rf_cur = rf_windows

    assert lr_ref["feature_matrix"].equals(rf_ref["feature_matrix"])
    assert lr_cur["feature_matrix"].equals(rf_cur["feature_matrix"])


def test_feature_drift_is_identical_across_the_two_models(lr_windows, rf_windows):
    """Feature drift stays MODEL-INDEPENDENT -- unchanged by this work.

    Identical here is the correct result, not a missing model signal: feature
    drift measures the input population, which both models shared. Making it
    differ per model would mean it had stopped measuring the data.
    """
    lr_ref, lr_cur = lr_windows
    rf_ref, rf_cur = rf_windows

    lr_feature_drift = drift_report(lr_ref["feature_matrix"], lr_cur["feature_matrix"])
    rf_feature_drift = drift_report(rf_ref["feature_matrix"], rf_cur["feature_matrix"])

    assert lr_feature_drift == rf_feature_drift


def test_prediction_drift_differs_across_the_two_models(lr_windows, rf_windows):
    """Prediction drift IS model-specific -- the point of this module.

    Same reference rows, same current rows, same code path; different models,
    different answers. Both output channels differ, so the difference is not an
    artefact of one metric's binning.
    """
    lr = _prediction_drift(lr_windows)
    rf = _prediction_drift(rf_windows)

    assert lr["label_psi"] != rf["label_psi"]
    assert lr["score_psi"] != rf["score_psi"]


def test_the_two_models_move_their_prediction_rates_in_opposite_directions(
    lr_windows, rf_windows
):
    """Why misattributing a prediction-drift result is a wrong statement.

    On these windows LR flags MORE bad credit in the current period and RF flags
    LESS. A result labelled with the wrong model would report the opposite of
    what that model actually did -- not a cosmetic mislabel.
    """
    lr = _prediction_drift(lr_windows)
    rf = _prediction_drift(rf_windows)

    def bad_rate_delta(result: dict) -> float:
        bad = next(e for e in result["per_class"] if e["class"] == 1)
        return bad["current_rate"] - bad["reference_rate"]

    assert bad_rate_delta(lr) > 0
    assert bad_rate_delta(rf) < 0


def test_prediction_drift_can_exceed_feature_drift_on_the_same_windows(rf_windows):
    """The monitoring gap this module closes.

    For Random Forest on these windows the input population is stable while the
    model's own score distribution has moved far more. Feature drift alone
    cannot surface that, which is exactly why output drift is measured
    separately rather than folded into the feature-drift number.
    """
    rf_ref, rf_cur = rf_windows

    feature_psi = drift_report(rf_ref["feature_matrix"], rf_cur["feature_matrix"])["psi"]
    score_psi = _prediction_drift(rf_windows)["score_psi"]

    assert score_psi > feature_psi


# ---------------------------------------------------------------------------
# Windows are directional, and the score channel degrades safely
# ---------------------------------------------------------------------------


def test_swapping_reference_and_current_is_a_different_measurement(lr_windows):
    """PSI bins come from the REFERENCE, so the pair is ordered, not symmetric.

    Pinned so a caller cannot assume the argument order is interchangeable.
    """
    reference, current = lr_windows

    forward = prediction_drift_report(
        reference["predictions"], current["predictions"],
        reference_scores=reference["probabilities"],
        current_scores=current["probabilities"],
    )
    backward = prediction_drift_report(
        current["predictions"], reference["predictions"],
        reference_scores=current["probabilities"],
        current_scores=reference["probabilities"],
    )

    assert forward["score_psi"] != backward["score_psi"]
    # The label channel is a symmetric sum over class frequencies, so it is
    # unchanged -- stated explicitly so the asymmetry above is not read as a bug.
    assert forward["label_psi"] == backward["label_psi"]
    assert forward["per_class"] != backward["per_class"]


def test_a_model_without_probabilities_still_gets_label_drift(lr_windows):
    """Degrading safely: withholding scores entirely must not lose the run.

    Simulates the capability a ``RESTAdapter`` may lack. The label channel is
    still measured and the overall status still comes from it.
    """
    reference, current = lr_windows

    with_scores = _prediction_drift(lr_windows)
    labels_only = prediction_drift_report(
        reference["predictions"], current["predictions"]
    )

    assert labels_only["score_availability"] == SCORE_AVAILABILITY_UNAVAILABLE
    assert labels_only["score_psi"] is None
    # The label measurement is untouched by the score channel's absence.
    assert labels_only["label_psi"] == with_scores["label_psi"]
    assert labels_only["status"] == labels_only["label_status"]


def test_prediction_drift_is_deterministic(lr_windows):
    assert _prediction_drift(lr_windows) == _prediction_drift(lr_windows)
