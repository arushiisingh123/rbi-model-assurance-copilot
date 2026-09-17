"""Unit tests for prediction/output drift (owner: Arushi).

Runs on inline fixtures only -- no real model, no dataset, no artifact. The
real-model behaviour lives in ``test_prediction_drift_integration.py``.

Deliberately NOT duplicated here: the quantile-PSI and KS arithmetic itself
(``test_drift.py`` owns it), the PSI threshold boundaries
(``tests/config/test_thresholds.py`` owns them), and ``worst_status``
(same file). This file covers only what is specific to the output channels:
which PSI formulation applies to which representation, and how an absent score
channel is reported.
"""
import numpy as np
import pandas as pd
import pytest

from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    classify_psi,
)
from app.drift.drift import _compute_feature_psi, drift_report
from app.drift.prediction_drift import (
    SCORE_AVAILABILITY_COMPUTED,
    SCORE_AVAILABILITY_UNAVAILABLE,
    prediction_drift_report,
)

RESULT_KEYS = {
    "label_psi",
    "label_status",
    "classes_evaluated",
    "per_class",
    "score_psi",
    "score_ks_statistic",
    "score_status",
    "score_availability",
    "status",
    "is_mock",
}


def labels(n_zero: int, n_one: int) -> list:
    return [0] * n_zero + [1] * n_one


# ---------------------------------------------------------------------------
# THE CENTRAL REGRESSION PIN -- why labels may not use the quantile PSI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref_rate, cur_rate",
    [
        (0.02, 0.37),
        (0.05, 0.40),
        (0.08, 0.43),
        (0.15, 0.50),
    ],
)
def test_quantile_psi_collapses_to_zero_on_binary_labels(ref_rate, cur_rate):
    """The defect this module exists to avoid, pinned on the real helper.

    ``_compute_feature_psi`` derives 10 quantile edges from the reference and
    collapses duplicates. On two-valued data those edges frequently reduce to
    ``[0, 1]``, which after -inf/+inf widening is a SINGLE bin holding all the
    mass -- so PSI is identically 0.0 however far the base rate moved.

    Each case below is a 35-percentage-point shift in the positive rate, which
    ``classify_psi`` would read as PASS. If a future change ever makes the
    quantile PSI safe on binary data this test fails, which is the signal to
    revisit ``_compute_label_psi`` -- not to weaken this test.
    """
    n = 400
    ref = np.array(labels(n - round(n * ref_rate), round(n * ref_rate)), dtype=float)
    cur = np.array(labels(n - round(n * cur_rate), round(n * cur_rate)), dtype=float)

    assert _compute_feature_psi(ref, cur) == 0.0
    assert classify_psi(_compute_feature_psi(ref, cur)) == STATUS_PASS

    # The categorical PSI this module uses sees the same shift as a real one.
    result = prediction_drift_report(ref, cur)
    assert result["label_psi"] > 0.25
    assert result["label_status"] == STATUS_FAIL


def test_feature_drift_on_a_label_column_would_report_zero_drift():
    """The same trap reached through the public feature-drift function.

    Routing predictions through ``drift_report()`` as if they were a feature
    column is the mistake this module prevents. Pinned so the reason prediction
    drift is a separate function is visible in a test, not only in a docstring.
    """
    ref = pd.DataFrame({"prediction": labels(380, 20)})
    cur = pd.DataFrame({"prediction": labels(260, 140)})

    feature_style = drift_report(ref, cur)
    assert feature_style["psi"] == 0.0
    assert feature_style["status"] == STATUS_PASS

    prediction_style = prediction_drift_report(
        ref["prediction"], cur["prediction"]
    )
    assert prediction_style["label_psi"] > 0.0
    assert prediction_style["label_status"] == STATUS_FAIL


def test_categorical_psi_matches_quantile_psi_where_binning_survives():
    """The categorical form is the same metric, not a competing definition.

    At a 0.30 reference rate the quantile grid does produce a usable interior
    edge, and there the two agree exactly. That is what makes the categorical
    form a fix for the degeneration rather than a different measurement.
    """
    ref = np.array(labels(140, 60), dtype=float)   # 0.30
    cur = np.array(labels(80, 120), dtype=float)   # 0.60

    quantile = round(_compute_feature_psi(ref, cur), 4)
    assert quantile > 0.0, "guard: this case must be one where binning survives"

    assert prediction_drift_report(ref, cur)["label_psi"] == quantile


# ---------------------------------------------------------------------------
# Label channel
# ---------------------------------------------------------------------------


def test_result_has_the_documented_key_set():
    result = prediction_drift_report(labels(50, 50), labels(50, 50))

    assert set(result.keys()) == RESULT_KEYS
    assert result["is_mock"] is False


def test_identical_windows_report_no_label_drift():
    result = prediction_drift_report(labels(70, 30), labels(70, 30))

    assert result["label_psi"] == 0.0
    assert result["label_status"] == STATUS_PASS


def test_a_single_class_in_both_windows_is_stable_not_pending():
    """A model that always predicts GOOD has a real, stable distribution.

    PENDING would claim the distribution could not be compared; it could, and
    it did not move.
    """
    result = prediction_drift_report([0] * 50, [0] * 50)

    assert result["label_status"] == STATUS_PASS
    assert result["label_psi"] == 0.0
    assert result["classes_evaluated"] == [0]


def test_a_class_appearing_only_in_one_window_is_still_evaluated():
    """A newly-emitted class is drift, not an error.

    The epsilon substitution handles the zero reference frequency instead of
    producing log(0).
    """
    result = prediction_drift_report([0] * 100, labels(50, 50))

    assert result["classes_evaluated"] == [0, 1]
    assert np.isfinite(result["label_psi"])
    assert result["label_psi"] > 0.0


def test_per_class_rates_align_with_classes_evaluated():
    result = prediction_drift_report(labels(70, 30), labels(40, 60))

    assert [e["class"] for e in result["per_class"]] == result["classes_evaluated"]
    assert result["per_class"] == [
        {"class": 0, "reference_rate": 0.7, "current_rate": 0.4},
        {"class": 1, "reference_rate": 0.3, "current_rate": 0.6},
    ]


def test_classes_are_reported_as_native_python_values():
    """Class labels travel into evidence records and JSON responses.

    A numpy scalar compares equal to its Python counterpart but is not
    JSON-serializable, so it must not leak out of the module.
    """
    result = prediction_drift_report(
        np.array(labels(60, 40)), np.array(labels(30, 70))
    )

    for cls in result["classes_evaluated"]:
        assert not isinstance(cls, np.generic)
        assert isinstance(cls, int)


def test_non_binary_labels_are_supported():
    """Nothing in the label channel assumes two classes.

    A future multi-class model must not need a second implementation.
    """
    ref = ["low"] * 60 + ["mid"] * 30 + ["high"] * 10
    cur = ["low"] * 20 + ["mid"] * 30 + ["high"] * 50

    result = prediction_drift_report(ref, cur)

    assert set(result["classes_evaluated"]) == {"low", "mid", "high"}
    assert len(result["per_class"]) == 3
    assert result["label_psi"] > 0.0


def test_windows_may_differ_in_length():
    """Reference and current are independent populations, not paired rows."""
    result = prediction_drift_report(labels(700, 300), labels(70, 30))

    assert result["label_psi"] == 0.0
    assert result["label_status"] == STATUS_PASS


# ---------------------------------------------------------------------------
# Score channel, and its absence
# ---------------------------------------------------------------------------


def test_score_channel_is_unavailable_when_no_scores_are_given():
    """A model without probabilities is unmeasured, not stable.

    ``None`` rather than 0.0: a zero would read as a PASS-worthy result for a
    measurement that never happened.
    """
    result = prediction_drift_report(labels(70, 30), labels(70, 30))

    assert result["score_availability"] == SCORE_AVAILABILITY_UNAVAILABLE
    assert result["score_psi"] is None
    assert result["score_ks_statistic"] is None
    assert result["score_status"] == STATUS_PENDING


def test_overall_status_ignores_an_unavailable_score_channel():
    """A model with no probability capability is judged on its labels alone.

    Its overall status must be the label status, not PENDING -- the label
    channel WAS measured.
    """
    result = prediction_drift_report(labels(380, 20), labels(260, 140))

    assert result["score_availability"] == SCORE_AVAILABILITY_UNAVAILABLE
    assert result["status"] == result["label_status"] == STATUS_FAIL


def test_score_channel_uses_the_continuous_metrics():
    """Scores are continuous, so the existing quantile PSI and KS apply."""
    rng = np.random.default_rng(0)
    ref_scores = rng.beta(2, 5, 400)
    cur_scores = rng.beta(5, 2, 400)

    result = prediction_drift_report(
        labels(300, 100),
        labels(100, 300),
        reference_scores=ref_scores,
        current_scores=cur_scores,
    )

    assert result["score_availability"] == SCORE_AVAILABILITY_COMPUTED
    assert result["score_psi"] > 0.25
    assert result["score_status"] == STATUS_FAIL
    assert 0.0 <= result["score_ks_statistic"] <= 1.0


def test_overall_status_is_the_worse_of_the_two_channels():
    """A stable label distribution must not mask a drifting score distribution.

    This is the case the label channel alone cannot see: the decision threshold
    keeps the same number of rows on each side while the underlying scores move
    substantially.
    """
    rng = np.random.default_rng(7)
    ref_scores = np.clip(rng.normal(0.30, 0.02, 400), 0, 1)
    cur_scores = np.clip(rng.normal(0.30, 0.20, 400), 0, 1)

    result = prediction_drift_report(
        labels(200, 200),
        labels(200, 200),
        reference_scores=ref_scores,
        current_scores=cur_scores,
    )

    assert result["label_status"] == STATUS_PASS
    assert result["score_status"] == STATUS_FAIL
    assert result["status"] == STATUS_FAIL


def test_non_finite_scores_are_excluded_rather_than_corrupting_psi():
    """Same cleaning rule feature drift applies, inherited not reimplemented."""
    ref_scores = [0.1, 0.2, np.nan, 0.4]
    cur_scores = [0.1, np.inf, 0.3, 0.4]

    result = prediction_drift_report(
        labels(2, 2),
        labels(2, 2),
        reference_scores=ref_scores,
        current_scores=cur_scores,
    )

    assert result["score_availability"] == SCORE_AVAILABILITY_COMPUTED
    assert np.isfinite(result["score_psi"])


def test_scores_with_no_finite_values_report_as_unavailable():
    result = prediction_drift_report(
        labels(2, 2),
        labels(2, 2),
        reference_scores=[np.nan, np.nan, np.nan, np.nan],
        current_scores=[0.1, 0.2, 0.3, 0.4],
    )

    assert result["score_availability"] == SCORE_AVAILABILITY_UNAVAILABLE
    assert result["score_psi"] is None


# ---------------------------------------------------------------------------
# PENDING and input validation
# ---------------------------------------------------------------------------


def test_empty_window_is_pending_not_fail():
    """Absent output is not evidence of drift -- the drift_report() rule."""
    result = prediction_drift_report([], labels(50, 50))

    assert result["status"] == STATUS_PENDING
    assert result["label_status"] == STATUS_PENDING
    assert result["label_psi"] == 0.0
    assert result["classes_evaluated"] == []
    assert result["per_class"] == []
    assert result["score_availability"] == SCORE_AVAILABILITY_UNAVAILABLE
    assert set(result.keys()) == RESULT_KEYS


def test_all_null_predictions_are_pending():
    result = prediction_drift_report([None, None], labels(20, 20))

    assert result["status"] == STATUS_PENDING


@pytest.mark.parametrize("bad", [None])
def test_none_predictions_are_refused(bad):
    with pytest.raises(ValueError, match="must not be None"):
        prediction_drift_report(bad, labels(10, 10))
    with pytest.raises(ValueError, match="must not be None"):
        prediction_drift_report(labels(10, 10), bad)


def test_a_dataframe_of_predictions_is_refused():
    """Prediction drift compares one output column, not a frame.

    Refusing names the mistake; silently taking the first column would measure
    something the caller did not ask for.
    """
    with pytest.raises(ValueError, match="not a\\s+DataFrame|1-D sequence"):
        prediction_drift_report(
            pd.DataFrame({"predictions": labels(10, 10)}), labels(10, 10)
        )


def test_scores_misaligned_with_their_own_predictions_are_refused():
    """Two channels measured over different record sets are not one window."""
    with pytest.raises(ValueError, match="describe the same records"):
        prediction_drift_report(
            labels(50, 50),
            labels(50, 50),
            reference_scores=[0.5] * 99,
            current_scores=[0.5] * 100,
        )


def test_a_pandas_index_does_not_change_the_result():
    """Reference and current are unpaired, so an index is safely discarded.

    Unlike fairness, where predictions and the sensitive feature are paired by
    position and a stray index must be refused, there is nothing here to
    mis-pair.
    """
    plain = prediction_drift_report(labels(70, 30), labels(40, 60))
    indexed = prediction_drift_report(
        pd.Series(labels(70, 30), index=range(500, 600)),
        pd.Series(labels(40, 60), index=range(900, 1000)),
    )

    assert plain == indexed


def test_result_is_deterministic():
    first = prediction_drift_report(
        labels(70, 30), labels(40, 60),
        reference_scores=[0.3] * 100, current_scores=[0.6] * 100,
    )
    second = prediction_drift_report(
        labels(70, 30), labels(40, 60),
        reference_scores=[0.3] * 100, current_scores=[0.6] * 100,
    )

    assert first == second


def test_module_defines_no_threshold_of_its_own():
    """Status must come from app.config.thresholds, the single source.

    Guards against a second set of drift bands appearing here later.
    """
    import app.drift.prediction_drift as module

    for name in dir(module):
        if name.startswith("__"):
            continue
        assert "THRESHOLD" not in name.upper(), (
            f"prediction_drift must not define its own threshold: {name}"
        )
