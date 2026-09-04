"""Comprehensive tests for app.fairness.fairness.

Verifies fairness metrics (demographic parity difference, disparate impact ratio),
status classification against the authoritative central thresholds, explicit
favorable_label semantics, edge cases that must return PENDING rather than an
invented PASS/FAIL, schema conformance, and Attribute 9 naming.

Every test states favorable_label explicitly where the value affects the result:
the favourable outcome is domain-specific and must never be assumed silently.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    STATUS_WARNING,
    VALID_STATUSES,
    classify_disparate_impact,
)
from app.fairness import fairness_report
from app.fairness.fairness import DEFAULT_FAVORABLE_LABEL

GERMAN_CREDIT_CSV = (
    Path(__file__).parent.parent.parent / "data" / "german_credit" / "german_credit.csv"
)


# --------------------------------------------------------------------------
# favorable_label semantics
# --------------------------------------------------------------------------


def test_default_favorable_label_is_zero_for_current_credit_model():
    """The current credit model encodes 0 = GOOD, 1 = BAD.

    The default must therefore be 0. Defaulting to 1 would silently measure the
    rate of receiving a BAD credit decision as if it were the favourable one.
    """
    assert DEFAULT_FAVORABLE_LABEL == 0


def test_favorable_label_changes_the_result():
    # Group A: 4/5 predicted 1, Group B: 2/5 predicted 1.
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5

    # favorable_label=1 -> rates 0.8 / 0.4 -> ratio 0.5
    res_one = fairness_report(preds, sens, favorable_label=1)
    # favorable_label=0 -> rates 0.2 / 0.6 -> ratio 0.3333
    res_zero = fairness_report(preds, sens, favorable_label=0)

    assert res_one["disparate_impact_ratio"] == pytest.approx(0.5, abs=1e-4)
    assert res_zero["disparate_impact_ratio"] == pytest.approx(1 / 3, abs=1e-4)
    assert res_one["disparate_impact_ratio"] != res_zero["disparate_impact_ratio"]


def test_omitting_favorable_label_uses_the_default_not_one():
    """Regression: omitting favorable_label must behave as 0, not as 1."""
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5

    res_default = fairness_report(preds, sens)
    res_zero = fairness_report(preds, sens, favorable_label=0)
    res_one = fairness_report(preds, sens, favorable_label=1)

    assert res_default == res_zero
    assert res_default != res_one


def test_favorable_label_none_raises():
    with pytest.raises(ValueError, match="favorable_label must not be None"):
        fairness_report([1, 0], ["A", "B"], favorable_label=None)


def test_favorable_label_absent_from_predictions_is_pending():
    """A label that never occurs gives every group a zero selection rate.

    That is an assessment which could not be performed, not a pass.
    """
    preds = [0, 0, 1, 1]
    sens = ["A", "A", "B", "B"]
    res = fairness_report(preds, sens, favorable_label=7)

    assert res["status"] == STATUS_PENDING


@pytest.mark.skipif(
    not GERMAN_CREDIT_CSV.exists(), reason="German Credit dataset not available"
)
def test_real_dataset_polarity_regression():
    """Semantic sanity check on the real dataset and real Attribute 9 column.

    Reading the favourable label backwards flips the verdict on identical data.
    The expected values are asserted loosely (band + status), not pinned to the
    implementation, so this documents the semantics without freezing arithmetic.
    """
    models = pytest.importorskip("app.models.preprocessing")
    df = models.load_dataset(str(GERMAN_CREDIT_CSV))
    X, y, _, _ = models.preprocess(df)
    sens = X["personal_status_sex"]

    good_favourable = fairness_report(y, sens, favorable_label=0)
    bad_favourable = fairness_report(y, sens, favorable_label=1)

    # 0 = GOOD is the correct reading for this model.
    assert good_favourable["disparate_impact_ratio"] == pytest.approx(0.8179, abs=5e-3)
    assert good_favourable["status"] == STATUS_PASS

    # 1 = BAD is the incorrect reading, and it produces a different verdict.
    assert bad_favourable["disparate_impact_ratio"] == pytest.approx(0.6661, abs=5e-3)
    assert bad_favourable["status"] == STATUS_FAIL

    assert good_favourable["status"] != bad_favourable["status"]


# --------------------------------------------------------------------------
# Metric calculation
# --------------------------------------------------------------------------


def test_perfect_parity():
    preds = [1, 0, 1, 0]
    sens = ["A", "A", "B", "B"]
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["demographic_parity_diff"] == 0.0
    assert res["disparate_impact_ratio"] == 1.0
    assert res["status"] == STATUS_PASS
    assert res["is_mock"] is False


def test_rates_08_vs_04_fail():
    # A: 4/5 = 0.8, B: 2/5 = 0.4 -> diff 0.4, ratio 0.5 -> FAIL
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["demographic_parity_diff"] == pytest.approx(0.4, abs=1e-4)
    assert res["disparate_impact_ratio"] == pytest.approx(0.5, abs=1e-4)
    assert res["status"] == STATUS_FAIL


def test_warning_band():
    # A: 4/4 = 1.0, B: 3/4 = 0.75 -> ratio 0.75 -> WARNING
    preds = [1, 1, 1, 1] + [1, 1, 1, 0]
    sens = ["A"] * 4 + ["B"] * 4
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["demographic_parity_diff"] == pytest.approx(0.25, abs=1e-4)
    assert res["disparate_impact_ratio"] == pytest.approx(0.75, abs=1e-4)
    assert res["status"] == STATUS_WARNING


def test_nan_rows_are_dropped():
    preds = [1, 0, None, 1, 0]
    sens = ["A", "A", "B", "B", None]
    # Remaining: A -> [1, 0] rate 0.5 ; B -> [1] rate 1.0
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["demographic_parity_diff"] == pytest.approx(0.5, abs=1e-4)
    assert res["disparate_impact_ratio"] == pytest.approx(0.5, abs=1e-4)
    assert res["status"] == STATUS_FAIL


def test_non_binary_prediction_values():
    """Predictions need not be 0/1; the favourable label just has to match."""
    preds = ["approve", "deny", "approve", "deny", "deny", "deny"]
    sens = ["A", "A", "A", "B", "B", "B"]
    # A: 2/3, B: 0/3 -> ratio 0.0 -> FAIL
    res = fairness_report(preds, sens, favorable_label="approve")

    assert res["disparate_impact_ratio"] == 0.0
    assert res["status"] == STATUS_FAIL


# --------------------------------------------------------------------------
# Threshold boundaries -- values come from app/config/thresholds.py
# --------------------------------------------------------------------------


def _two_group_frame(favourable_a: int, total_a: int, favourable_b: int, total_b: int):
    preds = [1] * favourable_a + [0] * (total_a - favourable_a)
    preds += [1] * favourable_b + [0] * (total_b - favourable_b)
    sens = ["A"] * total_a + ["B"] * total_b
    return preds, sens


def test_di_boundary_exactly_pass_threshold():
    # A: 4/5 = 0.8, B: 5/5 = 1.0 -> ratio exactly 0.80 -> PASS
    preds, sens = _two_group_frame(4, 5, 5, 5)
    res = fairness_report(preds, sens, favorable_label=1)
    assert res["disparate_impact_ratio"] == pytest.approx(0.80, abs=1e-9)
    assert res["status"] == STATUS_PASS


def test_di_boundary_just_below_pass_threshold():
    # A: 79/100 = 0.79, B: 100/100 = 1.0 -> ratio 0.79 -> WARNING
    preds, sens = _two_group_frame(79, 100, 100, 100)
    res = fairness_report(preds, sens, favorable_label=1)
    assert res["disparate_impact_ratio"] == pytest.approx(0.79, abs=1e-9)
    assert res["status"] == STATUS_WARNING


def test_di_boundary_exactly_warning_threshold():
    # A: 7/10 = 0.7, B: 10/10 = 1.0 -> ratio exactly 0.70 -> WARNING
    preds, sens = _two_group_frame(7, 10, 10, 10)
    res = fairness_report(preds, sens, favorable_label=1)
    assert res["disparate_impact_ratio"] == pytest.approx(0.70, abs=1e-9)
    assert res["status"] == STATUS_WARNING


def test_di_boundary_just_below_warning_threshold():
    # A: 69/100 = 0.69, B: 100/100 = 1.0 -> ratio 0.69 -> FAIL
    preds, sens = _two_group_frame(69, 100, 100, 100)
    res = fairness_report(preds, sens, favorable_label=1)
    assert res["disparate_impact_ratio"] == pytest.approx(0.69, abs=1e-9)
    assert res["status"] == STATUS_FAIL


def test_status_always_matches_reported_ratio():
    """Regression: the reported ratio and the status must never disagree.

    Previously the status was computed from the unrounded ratio while the
    rounded ratio was reported, so a ratio of 0.79996 was reported as "0.8"
    (a PASS-range number) alongside a WARNING status.
    """
    # The exact case that used to disagree: 19999/25000 = 0.79996 -> rounds to 0.8
    preds, sens = _two_group_frame(19999, 25000, 10, 10)
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["disparate_impact_ratio"] == 0.8
    assert res["status"] == STATUS_PASS
    assert res["status"] == classify_disparate_impact(res["disparate_impact_ratio"])


def test_status_invariant_across_many_ratios():
    """The invariant must hold for arbitrary group splits."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        total_a = int(rng.integers(5, 200))
        total_b = int(rng.integers(5, 200))
        fav_a = int(rng.integers(0, total_a + 1))
        fav_b = int(rng.integers(1, total_b + 1))
        preds, sens = _two_group_frame(fav_a, total_a, fav_b, total_b)
        res = fairness_report(preds, sens, favorable_label=1)
        if res["status"] == STATUS_PENDING:
            continue
        assert res["status"] == classify_disparate_impact(
            res["disparate_impact_ratio"]
        )


def test_demographic_parity_difference_is_not_classified():
    """DPD is reported as a metric only; no threshold exists for it.

    A large DPD with a passing ratio must still be PASS.
    """
    # A: 100/100 = 1.0, B: 80/100 = 0.8 -> DPD 0.2 (large), ratio 0.8 -> PASS
    preds, sens = _two_group_frame(100, 100, 80, 100)
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["demographic_parity_diff"] == pytest.approx(0.2, abs=1e-4)
    assert res["status"] == STATUS_PASS


# --------------------------------------------------------------------------
# PENDING edge cases
# --------------------------------------------------------------------------


def test_single_group_is_pending():
    res = fairness_report([1, 0, 1], ["A", "A", "A"], favorable_label=1)

    assert res["status"] == STATUS_PENDING
    assert res["demographic_parity_diff"] == 0.0
    assert res["disparate_impact_ratio"] == 1.0
    assert res["is_mock"] is False


def test_zero_maximum_selection_rate_is_pending_not_pass():
    """Regression: no group receives the favourable outcome.

    The ratio is undefined, so this previously returned ratio 1.0 -> PASS.
    An assessment that could not be performed must be PENDING.
    """
    preds = [0, 0, 0, 0]
    sens = ["A", "A", "B", "B"]
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["status"] == STATUS_PENDING
    assert res["demographic_parity_diff"] == 0.0
    assert res["disparate_impact_ratio"] == 1.0


def test_group_present_but_with_no_favourable_outcomes_is_not_pending():
    """One group at zero is a real (maximal) disparity, not an unassessable run."""
    preds = [1, 1, 0, 0]
    sens = ["A", "A", "B", "B"]
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["disparate_impact_ratio"] == 0.0
    assert res["status"] == STATUS_FAIL


def test_input_validation_errors():
    with pytest.raises(ValueError, match="must not be None"):
        fairness_report(predictions=None, sensitive_feature=["A", "B"])
    with pytest.raises(ValueError, match="must not be None"):
        fairness_report(predictions=[1, 0], sensitive_feature=None)
    with pytest.raises(ValueError, match="Length mismatch"):
        fairness_report(predictions=[1, 0, 1], sensitive_feature=["A", "B"])
    with pytest.raises(ValueError, match="must not be empty"):
        fairness_report(predictions=[], sensitive_feature=[])
    with pytest.raises(ValueError, match="No valid rows remaining"):
        fairness_report(predictions=[None, None], sensitive_feature=["A", "B"])


# --------------------------------------------------------------------------
# Output contract and Attribute 9 naming
# --------------------------------------------------------------------------


def test_schema_full_key_set_and_attribute_9_name():
    sens = pd.Series(["A91", "A92", "A91", "A92"], name="personal_status_and_sex")
    res = fairness_report([1, 1, 0, 0], sens, favorable_label=1)

    assert set(res.keys()) == {
        "protected_attribute",
        "demographic_parity_diff",
        "disparate_impact_ratio",
        "status",
        "is_mock",
    }
    assert res["protected_attribute"] == "personal_status_and_sex"
    assert res["status"] in VALID_STATUSES
    assert res["is_mock"] is False


def test_gender_and_sex_names_are_rejected():
    """Attribute 9 must never be reported as a standalone gender/sex field."""
    for bad_name in ("gender", "Gender", "sex", "SEX"):
        sens = pd.Series(["A", "B", "A", "B"], name=bad_name)
        res = fairness_report([1, 0, 1, 0], sens, favorable_label=1)
        assert res["protected_attribute"] == "personal_status_and_sex"
        assert res["protected_attribute"].lower() not in ("gender", "sex")


def test_raw_attribute_9_categories_are_used_as_is():
    """No codebook remapping: A91..A94 are four distinct groups."""
    sens = pd.Series(
        ["A91", "A92", "A93", "A94"] * 5, name="personal_status_sex"
    )
    preds = [1, 0, 1, 0] * 5
    res = fairness_report(preds, sens, favorable_label=1)

    # A91/A93 always favourable, A92/A94 never -> ratio 0.0
    assert res["disparate_impact_ratio"] == 0.0
    assert res["protected_attribute"] == "personal_status_sex"


def test_non_series_sensitive_feature_uses_default_name():
    res = fairness_report([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=1)
    assert res["protected_attribute"] == "personal_status_and_sex"
