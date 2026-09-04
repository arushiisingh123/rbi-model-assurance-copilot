"""Comprehensive tests for app.fairness.fairness.

Verifies fairness metrics (demographic parity difference, disparate impact ratio),
status classification against authoritative thresholds, additive favorable_label keyword,
edge cases (single group PENDING, zero rates, nulls), schema conformance, and re-export.
"""
import numpy as np
import pandas as pd
import pytest
from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    STATUS_WARNING,
    VALID_STATUSES,
)
from app.fairness import fairness_report


def test_fairness_report_perfect_parity():
    # Group A: 50% favorable, Group B: 50% favorable -> diff 0.0, ratio 1.0, PASS
    preds = [1, 0, 1, 0]
    sens = ["A", "A", "B", "B"]
    res = fairness_report(preds, sens)

    assert res["demographic_parity_diff"] == 0.0
    assert res["disparate_impact_ratio"] == 1.0
    assert res["status"] == STATUS_PASS
    assert res["is_mock"] is False


def test_fairness_report_rates_08_vs_04_fail():
    # Group A: 80% favorable (4/5), Group B: 40% favorable (2/5)
    # diff = 0.8 - 0.4 = 0.4, ratio = 0.4 / 0.8 = 0.5 (< 0.70 -> FAIL)
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5
    res = fairness_report(preds, sens)

    assert pytest.approx(res["demographic_parity_diff"], abs=1e-4) == 0.4
    assert pytest.approx(res["disparate_impact_ratio"], abs=1e-4) == 0.5
    assert res["status"] == STATUS_FAIL
    assert res["is_mock"] is False


def test_fairness_report_warning_band():
    # Group A: 100% favorable (4/4 = 1.0), Group B: 75% favorable (3/4 = 0.75)
    # ratio = 0.75 / 1.0 = 0.75 (inside [0.70, 0.80) -> WARNING)
    preds = [1, 1, 1, 1] + [1, 1, 1, 0]
    sens = ["A"] * 4 + ["B"] * 4
    res = fairness_report(preds, sens)

    assert pytest.approx(res["demographic_parity_diff"], abs=1e-4) == 0.25
    assert pytest.approx(res["disparate_impact_ratio"], abs=1e-4) == 0.75
    assert res["status"] == STATUS_WARNING


def test_fairness_report_favorable_label_inversion():
    # With favorable_label=1:
    # Group A: 80% 1s, Group B: 40% 1s -> ratio = 0.5 (FAIL)
    # With favorable_label=0 (e.g. 0 is Good / non-default):
    # Group A: 20% 0s (1/5), Group B: 60% 0s (3/5) -> ratio = 0.2 / 0.6 = 0.3333 (FAIL), diff = 0.4
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5

    res_default = fairness_report(preds, sens, favorable_label=1)
    res_inverted = fairness_report(preds, sens, favorable_label=0)

    assert pytest.approx(res_default["disparate_impact_ratio"], abs=1e-4) == 0.5
    assert pytest.approx(res_inverted["disparate_impact_ratio"], abs=1e-4) == pytest.approx(1 / 3, abs=1e-4)
    assert pytest.approx(res_inverted["demographic_parity_diff"], abs=1e-4) == 0.4


def test_fairness_report_zero_selection_rate():
    # All predictions are 0 (unfavorable), so all groups have selection rate 0.0
    preds = [0, 0, 0, 0]
    sens = ["A", "A", "B", "B"]
    res = fairness_report(preds, sens, favorable_label=1)

    assert res["demographic_parity_diff"] == 0.0
    assert res["disparate_impact_ratio"] == 1.0
    assert res["status"] == STATUS_PASS


def test_fairness_report_single_group_pending():
    # Only 1 group -> status PENDING, neutral metrics
    preds = [1, 0, 1]
    sens = ["A", "A", "A"]
    res = fairness_report(preds, sens)

    assert res["status"] == STATUS_PENDING
    assert res["demographic_parity_diff"] == 0.0
    assert res["disparate_impact_ratio"] == 1.0
    assert res["is_mock"] is False


def test_fairness_report_missing_nan_rows_dropped():
    # NaNs in predictions or sensitive feature are dropped
    preds = [1, 0, None, 1, 0]
    sens = ["A", "A", "B", "B", None]
    res = fairness_report(preds, sens)

    # Remaining valid rows:
    # A: [1, 0] -> rate 0.5
    # B: [1] -> rate 1.0
    # diff = 0.5, ratio = 0.5 -> FAIL
    assert pytest.approx(res["demographic_parity_diff"], abs=1e-4) == 0.5
    assert pytest.approx(res["disparate_impact_ratio"], abs=1e-4) == 0.5
    assert res["status"] == STATUS_FAIL


def test_fairness_report_input_validation_errors():
    with pytest.raises(ValueError, match="must not be None"):
        fairness_report(predictions=None, sensitive_feature=["A", "B"])
    with pytest.raises(ValueError, match="must not be None"):
        fairness_report(predictions=[1, 0], sensitive_feature=None)
    with pytest.raises(ValueError, match="Length mismatch"):
        fairness_report(predictions=[1, 0, 1], sensitive_feature=["A", "B"])
    with pytest.raises(ValueError, match="must not be empty"):
        fairness_report(predictions=[], sensitive_feature=[])


def test_fairness_report_schema_and_protected_attribute_name():
    # Test Series with custom name
    sens_series = pd.Series(["A91", "A92", "A91", "A92"], name="personal_status_and_sex")
    res = fairness_report(predictions=[1, 1, 0, 0], sensitive_feature=sens_series)

    # Exact key set
    expected_keys = {
        "protected_attribute",
        "demographic_parity_diff",
        "disparate_impact_ratio",
        "status",
        "is_mock",
    }
    assert set(res.keys()) == expected_keys
    assert res["protected_attribute"] == "personal_status_and_sex"
    assert res["protected_attribute"].lower() != "gender"
    assert res["status"] in VALID_STATUSES
    assert res["is_mock"] is False


def test_fairness_report_never_returns_gender_string():
    # Even if someone names the series "gender", protected_attribute must not be "gender"
    sens_series = pd.Series(["A", "B", "A", "B"], name="gender")
    res = fairness_report(predictions=[1, 0, 1, 0], sensitive_feature=sens_series)
    assert res["protected_attribute"] != "gender"
    assert res["protected_attribute"] == "personal_status_and_sex"
