"""Comprehensive tests for app.drift.drift.

Verifies PSI and KS calculation, status classification against the authoritative
central thresholds, MAX aggregation across features, feature-eligibility rules
(numeric only, non-boolean, present in both frames, finite values only), edge
cases that must return PENDING rather than an invented FAIL, and schema
conformance.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    STATUS_WARNING,
    VALID_STATUSES,
    classify_psi,
)
from app.drift import drift_report
from app.drift.scenario import build_drift_scenario


# --------------------------------------------------------------------------
# Core calculation
# --------------------------------------------------------------------------


def test_identical_frames_report_no_drift():
    vals = np.linspace(10, 100, 100)
    ref = pd.DataFrame({"income": vals, "age": vals / 2})
    cur = pd.DataFrame({"income": vals, "age": vals / 2})

    res = drift_report(ref, cur)

    assert res["psi"] == 0.0
    assert res["ks_statistic"] == 0.0
    assert res["status"] == STATUS_PASS
    assert res["is_mock"] is False
    assert res["features_evaluated"] == ["income", "age"]


def test_known_fixture_hand_computed_ks():
    # ECDF_ref: 1->0.25, 2->0.5, 3->0.75, 4->1.0
    # ECDF_cur: 1->0.0,  2->0.0, 3->0.25, 4->0.5
    # max |diff| = 0.50 at x = 2
    ref = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"val": [3.0, 4.0, 5.0, 6.0]})

    assert drift_report(ref, cur)["ks_statistic"] == 0.50


def test_psi_status_boundaries_via_scenario():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"metric": rng.normal(100, 15, 1000)})

    _, cur_pass = build_drift_scenario(ref, shift_features="metric", shift_amount=0.0)
    res_pass = drift_report(ref, cur_pass)
    assert res_pass["psi"] < 0.10
    assert res_pass["status"] == STATUS_PASS

    _, cur_warn = build_drift_scenario(ref, shift_features="metric", shift_amount=0.35)
    res_warn = drift_report(ref, cur_warn)
    assert 0.10 <= res_warn["psi"] <= 0.25
    assert res_warn["status"] == STATUS_WARNING

    _, cur_fail = build_drift_scenario(ref, shift_features="metric", shift_amount=1.5)
    res_fail = drift_report(ref, cur_fail)
    assert res_fail["psi"] > 0.25
    assert res_fail["status"] == STATUS_FAIL


def test_max_aggregation_drives_status():
    vals = np.linspace(10, 100, 500)
    ref = pd.DataFrame({"stable_feature": vals, "drifted_feature": vals})
    cur = pd.DataFrame({"stable_feature": vals, "drifted_feature": vals + 200.0})

    res = drift_report(ref, cur)
    assert res["psi"] > 0.25
    assert res["status"] == STATUS_FAIL


def test_multiple_features_all_listed():
    rng = np.random.default_rng(123)
    ref = pd.DataFrame(
        {
            "a": rng.normal(50, 10, 300),
            "b": rng.normal(0, 1, 300),
            "c": rng.normal(-5, 2, 300),
        }
    )
    cur = pd.DataFrame(
        {
            "a": rng.normal(65, 10, 300),
            "b": rng.normal(0, 1, 300),
            "c": rng.normal(-5, 2, 300),
        }
    )

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["a", "b", "c"]
    assert res["psi"] > 0.0
    assert res["ks_statistic"] > 0.0


def test_small_dataset_does_not_crash():
    ref = pd.DataFrame({"v": [1.0, 2.0]})
    cur = pd.DataFrame({"v": [3.0, 4.0]})

    res = drift_report(ref, cur)
    assert res["status"] in VALID_STATUSES
    assert res["features_evaluated"] == ["v"]


# --------------------------------------------------------------------------
# Status is driven by PSI only, and always matches the reported PSI
# --------------------------------------------------------------------------


def test_ks_never_affects_status():
    # KS is 0.50 here, but the status must follow PSI alone.
    ref = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"val": [3.0, 4.0, 5.0, 6.0]})
    res = drift_report(ref, cur)

    assert res["ks_statistic"] == 0.50
    assert res["status"] == classify_psi(res["psi"])


def test_status_always_matches_reported_psi():
    """Regression: the reported PSI and the status must never disagree.

    The status is derived from the rounded value that is actually reported, so
    a PSI printed as 0.1 can never carry a PASS status (0.10 is WARNING).
    """
    rng = np.random.default_rng(11)
    ref = pd.DataFrame({"metric": rng.normal(100, 15, 800)})

    for shift in [0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.5, 0.8, 1.2, 2.0]:
        _, cur = build_drift_scenario(ref, shift_features="metric", shift_amount=shift)
        res = drift_report(ref, cur)
        assert res["status"] == classify_psi(res["psi"]), f"disagreed at shift={shift}"


# --------------------------------------------------------------------------
# Feature eligibility
# --------------------------------------------------------------------------


def test_categorical_columns_are_excluded():
    ref = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
    cur = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["num"]
    assert "cat" not in res["features_evaluated"]


def test_boolean_columns_are_excluded():
    """Booleans are numeric to pandas but semantically categorical."""
    ref = pd.DataFrame({"num": [1.0, 2.0, 3.0], "flag": [True, False, True]})
    cur = pd.DataFrame({"num": [1.0, 2.0, 3.0], "flag": [False, False, True]})

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["num"]


def test_features_present_in_only_one_frame_are_excluded():
    ref = pd.DataFrame({"shared": [1.0, 2.0, 3.0], "ref_only": [1.0, 2.0, 3.0]})
    cur = pd.DataFrame({"shared": [1.0, 2.0, 3.0], "cur_only": [1.0, 2.0, 3.0]})

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["shared"]


def test_feature_with_no_usable_values_is_not_listed_as_evaluated():
    """A column that is all-NaN cannot be evaluated, so it must not be claimed."""
    ref = pd.DataFrame({"good": [1.0, 2.0, 3.0], "empty": [np.nan, np.nan, np.nan]})
    cur = pd.DataFrame({"good": [1.0, 2.0, 3.0], "empty": [1.0, 2.0, 3.0]})

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["good"]


def test_all_features_unusable_is_pending():
    ref = pd.DataFrame({"v": [np.nan, np.nan, np.nan]})
    cur = pd.DataFrame({"v": [1.0, 2.0, 3.0]})

    res = drift_report(ref, cur)
    assert res["status"] == STATUS_PENDING
    assert res["features_evaluated"] == []


# --------------------------------------------------------------------------
# Non-finite handling
# --------------------------------------------------------------------------


def test_infinities_are_excluded_not_binned():
    """Regression: infinities used to corrupt the quantile edges.

    Before the fix, a single +inf produced a PSI of ~2.58 and a spurious FAIL.
    Dropping the non-finite value must give exactly the same answer as if that
    row had simply been absent.
    """
    ref = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    cur_with_inf = pd.DataFrame({"v": [1.0, 2.0, np.inf, 4.0, 5.0]})
    cur_without = pd.DataFrame({"v": [1.0, 2.0, 4.0, 5.0]})

    assert drift_report(ref, cur_with_inf) == drift_report(ref, cur_without)


def test_infinities_do_not_produce_spurious_fail_or_warnings():
    """A single infinity in an otherwise stable sample must not fabricate drift.

    Before the fix, np.quantile over the infinity emitted a RuntimeWarning and
    produced a PSI of ~2.58, reporting FAIL on data that had not drifted.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(100, 15, 500)
    ref = pd.DataFrame({"v": base})

    contaminated = base.copy()
    contaminated[0] = np.inf
    cur = pd.DataFrame({"v": contaminated})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = drift_report(ref, cur)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, f"unexpected RuntimeWarning: {runtime_warnings}"
    assert res["status"] == STATUS_PASS
    assert res["psi"] < 0.10


def test_negative_infinity_in_reference_is_excluded():
    ref_with_inf = pd.DataFrame({"v": [-np.inf, 1.0, 2.0, 3.0, 4.0]})
    ref_without = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})

    assert drift_report(ref_with_inf, cur) == drift_report(ref_without, cur)


def test_nans_dropped_per_feature():
    ref = pd.DataFrame({"val": [1.0, 2.0, None, 4.0, 5.0]})
    cur = pd.DataFrame({"val": [None, 2.0, 3.0, 4.0, 5.0]})

    res = drift_report(ref, cur)
    assert res["status"] in VALID_STATUSES
    assert res["features_evaluated"] == ["val"]


# --------------------------------------------------------------------------
# Degenerate distributions
# --------------------------------------------------------------------------


def test_constant_feature_reports_zero_psi():
    df = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0]})
    res = drift_report(df, df)

    assert res["psi"] == 0.0
    assert res["status"] == STATUS_PASS


def test_duplicate_quantile_edges_collapse_and_drift_is_still_detected():
    """Heavily tied data produces duplicate quantile edges.

    np.unique collapses them (here 11 raw quantiles reduce to 3 edges). The
    calculation must not error, and a genuine distribution change must still be
    detected rather than swallowed by the collapsed bins.
    """
    ref = pd.DataFrame({"tied": [1.0] * 90 + [2.0] * 10})
    cur = pd.DataFrame({"tied": [1.0] * 10 + [2.0] * 90})

    res = drift_report(ref, cur)

    assert res["status"] == STATUS_FAIL
    assert res["psi"] > 0.25
    assert res["ks_statistic"] == pytest.approx(0.8, abs=1e-4)
    assert res["features_evaluated"] == ["tied"]


def test_all_identical_values_produce_no_drift():
    """A reference with zero spread has nothing to bin against -> PSI 0.0."""
    ref = pd.DataFrame({"tied": [7.0] * 50})
    cur = pd.DataFrame({"tied": [7.0] * 50})

    res = drift_report(ref, cur)
    assert res["psi"] == 0.0
    assert res["status"] == STATUS_PASS


# --------------------------------------------------------------------------
# PENDING / validation
# --------------------------------------------------------------------------


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="must not be None"):
        drift_report(None, pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="must not be None"):
        drift_report(pd.DataFrame({"a": [1]}), None)
    with pytest.raises(ValueError, match="must be pandas DataFrames"):
        drift_report([1, 2], pd.DataFrame({"a": [1]}))


def test_empty_frames_are_pending():
    empty = pd.DataFrame()
    res = drift_report(empty, empty)

    assert res["status"] == STATUS_PENDING
    assert res["psi"] == 0.0
    assert res["ks_statistic"] == 0.0
    assert res["features_evaluated"] == []


def test_empty_current_frame_is_pending():
    ref = pd.DataFrame({"v": [1.0, 2.0]})
    res = drift_report(ref, pd.DataFrame())

    assert res["status"] == STATUS_PENDING


def test_no_common_numeric_columns_is_pending():
    ref = pd.DataFrame({"name": ["Alice", "Bob"]})
    cur = pd.DataFrame({"name": ["Charlie", "David"]})

    res = drift_report(ref, cur)
    assert res["status"] == STATUS_PENDING
    assert res["features_evaluated"] == []


def test_missing_data_is_never_reported_as_fail():
    """Absent data is not evidence of drift."""
    for ref, cur in [
        (pd.DataFrame(), pd.DataFrame()),
        (pd.DataFrame({"a": ["x"]}), pd.DataFrame({"a": ["y"]})),
        (pd.DataFrame({"v": [np.nan]}), pd.DataFrame({"v": [1.0]})),
    ]:
        assert drift_report(ref, cur)["status"] != STATUS_FAIL


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_schema_full_key_set():
    ref = pd.DataFrame({"income": [100.0, 200.0, 300.0]})
    cur = pd.DataFrame({"income": [110.0, 210.0, 310.0]})
    res = drift_report(ref, cur)

    assert set(res.keys()) == {
        "features_evaluated",
        "psi",
        "ks_statistic",
        "status",
        "is_mock",
    }
    assert res["is_mock"] is False
    assert res["status"] in VALID_STATUSES
    assert isinstance(res["features_evaluated"], list)


def test_synthetic_scenario_drift_is_detected_end_to_end():
    """The synthetic scenario exists to prove detection works.

    This is demonstrated capability, not observed production drift.
    """
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"credit_amount": rng.normal(3000, 800, 600)})
    _, cur = build_drift_scenario(ref, shift_features="credit_amount", shift_amount=1.5)

    res = drift_report(ref, cur)
    assert res["psi"] > 0.25
    assert res["status"] == STATUS_FAIL
    assert res["is_mock"] is False
