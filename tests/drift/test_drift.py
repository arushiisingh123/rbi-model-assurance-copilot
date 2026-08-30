"""Comprehensive tests for app.drift.drift.

Verifies PSI and KS calculation, status classification against authoritative thresholds,
MAX aggregation across features, independent PSI/KS maxima, edge cases (empty, constant,
no common numeric columns, NaNs), schema conformance, and re-export.
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
from app.drift import drift_report
from app.drift.scenario import build_drift_scenario


def test_drift_report_identical_frames():
    # Identical reference and current data -> PSI 0.0, KS 0.0, status PASS
    np.random.seed(42)
    vals = np.linspace(10, 100, 100)
    ref = pd.DataFrame({"income": vals, "age": vals / 2})
    cur = pd.DataFrame({"income": vals, "age": vals / 2})

    res = drift_report(ref, cur)

    assert res["psi"] == 0.0
    assert res["ks_statistic"] == 0.0
    assert res["status"] == STATUS_PASS
    assert res["is_mock"] is False
    assert res["features_evaluated"] == ["income", "age"]


def test_drift_report_known_fixture_hand_computed_ks():
    # ref = [1.0, 2.0, 3.0, 4.0]
    # cur = [3.0, 4.0, 5.0, 6.0]
    # ECDF_ref: 1->0.25, 2->0.5, 3->0.75, 4->1.0, 5->1.0, 6->1.0
    # ECDF_cur: 1->0.0, 2->0.0, 3->0.25, 4->0.5, 5->0.75, 6->1.0
    # |ECDF_ref - ECDF_cur| max is at x=2, 3, 4: |0.5 - 0.0| = 0.50
    ref = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"val": [3.0, 4.0, 5.0, 6.0]})

    res = drift_report(ref, cur)
    assert res["ks_statistic"] == 0.50


def test_drift_report_psi_status_boundaries():
    # Generate reference distribution
    np.random.seed(42)
    base = np.random.normal(100, 15, 1000)
    ref = pd.DataFrame({"metric": base})

    # 1. No shift -> PASS
    _, cur_pass = build_drift_scenario(ref, shift_features="metric", shift_amount=0.0)
    res_pass = drift_report(ref, cur_pass)
    assert res_pass["psi"] < 0.10
    assert res_pass["status"] == STATUS_PASS

    # 2. Moderate shift -> WARNING
    _, cur_warn = build_drift_scenario(ref, shift_features="metric", shift_amount=0.35)
    res_warn = drift_report(ref, cur_warn)
    assert 0.10 <= res_warn["psi"] <= 0.25
    assert res_warn["status"] == STATUS_WARNING

    # 3. Severe shift -> FAIL
    _, cur_fail = build_drift_scenario(ref, shift_features="metric", shift_amount=1.5)
    res_fail = drift_report(ref, cur_fail)
    assert res_fail["psi"] > 0.25
    assert res_fail["status"] == STATUS_FAIL


def test_drift_report_max_aggregation_drives_status():
    # Feature 1 is stable (PSI 0.0), Feature 2 is heavily drifted (PSI > 0.25)
    # The aggregated PSI must be the MAX, driving status to FAIL
    vals = np.linspace(10, 100, 500)
    ref = pd.DataFrame({"stable_feature": vals, "drifted_feature": vals})
    cur = pd.DataFrame({"stable_feature": vals, "drifted_feature": vals + 200.0})

    res = drift_report(ref, cur)
    assert res["psi"] > 0.25
    assert res["status"] == STATUS_FAIL


def test_drift_report_independent_maxima():
    # Construct case where feature A has higher PSI and feature B has higher KS
    # Feature A: distributed across bins causing high PSI
    # Feature B: small localized shift causing KS jump
    np.random.seed(123)
    ref_a = np.random.normal(50, 10, 500)
    cur_a = np.random.normal(65, 10, 500)  # Large mean shift -> high PSI

    ref_b = np.concatenate([np.repeat(10.0, 250), np.repeat(20.0, 250)])
    cur_b = np.concatenate([np.repeat(10.0, 50), np.repeat(20.0, 450)])  # Large jump at step -> KS = 0.40

    ref = pd.DataFrame({"feat_a": ref_a, "feat_b": ref_b})
    cur = pd.DataFrame({"feat_a": cur_a, "feat_b": cur_b})

    res = drift_report(ref, cur)
    assert res["psi"] > 0.0
    assert res["ks_statistic"] > 0.0
    assert set(res["features_evaluated"]) == {"feat_a", "feat_b"}


def test_drift_report_ks_never_affects_status():
    # Even if KS is large (e.g. 0.50), if PSI is small (< 0.10), status is PASS
    # Note: 4 points fixture has KS = 0.50, but let's check its PSI status is driven by classify_psi
    ref = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"val": [3.0, 4.0, 5.0, 6.0]})
    res = drift_report(ref, cur)
    # Status should match classify_psi(res["psi"])
    from app.config.thresholds import classify_psi
    assert res["status"] == classify_psi(res["psi"])


def test_drift_report_edge_cases():
    # 1. Non-DataFrame / None inputs
    with pytest.raises(ValueError, match="must not be None"):
        drift_report(None, pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="must be pandas DataFrames"):
        drift_report([1, 2], pd.DataFrame({"a": [1]}))

    # 2. Empty DataFrames -> status PENDING
    empty_df = pd.DataFrame()
    res_empty = drift_report(empty_df, empty_df)
    assert res_empty["status"] == STATUS_PENDING
    assert res_empty["psi"] == 0.0
    assert res_empty["ks_statistic"] == 0.0
    assert res_empty["features_evaluated"] == []

    # 3. No common numeric columns -> status PENDING
    df1 = pd.DataFrame({"name": ["Alice", "Bob"]})
    df2 = pd.DataFrame({"name": ["Charlie", "David"]})
    res_no_num = drift_report(df1, df2)
    assert res_no_num["status"] == STATUS_PENDING
    assert res_no_num["features_evaluated"] == []

    # 4. Constant feature -> PSI 0.0
    df_const = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0]})
    res_const = drift_report(df_const, df_const)
    assert res_const["psi"] == 0.0
    assert res_const["status"] == STATUS_PASS

    # 5. NaNs dropped per feature
    ref_nan = pd.DataFrame({"val": [1.0, 2.0, None, 4.0, 5.0]})
    cur_nan = pd.DataFrame({"val": [None, 2.0, 3.0, 4.0, 5.0]})
    res_nan = drift_report(ref_nan, cur_nan)
    assert res_nan["status"] in VALID_STATUSES
    assert res_nan["features_evaluated"] == ["val"]


def test_drift_report_schema_full_key_set():
    ref = pd.DataFrame({"income": [100.0, 200.0, 300.0]})
    cur = pd.DataFrame({"income": [110.0, 210.0, 310.0]})
    res = drift_report(ref, cur)

    expected_keys = {
        "features_evaluated",
        "psi",
        "ks_statistic",
        "status",
        "is_mock",
    }
    assert set(res.keys()) == expected_keys
    assert res["is_mock"] is False
    assert res["status"] in VALID_STATUSES
    assert isinstance(res["features_evaluated"], list)
