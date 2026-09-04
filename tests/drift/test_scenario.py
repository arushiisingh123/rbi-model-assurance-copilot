"""Tests for app.drift.scenario.
Verifies deterministic behavior, reproducibility, non-mutation of input frames,
accurate standard-deviation shifting, and synthetic disclaimer in docstring.
"""
import pandas as pd
import pytest
import app.drift.scenario as scenario_module
from app.drift.scenario import build_drift_scenario
def test_build_drift_scenario_deterministic_identical_calls():
    df = pd.DataFrame({"income": [100.0, 200.0, 300.0, 400.0, 500.0], "age": [20, 30, 40, 50, 60]})
    ref1, cur1 = build_drift_scenario(df, shift_features="income", shift_amount=1.5)
    ref2, cur2 = build_drift_scenario(df, shift_features="income", shift_amount=1.5)
    pd.testing.assert_frame_equal(ref1, ref2)
    pd.testing.assert_frame_equal(cur1, cur2)
def test_build_drift_scenario_different_shifts_produce_different_outputs():
    df = pd.DataFrame({"income": [100.0, 200.0, 300.0, 400.0, 500.0]})
    _, cur_small = build_drift_scenario(df, shift_features="income", shift_amount=0.5)
    _, cur_large = build_drift_scenario(df, shift_features="income", shift_amount=2.0)
    assert not cur_small.equals(cur_large)
    assert cur_large["income"].iloc[0] > cur_small["income"].iloc[0]
def test_build_drift_scenario_does_not_mutate_input():
    df = pd.DataFrame({"income": [100.0, 200.0, 300.0, 400.0, 500.0]})
    df_orig = df.copy()
    ref, cur = build_drift_scenario(df, shift_features="income", shift_amount=2.0)
    pd.testing.assert_frame_equal(df, df_orig)
    pd.testing.assert_frame_equal(ref, df_orig)
    assert not cur.equals(df_orig)
def test_build_drift_scenario_exact_shift_calculation():
    df = pd.DataFrame({"income": [100.0, 200.0, 300.0, 400.0, 500.0]})
    std_val = df["income"].std()
    shift_amount = 1.5
    _, cur = build_drift_scenario(df, shift_features=["income"], shift_amount=shift_amount)
    expected_shifted = df["income"] + shift_amount * std_val
    pd.testing.assert_series_equal(cur["income"], expected_shifted, check_names=False)
def test_build_drift_scenario_errors():
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        build_drift_scenario({"a": [1, 2]}, shift_features="a", shift_amount=1.0)
    df = pd.DataFrame({"income": [100.0, 200.0]})
    with pytest.raises(ValueError, match="Features not found"):
        build_drift_scenario(df, shift_features="non_existent", shift_amount=1.0)
def test_build_drift_scenario_constant_feature_zero_shift():
    # When feature is constant, std() is 0.0, so offset is shift_amount * 0.0 == 0.0
    df = pd.DataFrame({"const_feature": [5.0, 5.0, 5.0, 5.0]})
    _, cur = build_drift_scenario(df, shift_features="const_feature", shift_amount=2.0)
    pd.testing.assert_series_equal(cur["const_feature"], df["const_feature"])
def test_build_drift_scenario_docstring_states_synthetic():
    doc = scenario_module.__doc__
    assert doc is not None
    assert "SYNTHETIC" in doc
    assert "real-world" in doc or "real population" in doc
