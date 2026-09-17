"""Unit tests for the synthetic bank customer generator (owner: Namitha)."""
import pandas as pd
import pytest

from app.synthetic_bank.data_generator import (
    CATEGORICAL_FEATURES,
    CURRENT_SCENARIOS,
    EMPLOYMENT_TYPES,
    FEATURE_COLUMNS,
    LABEL_SEMANTICS,
    LOAN_PURPOSES,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    REGIONS,
    TARGET_COLUMN,
    generate_current,
    generate_customers,
    generate_drift_customers,
    generate_edge_case_customers,
    generate_missing_data_customers,
    generate_reference,
)


def test_generate_customers_is_deterministic():
    df1 = generate_customers(n=500, random_state=42)
    df2 = generate_customers(n=500, random_state=42)
    assert df1.equals(df2)


def test_generate_customers_different_seed_differs():
    df1 = generate_customers(n=500, random_state=42)
    df2 = generate_customers(n=500, random_state=7)
    assert not df1.equals(df2)


def test_generate_customers_shape_and_columns():
    df = generate_customers(n=200, random_state=42)
    assert len(df) == 200
    assert set(df.columns) == set(FEATURE_COLUMNS) | {TARGET_COLUMN}


def test_generate_customers_positive_rate_in_plausible_range():
    df = generate_customers(n=2000, random_state=42)
    positive_rate = df[TARGET_COLUMN].mean()
    assert 0.10 <= positive_rate <= 0.40


def test_generate_customers_categorical_values_within_documented_sets():
    df = generate_customers(n=500, random_state=42)
    assert set(df["employment_type"].unique()).issubset(set(EMPLOYMENT_TYPES))
    assert set(df["region"].unique()).issubset(set(REGIONS))
    assert set(df["loan_purpose"].unique()).issubset(set(LOAN_PURPOSES))


def test_generate_customers_numeric_ranges_are_plausible():
    df = generate_customers(n=1000, random_state=42)
    assert (df["age"] >= 18).all() and (df["age"] <= 75).all()
    assert (df["annual_income"] > 0).all()
    assert (df["employment_years"] >= 0).all()
    assert (df["existing_loans"] >= 0).all()
    assert (df["credit_utilization_ratio"] >= 0).all()
    assert (df["credit_utilization_ratio"] <= 1).all()
    assert (df["late_payments_12m"] >= 0).all()
    assert (df["loan_amount"] > 0).all()


def test_default_flag_has_real_signal_from_employment_type():
    df = generate_customers(n=3000, random_state=42)
    rates = df.groupby("employment_type")[TARGET_COLUMN].mean()
    # Unemployed applicants must default meaningfully more often than salaried --
    # if this ever failed it would mean default_flag degenerated into pure noise.
    assert rates["unemployed"] > rates["salaried"]


def test_default_flag_has_real_signal_from_credit_utilization():
    df = generate_customers(n=3000, random_state=42)
    high_util = df[df["credit_utilization_ratio"] >= df["credit_utilization_ratio"].median()]
    low_util = df[df["credit_utilization_ratio"] < df["credit_utilization_ratio"].median()]
    assert high_util[TARGET_COLUMN].mean() > low_util[TARGET_COLUMN].mean()


# =====================================================================
# LABEL_SEMANTICS (additive metadata)
# =====================================================================


def test_label_semantics_matches_existing_label_values():
    assert LABEL_SEMANTICS["positive_class"] == POSITIVE_CLASS == 1
    assert LABEL_SEMANTICS["favorable_outcome_label"] == 0
    assert LABEL_SEMANTICS["probabilities_represent"] == "P(class == 1) = P(BAD)"


# =====================================================================
# generate_drift_customers -- controlled INPUT-FEATURE drift (data only)
# =====================================================================


def test_generate_drift_customers_is_deterministic():
    df1 = generate_drift_customers(n=500, random_state=99)
    df2 = generate_drift_customers(n=500, random_state=99)
    assert df1.equals(df2)


def test_generate_drift_customers_schema_matches_baseline():
    baseline = generate_customers(n=200, random_state=42)
    drifted = generate_drift_customers(n=200, random_state=99)
    assert set(drifted.columns) == set(baseline.columns)


def test_generate_drift_customers_shifts_credit_utilization_and_late_payments():
    baseline = generate_customers(n=5000, random_state=42)
    drifted = generate_drift_customers(n=5000, random_state=99)

    assert drifted["credit_utilization_ratio"].mean() > baseline["credit_utilization_ratio"].mean()
    assert drifted["late_payments_12m"].mean() > baseline["late_payments_12m"].mean()


def test_generate_drift_customers_shifts_employment_type_mix():
    baseline = generate_customers(n=5000, random_state=42)
    drifted = generate_drift_customers(n=5000, random_state=99)

    baseline_unemployed_rate = (baseline["employment_type"] == "unemployed").mean()
    drifted_unemployed_rate = (drifted["employment_type"] == "unemployed").mean()
    assert drifted_unemployed_rate > baseline_unemployed_rate


def test_generate_drift_customers_values_stay_within_documented_ranges():
    drifted = generate_drift_customers(n=2000, random_state=99)

    assert (drifted["credit_utilization_ratio"] >= 0.0).all()
    assert (drifted["credit_utilization_ratio"] <= 1.0).all()
    assert (drifted["late_payments_12m"] >= 0).all()
    assert (drifted["late_payments_12m"] <= 12).all()
    assert set(drifted["employment_type"].unique()).issubset(set(EMPLOYMENT_TYPES))
    assert (drifted["age"] >= 18).all() and (drifted["age"] <= 75).all()


def test_generate_drift_customers_does_not_require_target_rate_to_shift():
    # Only the documented INPUT features are asserted above -- this test
    # exists to record that the target/default rate shifting is an
    # incidental side effect, never a requirement this function enforces.
    drifted = generate_drift_customers(n=2000, random_state=99)
    assert TARGET_COLUMN in drifted.columns


# =====================================================================
# generate_edge_case_customers -- deterministic boundary inputs
# =====================================================================


def test_generate_edge_case_customers_is_deterministic():
    df1 = generate_edge_case_customers()
    df2 = generate_edge_case_customers()
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_edge_case_customers_has_valid_schema_no_target():
    df = generate_edge_case_customers()
    assert list(df.columns) == FEATURE_COLUMNS
    assert TARGET_COLUMN not in df.columns


def test_generate_edge_case_customers_includes_documented_boundaries():
    df = generate_edge_case_customers()
    assert df["age"].min() == 18
    assert df["age"].max() == 75
    assert df["annual_income"].min() == 5_000.0
    assert df["credit_utilization_ratio"].min() == 0.0
    assert df["credit_utilization_ratio"].max() == 1.0
    assert df["existing_loans"].min() == 0
    assert df["existing_loans"].max() == 8
    assert df["late_payments_12m"].min() == 0
    assert df["late_payments_12m"].max() == 12
    assert df["employment_years"].min() == 0.0


def test_generate_edge_case_customers_categorical_values_are_valid():
    df = generate_edge_case_customers()
    assert set(df["employment_type"].unique()).issubset(set(EMPLOYMENT_TYPES))
    assert set(df["region"].unique()).issubset(set(REGIONS))
    assert set(df["loan_purpose"].unique()).issubset(set(LOAN_PURPOSES))


# =====================================================================
# generate_missing_data_customers -- deterministic NaN injection
# =====================================================================


def test_generate_missing_data_customers_is_deterministic():
    df1 = generate_missing_data_customers(n=500, random_state=11, missing_rate=0.2)
    df2 = generate_missing_data_customers(n=500, random_state=11, missing_rate=0.2)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_missing_data_customers_schema_no_target():
    df = generate_missing_data_customers(n=200, random_state=11, missing_rate=0.1)
    assert list(df.columns) == FEATURE_COLUMNS
    assert TARGET_COLUMN not in df.columns


def test_generate_missing_data_customers_requested_rate_is_reasonably_represented():
    n = 5000
    missing_rate = 0.2
    df = generate_missing_data_customers(n=n, random_state=11, missing_rate=missing_rate)

    for col in FEATURE_COLUMNS:
        observed_rate = df[col].isna().mean()
        assert abs(observed_rate - missing_rate) < 0.03, (
            f"column '{col}' missing rate {observed_rate} too far from requested {missing_rate}"
        )


def test_generate_missing_data_customers_zero_rate_injects_nothing():
    df = generate_missing_data_customers(n=300, random_state=11, missing_rate=0.0)
    assert df.isna().sum().sum() == 0


def test_generate_missing_data_customers_respects_columns_subset():
    df = generate_missing_data_customers(
        n=1000, random_state=11, missing_rate=0.3, columns=["age"]
    )
    assert df["age"].isna().sum() > 0
    other_columns = [c for c in FEATURE_COLUMNS if c != "age"]
    assert df[other_columns].isna().sum().sum() == 0


def test_generate_missing_data_customers_rejects_invalid_missing_rate():
    with pytest.raises(ValueError):
        generate_missing_data_customers(n=100, random_state=11, missing_rate=1.0)
    with pytest.raises(ValueError):
        generate_missing_data_customers(n=100, random_state=11, missing_rate=-0.1)


def test_generate_missing_data_customers_rejects_unknown_column():
    with pytest.raises(ValueError):
        generate_missing_data_customers(n=100, random_state=11, columns=["not_a_real_column"])


# =====================================================================
# generate_reference / generate_current -- named datasets (data only)
# =====================================================================


def test_generate_reference_is_deterministic_and_matches_generate_customers():
    ref1 = generate_reference(n=500, random_state=42)
    ref2 = generate_reference(n=500, random_state=42)
    assert ref1.equals(ref2)
    assert ref1.equals(generate_customers(n=500, random_state=42))


def test_generate_current_is_deterministic_per_scenario():
    for scenario in CURRENT_SCENARIOS:
        first = generate_current(scenario=scenario, n=300, random_state=43)
        second = generate_current(scenario=scenario, n=300, random_state=43)
        pd.testing.assert_frame_equal(first, second)


def test_generate_current_normal_and_drift_preserve_feature_and_target_schema():
    reference = generate_reference(n=300, random_state=42)
    for scenario in ("normal", "drift"):
        current = generate_current(scenario=scenario, n=300, random_state=43)
        assert set(current.columns) == set(reference.columns)


def test_generate_current_edge_and_missing_preserve_feature_schema_only():
    for scenario in ("edge", "missing"):
        current = generate_current(scenario=scenario, n=300, random_state=43)
        assert list(current.columns) == FEATURE_COLUMNS
        assert TARGET_COLUMN not in current.columns


def test_generate_current_dispatches_to_expected_generator():
    drift_via_current = generate_current(scenario="drift", n=300, random_state=99)
    drift_direct = generate_drift_customers(n=300, random_state=99)
    assert drift_via_current.equals(drift_direct)


def test_generate_current_rejects_invalid_scenario():
    with pytest.raises(ValueError):
        generate_current(scenario="not_a_real_scenario")
