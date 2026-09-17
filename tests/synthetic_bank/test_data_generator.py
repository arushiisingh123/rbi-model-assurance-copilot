"""Unit tests for the synthetic bank customer generator (owner: Manas)."""
from app.synthetic_bank.data_generator import (
    CATEGORICAL_FEATURES,
    EMPLOYMENT_TYPES,
    FEATURE_COLUMNS,
    LOAN_PURPOSES,
    NUMERIC_FEATURES,
    REGIONS,
    TARGET_COLUMN,
    generate_customers,
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
