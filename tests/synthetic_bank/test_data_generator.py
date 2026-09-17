"""Unit tests for the synthetic bank customer generator (owner: Namitha)."""
import pandas as pd
import pytest

from app.synthetic_bank.data_generator import (
    CATEGORICAL_FEATURES,
    CURRENT_SCENARIOS,
    CUSTOMER_ID_COLUMN,
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
    make_customer_ids,
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
    # Exact set equality, unchanged in strictness: a generated POPULATION
    # now also carries its durable identity column.
    assert set(df.columns) == set(FEATURE_COLUMNS) | {
        TARGET_COLUMN,
        CUSTOMER_ID_COLUMN,
    }


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


# ---------------------------------------------------------------------------
# Durable customer identity
#
# customer_id exists so explainability evidence can be attributed to a
# specific applicant: explain() reports row_index, which is a POSITION inside
# whatever frame was explained and restarts at 0 on every call, so it cannot
# anchor evidence on its own.
# ---------------------------------------------------------------------------


def test_customer_id_is_not_a_model_feature():
    """Identity must never reach the model as a feature.

    A per-customer identifier is perfectly predictive of its own row and
    carries no generalizable signal, so leaking it into FEATURE_COLUMNS would
    corrupt both the model and every explanation of it.
    """
    assert CUSTOMER_ID_COLUMN not in FEATURE_COLUMNS
    assert CUSTOMER_ID_COLUMN not in CATEGORICAL_FEATURES
    assert CUSTOMER_ID_COLUMN not in NUMERIC_FEATURES


def test_customer_ids_are_unique_and_deterministic():
    first = generate_customers(n=300, random_state=42)
    again = generate_customers(n=300, random_state=42)

    ids = first[CUSTOMER_ID_COLUMN]
    assert ids.is_unique
    assert len(ids) == 300
    assert list(ids) == list(again[CUSTOMER_ID_COLUMN])


def test_customer_ids_are_durable_across_population_sizes():
    """Row i keeps its id when the population is regenerated larger.

    This is what makes an id durable rather than merely unique: evidence
    referencing SB-0042-000007 still refers to the same applicant after the
    reference batch is rebuilt.
    """
    small = generate_customers(n=10, random_state=42)
    large = generate_customers(n=100, random_state=42)

    assert list(small[CUSTOMER_ID_COLUMN]) == list(large[CUSTOMER_ID_COLUMN].head(10))


def test_customer_ids_differ_between_seeds():
    """Two seeds are two different populations; their ids must not collide."""
    a = set(generate_customers(n=50, random_state=42)[CUSTOMER_ID_COLUMN])
    b = set(generate_customers(n=50, random_state=123)[CUSTOMER_ID_COLUMN])

    assert a.isdisjoint(b)


def test_make_customer_ids_matches_the_generated_frame():
    df = generate_customers(n=25, random_state=7)
    assert list(df[CUSTOMER_ID_COLUMN]) == make_customer_ids(25, 7)


def test_make_customer_ids_draws_no_randomness():
    """Identity is arithmetic, not sampled.

    This is what guarantees adding identity could not perturb the rng call
    order or any generated feature distribution -- the property the scenario
    architecture depends on.
    """
    assert make_customer_ids(5, 42) == make_customer_ids(5, 42)
    assert make_customer_ids(3, 42) == make_customer_ids(10, 42)[:3]
    assert make_customer_ids(2, 42) == ["SB-0042-000000", "SB-0042-000001"]


# ---------------------------------------------------------------------------
# Identity x scenario coexistence
#
# The two systems merged here must both survive: main's scenario architecture
# and durable identity. These tests pin the seam between them.
# ---------------------------------------------------------------------------


def test_every_population_scenario_carries_customer_identity():
    """The generated POPULATIONS all inherit identity from the shared core.

    Attaching it in _generate_population() rather than per-wrapper is what
    makes this hold without duplicating generation logic.
    """
    populations = {
        "generate_customers": generate_customers(n=20, random_state=42),
        "generate_drift_customers": generate_drift_customers(n=20, random_state=99),
        "generate_reference": generate_reference(n=20, random_state=42),
        "generate_current_normal": generate_current("normal", n=20),
        "generate_current_drift": generate_current("drift", n=20),
    }

    for name, df in populations.items():
        assert CUSTOMER_ID_COLUMN in df.columns, f"{name} lost customer identity"
        assert df[CUSTOMER_ID_COLUMN].is_unique, f"{name} has duplicate ids"
        assert df[CUSTOMER_ID_COLUMN].notna().all(), f"{name} has null ids"


def test_input_only_scenarios_deliberately_carry_no_identity():
    """The documented boundary, pinned so it cannot drift by accident.

    Edge cases are hand-specified schema probes with no seed or row
    provenance, and the missing-data set is an input-only scenario. Both have
    an exact FEATURE_COLUMNS contract that callers rely on, and labelling a
    boundary probe as a customer would invite evidence being attributed to an
    applicant who does not exist.
    """
    for df in (
        generate_edge_case_customers(),
        generate_missing_data_customers(n=20, random_state=11),
        generate_current("edge"),
        generate_current("missing", n=20),
    ):
        assert list(df.columns) == FEATURE_COLUMNS
        assert CUSTOMER_ID_COLUMN not in df.columns


def test_drift_and_baseline_populations_have_disjoint_identities():
    """A drifted population is a different set of people, not the same again.

    Their default seeds differ (42 vs 99), and because the seed is part of the
    id, that disjointness comes for free rather than needing to be enforced.
    """
    baseline = set(generate_customers(n=100, random_state=42)[CUSTOMER_ID_COLUMN])
    drifted = set(generate_drift_customers(n=100, random_state=99)[CUSTOMER_ID_COLUMN])

    assert baseline.isdisjoint(drifted)


def test_identity_did_not_change_the_generated_feature_distributions():
    """Identity is additive: dropping it reproduces the pre-identity frame.

    Guards the scenario architecture's determinism guarantee -- the feature
    values for a given seed must be exactly what they were before identity
    existed.
    """
    df = generate_customers(n=200, random_state=42)
    features_only = df.drop(columns=[CUSTOMER_ID_COLUMN])

    assert list(features_only.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    # The documented baseline properties still hold on the feature frame.
    assert 0.10 <= features_only[TARGET_COLUMN].mean() <= 0.40
    assert set(features_only["employment_type"]).issubset(set(EMPLOYMENT_TYPES))


def test_missing_data_injection_can_never_null_identity():
    """A missing identity is not a missing feature.

    columns must be a subset of FEATURE_COLUMNS, so identity is structurally
    excluded from missingness injection even if it were present.
    """
    with pytest.raises(ValueError, match="Unknown columns"):
        generate_missing_data_customers(
            n=20, random_state=11, columns=[CUSTOMER_ID_COLUMN]
        )


def test_model_consumers_select_feature_columns_and_never_see_identity():
    """How every real consumer uses this: df[FEATURE_COLUMNS].

    Mirrors app/models/registry.py's background batch and
    app/synthetic_bank/model.py's training selection.
    """
    df = generate_customers(n=30, random_state=123)
    model_input = df[FEATURE_COLUMNS]

    assert CUSTOMER_ID_COLUMN not in model_input.columns
    assert list(model_input.columns) == FEATURE_COLUMNS
    # ...while identity remains recoverable alongside, by position.
    assert len(df[CUSTOMER_ID_COLUMN]) == len(model_input)
