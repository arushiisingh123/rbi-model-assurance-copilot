"""Phase 2 integration tests: real credit model -> fairness (owner: Arushi).

Proves the end-to-end path on real data and the real trained model:

    data/german_credit/german_credit.csv
        -> app.models.preprocessing (load + preprocess + split)
        -> app.models.model.predict_batch()   (real sklearn Pipeline)
        -> feature_matrix["personal_status_and_sex"]   (real Attribute 9)
        -> app.fairness.fairness_report(..., favorable_label=0)

These tests call the real ``fairness_report()`` and never recompute a metric
themselves -- a test that reimplements the calculation would only prove the
test agrees with itself.

Contract references: ``docs/module-interfaces.md`` (Arushi's section) and
``docs/decisions.md``, "Phase 2 fairness and drift integration".
"""

import math

import pandas as pd
import pytest

from app.config.thresholds import VALID_STATUSES
from app.fairness import fairness_report

FAIRNESS_KEYS = {
    "protected_attribute",
    "demographic_parity_diff",
    "disparate_impact_ratio",
    "status",
    "is_mock",
    "groups",
}


@pytest.fixture(scope="module")
def sensitive_feature(real_model_output: dict) -> pd.Series:
    """Real Attribute 9 taken straight out of the model's feature matrix."""
    return real_model_output["feature_matrix"]["personal_status_and_sex"]


@pytest.fixture(scope="module")
def integration_report(
    real_model_output: dict, sensitive_feature: pd.Series
) -> dict:
    """The Phase 2 fairness result for the real model, favourable outcome = 0."""
    return fairness_report(
        real_model_output["predictions"],
        sensitive_feature,
        favorable_label=0,
    )


# --------------------------------------------------------------------------
# The integration path itself
# --------------------------------------------------------------------------


def test_model_output_supplies_everything_fairness_needs(real_model_output: dict):
    """Fairness needs no plumbing beyond the agreed model contract."""
    feature_matrix = real_model_output["feature_matrix"]

    assert isinstance(feature_matrix, pd.DataFrame)
    assert "personal_status_and_sex" in feature_matrix.columns
    assert (
        "personal_status_and_sex"
        in real_model_output["model_metadata"]["feature_names"]
    )
    assert real_model_output["is_mock"] is False
    # The deprecated spelling must not reappear anywhere downstream.
    assert "personal_status_sex" not in feature_matrix.columns


def test_predictions_and_sensitive_feature_are_aligned(
    real_model_output: dict, sensitive_feature: pd.Series
):
    predictions = real_model_output["predictions"]

    assert len(predictions) == len(sensitive_feature)
    assert len(predictions) > 0
    assert set(predictions).issubset({0, 1})


def test_model_output_satisfies_the_positional_pairing_contract(
    sensitive_feature: pd.Series,
):
    """The real integration path must not depend on luck.

    Fairness pairs predictions to groups by position, and rejects a pandas input
    whose index is not 0..n-1. predict_batch() resets its feature_matrix index,
    so the sensitive feature satisfies that contract -- this pins the dependency
    so a future change upstream fails here rather than silently downstream.
    """
    assert sensitive_feature.index.equals(pd.RangeIndex(len(sensitive_feature)))


def test_unreset_feature_matrix_index_is_rejected(
    real_model_output: dict, sensitive_feature: pd.Series
):
    """A pandas Series with a non-positional index must fail loudly.

    This simulates an upstream caller passing a feature Series whose original
    row labels were not reset before pairing it with the positional prediction
    list. The fairness boundary rejects it rather than allowing pandas to align
    by index.
    """
    unreset = sensitive_feature.copy()
    unreset.index = range(1000, 1000 + len(unreset))

    with pytest.raises(ValueError, match="non-positional index"):
        fairness_report(
            real_model_output["predictions"], unreset, favorable_label=0
        )


def test_real_model_fairness_report_contract(integration_report: dict):
    """The Phase 2 result honours the agreed output contract exactly."""
    assert set(integration_report.keys()) == FAIRNESS_KEYS
    assert integration_report["protected_attribute"] == "personal_status_and_sex"
    assert integration_report["status"] in VALID_STATUSES
    assert integration_report["is_mock"] is False


def test_real_model_fairness_metrics_are_valid_numbers(integration_report: dict):
    dp_diff = integration_report["demographic_parity_diff"]
    di_ratio = integration_report["disparate_impact_ratio"]

    for value in (dp_diff, di_ratio):
        assert isinstance(value, float)
        assert math.isfinite(value)

    # Both metrics are bounded by construction: they are built from selection
    # rates, which are proportions.
    assert 0.0 <= dp_diff <= 1.0
    assert 0.0 <= di_ratio <= 1.0


def test_protected_attribute_is_resolved_from_the_column_name(
    real_model_output: dict, sensitive_feature: pd.Series
):
    """The reported name comes from the real column, not a hardcoded default.

    Passing the same values without a Series name still yields the canonical
    name, so the contract holds either way -- but the named path is what the
    Phase 2 pipeline actually uses.
    """
    assert sensitive_feature.name == "personal_status_and_sex"

    unnamed = fairness_report(
        real_model_output["predictions"],
        list(sensitive_feature),
        favorable_label=0,
    )
    assert unnamed["protected_attribute"] == "personal_status_and_sex"


def test_attribute_9_is_used_as_raw_combined_categories(
    sensitive_feature: pd.Series,
):
    """No derived sex grouping is applied: the raw A9x categories are the groups.

    Attribute 9 combines marital/personal status with sex, so these labels must
    never be reduced to a standalone gender or sex field.
    """
    observed = set(sensitive_feature.unique())

    assert observed.issubset({"A91", "A92", "A93", "A94", "A95"})
    assert len(observed) >= 2
    # A95 ("female : single") has no instances in this dataset, which is why a
    # binary sex split would confound sex with marital status.
    assert "A95" not in observed


# --------------------------------------------------------------------------
# Favourable-label polarity on real model output
# --------------------------------------------------------------------------


def test_favorable_label_matches_model_declared_semantics(
    real_model_output: dict,
    sensitive_feature: pd.Series,
    integration_report: dict,
):
    """The model declares its favourable label; fairness must agree with it."""
    declared = real_model_output["model_metadata"]["label_semantics"][
        "favorable_outcome_label"
    ]
    assert declared == 0

    from_metadata = fairness_report(
        real_model_output["predictions"],
        sensitive_feature,
        favorable_label=declared,
    )
    assert from_metadata == integration_report


def test_polarity_regression_on_real_model_output(
    real_model_output: dict,
    sensitive_feature: pd.Series,
    integration_report: dict,
):
    """Reading the favourable label backwards changes the measured disparity.

    ``favorable_label=0`` measures the rate of receiving a GOOD credit decision.
    ``favorable_label=1`` measures the rate of receiving a BAD one, which is a
    different question and yields a different ratio on this data.

    Note what is deliberately NOT asserted: the two runs need not produce
    different *statuses*. On the current model both land in the FAIL band, so
    asserting a status difference would be asserting a coincidence.
    """
    inverted = fairness_report(
        real_model_output["predictions"],
        sensitive_feature,
        favorable_label=1,
    )

    assert (
        integration_report["disparate_impact_ratio"]
        != inverted["disparate_impact_ratio"]
    )
    assert integration_report["status"] in VALID_STATUSES
    assert inverted["status"] in VALID_STATUSES


def test_demographic_parity_difference_is_invariant_under_label_inversion(
    real_model_output: dict,
    sensitive_feature: pd.Series,
    integration_report: dict,
):
    """DPD cannot detect a flipped favourable label, but the ratio can.

    For binary predictions each group's selection rate under label 0 is
    ``1 - rate`` under label 1, so ``max - min`` is algebraically unchanged.
    Pinned here so nobody later mistakes the coincidence for a bug, and so it
    is on record that DPD alone is not a polarity check.
    """
    inverted = fairness_report(
        real_model_output["predictions"],
        sensitive_feature,
        favorable_label=1,
    )

    assert inverted["demographic_parity_diff"] == pytest.approx(
        integration_report["demographic_parity_diff"], abs=1e-9
    )


def test_integration_is_deterministic(
    real_model_output: dict,
    sensitive_feature: pd.Series,
    integration_report: dict,
):
    """Re-running the same evaluation yields an identical result."""
    again = fairness_report(
        real_model_output["predictions"],
        sensitive_feature,
        favorable_label=0,
    )
    assert again == integration_report

# --------------------------------------------------------------------------
# Per-group detail on real data (Phase 4, additive `groups` field)
# --------------------------------------------------------------------------

# Pinned from the real held-out test split scored by predict_batch().
REAL_GROUPS = [
    {"group": "A94", "count": 12, "favorable_count": 10, "selection_rate": 0.8333},
    {"group": "A93", "count": 118, "favorable_count": 98, "selection_rate": 0.8305},
    {"group": "A92", "count": 61, "favorable_count": 41, "selection_rate": 0.6721},
    {"group": "A91", "count": 9, "favorable_count": 3, "selection_rate": 0.3333},
]


def test_real_groups_exact_values(integration_report: dict):
    """The exact breakdown the dashboard will chart, pinned."""
    assert integration_report["groups"] == REAL_GROUPS


def test_real_group_counts_sum_to_the_scored_population(integration_report: dict):
    assert sum(entry["count"] for entry in integration_report["groups"]) == 200


def test_real_groups_reproduce_the_real_aggregates(integration_report: dict):
    """Recomputed from the exact counts, not the rounded rates.

    See ``test_groups_reproduce_the_reported_aggregates`` in
    ``tests/fairness/test_fairness.py``: dividing two rounded rates can differ
    from the reported ratio in the last digit, so the ratio is checked against
    full-precision rates.
    """
    rates = [
        entry["favorable_count"] / entry["count"]
        for entry in integration_report["groups"]
    ]

    assert round(max(rates) - min(rates), 4) == integration_report[
        "demographic_parity_diff"
    ]
    assert round(min(rates) / max(rates), 4) == integration_report[
        "disparate_impact_ratio"
    ]


def test_real_groups_are_raw_attribute_9_codes(integration_report: dict):
    """Attribute 9 combines marital status and sex; codes are never relabelled."""
    labels = {entry["group"] for entry in integration_report["groups"]}

    assert labels <= {"A91", "A92", "A93", "A94"}
    assert "A95" not in labels
    for forbidden in ("male", "female", "gender", "sex"):
        assert not any(forbidden in str(label).lower() for label in labels)
