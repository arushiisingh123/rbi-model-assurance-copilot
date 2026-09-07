"""Tests for the Phase 3 fairness evidence layer (owner: Arushi).

Verifies that ``fairness_evidence()`` reports exact group counts, favourable
counts and selection rates; that its summary agrees with the Phase 2
``fairness_report()`` value-for-value; that category values and first-observed
ordering survive; that PENDING cases still yield group evidence without
inventing disparity; that validation is identical to ``fairness_report()``; and
that evidence stays population-level with no per-record identifier.
"""

import pandas as pd
import pytest

from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    VALID_STATUSES,
)
from app.fairness import fairness_evidence, fairness_report

GROUP_KEYS = {
    "evidence_type",
    "protected_attribute",
    "group",
    "group_count",
    "favorable_count",
    "selection_rate",
    "favorable_label",
    "is_mock",
}

SUMMARY_KEYS = {
    "evidence_type",
    "protected_attribute",
    "demographic_parity_diff",
    "disparate_impact_ratio",
    "status",
    "favorable_label",
    "is_mock",
}


def _groups(records):
    return [r for r in records if r["evidence_type"] == "fairness_group"]


def _summary(records):
    summaries = [r for r in records if r["evidence_type"] == "fairness_summary"]
    assert len(summaries) == 1, "exactly one summary record expected"
    return summaries[0]


# --------------------------------------------------------------------------
# Normal multi-group evidence
# --------------------------------------------------------------------------


def test_multi_group_evidence_shape():
    # A: 4/5 favourable, B: 2/5 favourable
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5

    records = fairness_evidence(preds, sens, favorable_label=1)

    assert len(records) == 3  # two groups + one summary
    assert [r["evidence_type"] for r in records] == [
        "fairness_group",
        "fairness_group",
        "fairness_summary",
    ]
    assert records[-1]["evidence_type"] == "fairness_summary", "summary comes last"


def test_exact_group_counts_favorable_counts_and_rates():
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5

    groups = _groups(fairness_evidence(preds, sens, favorable_label=1))
    by_group = {g["group"]: g for g in groups}

    assert by_group["A"]["group_count"] == 5
    assert by_group["A"]["favorable_count"] == 4
    assert by_group["A"]["selection_rate"] == 0.8

    assert by_group["B"]["group_count"] == 5
    assert by_group["B"]["favorable_count"] == 2
    assert by_group["B"]["selection_rate"] == 0.4


def test_group_counts_sum_to_the_evaluated_population():
    preds = [1, 0, 1, 0, 1, 1]
    sens = ["A", "A", "B", "B", "C", "C"]

    groups = _groups(fairness_evidence(preds, sens, favorable_label=1))
    assert sum(g["group_count"] for g in groups) == len(preds)
    assert sum(g["favorable_count"] for g in groups) == sum(
        1 for p in preds if p == 1
    )


def test_selection_rate_is_rounded_to_four_places():
    # 1/3 = 0.333333... -> 0.3333
    preds = [1, 0, 0] + [1, 1, 1]
    sens = ["A"] * 3 + ["B"] * 3

    by_group = {g["group"]: g for g in _groups(fairness_evidence(preds, sens, favorable_label=1))}
    assert by_group["A"]["selection_rate"] == 0.3333
    assert by_group["B"]["selection_rate"] == 1.0


def test_selection_rate_equals_favorable_count_over_group_count():
    preds = [1, 0, 1, 1, 0, 1, 0]
    sens = ["A", "A", "A", "B", "B", "C", "C"]

    for group in _groups(fairness_evidence(preds, sens, favorable_label=1)):
        expected = round(group["favorable_count"] / group["group_count"], 4)
        assert group["selection_rate"] == expected


# --------------------------------------------------------------------------
# Agreement with the Phase 2 contract
# --------------------------------------------------------------------------


def test_summary_matches_fairness_report_exactly():
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5

    report = fairness_report(preds, sens, favorable_label=1)
    summary = _summary(fairness_evidence(preds, sens, favorable_label=1))

    assert summary["protected_attribute"] == report["protected_attribute"]
    assert summary["demographic_parity_diff"] == report["demographic_parity_diff"]
    assert summary["disparate_impact_ratio"] == report["disparate_impact_ratio"]
    assert summary["status"] == report["status"]
    assert summary["is_mock"] == report["is_mock"]


@pytest.mark.parametrize(
    "preds, sens, favorable_label",
    [
        ([1, 1, 1, 1, 0] + [1, 1, 0, 0, 0], ["A"] * 5 + ["B"] * 5, 1),
        ([1, 0, 1, 0], ["A", "A", "B", "B"], 0),
        ([1] * 4 + [1, 1, 1, 0], ["A"] * 4 + ["B"] * 4, 1),
        ([0, 0, 0, 0], ["A", "A", "B", "B"], 1),
        ([1, 0, 1], ["A", "A", "A"], 1),
    ],
)
def test_summary_agrees_with_report_across_cases(preds, sens, favorable_label):
    """The two entry points must never disagree, including PENDING cases."""
    report = fairness_report(preds, sens, favorable_label=favorable_label)
    summary = _summary(fairness_evidence(preds, sens, favorable_label=favorable_label))

    for key in (
        "protected_attribute",
        "demographic_parity_diff",
        "disparate_impact_ratio",
        "status",
        "is_mock",
    ):
        assert summary[key] == report[key], f"{key} disagreed"


def test_fairness_report_contract_is_unchanged_by_the_evidence_layer():
    """The Phase 2 five-key contract must not have grown evidence fields."""
    report = fairness_report([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=1)

    assert set(report.keys()) == {
        "protected_attribute",
        "demographic_parity_diff",
        "disparate_impact_ratio",
        "status",
        "is_mock",
    }
    assert "evidence_type" not in report
    assert "groups" not in report
    assert "group_count" not in report


# --------------------------------------------------------------------------
# Ordering and category preservation
# --------------------------------------------------------------------------


def test_first_observed_group_ordering_is_preserved():
    """Groups appear in the order first seen, not sorted."""
    preds = [1, 0, 1, 0, 1, 0]
    sens = ["zebra", "alpha", "alpha", "monkey", "zebra", "monkey"]

    order = [g["group"] for g in _groups(fairness_evidence(preds, sens, favorable_label=1))]
    assert order == ["zebra", "alpha", "monkey"]
    assert order != sorted(order), "sorting would have hidden the ordering bug"


def test_original_category_values_are_preserved():
    """Raw Attribute 9 codes are reported verbatim, never remapped."""
    preds = [1, 0, 1, 0]
    sens = pd.Series(["A91", "A92", "A93", "A94"], name="personal_status_and_sex")

    groups = _groups(fairness_evidence(preds, sens, favorable_label=1))
    assert [g["group"] for g in groups] == ["A91", "A92", "A93", "A94"]


def test_non_string_category_values_are_preserved():
    preds = [1, 0, 1, 0]
    sens = [10, 20, 10, 20]

    groups = _groups(fairness_evidence(preds, sens, favorable_label=1))
    assert {g["group"] for g in groups} == {10, 20}


# --------------------------------------------------------------------------
# favorable_label semantics
# --------------------------------------------------------------------------


def test_favorable_label_is_echoed_on_every_record():
    records = fairness_evidence([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=0)
    assert all(r["favorable_label"] == 0 for r in records)


def test_favorable_label_changes_counts_and_rates():
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5

    as_one = {g["group"]: g for g in _groups(fairness_evidence(preds, sens, favorable_label=1))}
    as_zero = {g["group"]: g for g in _groups(fairness_evidence(preds, sens, favorable_label=0))}

    assert as_one["A"]["favorable_count"] == 4
    assert as_zero["A"]["favorable_count"] == 1  # the complement
    assert as_one["A"]["selection_rate"] != as_zero["A"]["selection_rate"]

    # Group sizes are a property of the population, not of the label.
    assert as_one["A"]["group_count"] == as_zero["A"]["group_count"] == 5


def test_default_favorable_label_is_zero():
    """Omitting favorable_label must behave as 0, matching the credit model."""
    preds = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    sens = ["A"] * 5 + ["B"] * 5

    assert fairness_evidence(preds, sens) == fairness_evidence(
        preds, sens, favorable_label=0
    )


def test_favorable_label_none_raises():
    with pytest.raises(ValueError, match="favorable_label must not be None"):
        fairness_evidence([1, 0], ["A", "B"], favorable_label=None)


# --------------------------------------------------------------------------
# Canonical protected attribute name
# --------------------------------------------------------------------------


def test_canonical_attribute_name_on_every_record():
    sens = pd.Series(["A91", "A92", "A91", "A92"], name="personal_status_and_sex")
    records = fairness_evidence([1, 0, 1, 0], sens, favorable_label=1)

    assert all(r["protected_attribute"] == "personal_status_and_sex" for r in records)


def test_gender_and_sex_names_are_rejected_in_evidence():
    """Attribute 9 must never surface as a standalone gender/sex field."""
    for bad_name in ("gender", "Gender", "sex", "SEX"):
        sens = pd.Series(["A", "B", "A", "B"], name=bad_name)
        records = fairness_evidence([1, 0, 1, 0], sens, favorable_label=1)

        assert all(
            r["protected_attribute"] == "personal_status_and_sex" for r in records
        )
        assert all(
            r["protected_attribute"].lower() not in ("gender", "sex") for r in records
        )


def test_unnamed_sensitive_feature_uses_the_default_name():
    records = fairness_evidence([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=1)
    assert all(r["protected_attribute"] == "personal_status_and_sex" for r in records)


# --------------------------------------------------------------------------
# PENDING cases -- evidence without invented disparity
# --------------------------------------------------------------------------


def test_single_group_is_pending_but_still_reports_that_group():
    preds = [1, 0, 1, 1]
    sens = ["A", "A", "A", "A"]

    records = fairness_evidence(preds, sens, favorable_label=1)
    groups = _groups(records)
    summary = _summary(records)

    assert len(groups) == 1
    assert groups[0]["group"] == "A"
    assert groups[0]["group_count"] == 4
    assert groups[0]["favorable_count"] == 3
    assert groups[0]["selection_rate"] == 0.75

    assert summary["status"] == STATUS_PENDING
    assert summary["demographic_parity_diff"] == 0.0
    assert summary["disparate_impact_ratio"] == 1.0


def test_zero_maximum_selection_rate_is_pending_with_zero_rate_groups():
    """No group receives the favourable outcome: rates are 0, not a disparity."""
    preds = [0, 0, 0, 0]
    sens = ["A", "A", "B", "B"]

    records = fairness_evidence(preds, sens, favorable_label=1)
    groups = _groups(records)
    summary = _summary(records)

    assert len(groups) == 2
    assert all(g["favorable_count"] == 0 for g in groups)
    assert all(g["selection_rate"] == 0.0 for g in groups)
    assert all(g["group_count"] == 2 for g in groups)

    assert summary["status"] == STATUS_PENDING
    assert summary["demographic_parity_diff"] == 0.0
    assert summary["disparate_impact_ratio"] == 1.0


def test_pending_summary_matches_pending_report():
    preds = [0, 0, 0, 0]
    sens = ["A", "A", "B", "B"]

    report = fairness_report(preds, sens, favorable_label=1)
    summary = _summary(fairness_evidence(preds, sens, favorable_label=1))

    assert report["status"] == STATUS_PENDING
    assert summary["status"] == report["status"]
    assert summary["demographic_parity_diff"] == report["demographic_parity_diff"]
    assert summary["disparate_impact_ratio"] == report["disparate_impact_ratio"]


def test_a_group_with_no_favorable_outcomes_is_not_pending():
    """One group at zero is a real, maximal disparity -- not an unassessable run."""
    preds = [1, 1, 0, 0]
    sens = ["A", "A", "B", "B"]

    records = fairness_evidence(preds, sens, favorable_label=1)
    by_group = {g["group"]: g for g in _groups(records)}

    assert by_group["B"]["favorable_count"] == 0
    assert by_group["B"]["selection_rate"] == 0.0
    assert _summary(records)["status"] == STATUS_FAIL


# --------------------------------------------------------------------------
# Null handling and input validation (identical to fairness_report)
# --------------------------------------------------------------------------


def test_null_rows_are_dropped_before_counting():
    preds = [1, 0, None, 1, 0]
    sens = ["A", "A", "B", "B", None]
    # Surviving rows: A -> [1, 0]; B -> [1]

    by_group = {g["group"]: g for g in _groups(fairness_evidence(preds, sens, favorable_label=1))}

    assert by_group["A"]["group_count"] == 2
    assert by_group["A"]["favorable_count"] == 1
    assert by_group["B"]["group_count"] == 1
    assert by_group["B"]["favorable_count"] == 1
    # The dropped rows are not counted anywhere.
    assert sum(g["group_count"] for g in by_group.values()) == 3


def test_non_positional_pandas_index_is_rejected():
    """The Phase 3 alignment guard applies to evidence too."""
    preds = [1, 1, 0, 0]
    sens = pd.Series(["A", "A", "B", "B"], index=[2, 3, 4, 5])

    with pytest.raises(ValueError, match="non-positional index"):
        fairness_evidence(preds, sens, favorable_label=1)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="Length mismatch"):
        fairness_evidence([1, 0, 1], ["A", "B"], favorable_label=1)


def test_empty_inputs_raise():
    with pytest.raises(ValueError, match="must not be empty"):
        fairness_evidence([], [], favorable_label=1)


def test_none_inputs_raise():
    with pytest.raises(ValueError, match="must not be None"):
        fairness_evidence(None, ["A", "B"], favorable_label=1)
    with pytest.raises(ValueError, match="must not be None"):
        fairness_evidence([1, 0], None, favorable_label=1)


def test_all_null_rows_raise():
    with pytest.raises(ValueError, match="No valid rows remaining"):
        fairness_evidence([None, None], ["A", None], favorable_label=1)


def test_evidence_and_report_raise_on_the_same_inputs():
    """Validation must not diverge between the two entry points."""
    bad_cases = [
        (None, ["A", "B"]),
        ([1, 0], None),
        ([1, 0, 1], ["A", "B"]),
        ([], []),
        ([None, None], ["A", None]),
        ([1, 1, 0, 0], pd.Series(["A", "A", "B", "B"], index=[2, 3, 4, 5])),
    ]
    for preds, sens in bad_cases:
        with pytest.raises(ValueError):
            fairness_report(preds, sens, favorable_label=1)
        with pytest.raises(ValueError):
            fairness_evidence(preds, sens, favorable_label=1)


# --------------------------------------------------------------------------
# Record shape guarantees
# --------------------------------------------------------------------------


def test_exact_record_key_sets():
    records = fairness_evidence([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=1)

    for group in _groups(records):
        assert set(group.keys()) == GROUP_KEYS
    assert set(_summary(records).keys()) == SUMMARY_KEYS


def test_is_mock_is_false_on_every_record():
    records = fairness_evidence([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=1)
    assert all(r["is_mock"] is False for r in records)


def test_status_is_in_the_approved_vocabulary():
    records = fairness_evidence([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=1)
    assert _summary(records)["status"] in VALID_STATUSES


def test_no_instance_id_anywhere():
    """Fairness evidence is population-level; record identity has no place here.

    A selection rate describes a group, so attaching a record identifier would
    invite a reader to treat a group statistic as a claim about one applicant.
    """
    records = fairness_evidence([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=1)

    for record in records:
        assert "instance_id" not in record
        assert "instance_ids" not in record
        assert not any("instance" in key for key in record)
        assert not any("row_index" in key for key in record)


def test_counts_are_plain_python_ints():
    """Counts must be JSON-native for the downstream reporting boundary."""
    records = fairness_evidence([1, 0, 1, 0], ["A", "A", "B", "B"], favorable_label=1)

    for group in _groups(records):
        assert type(group["group_count"]) is int
        assert type(group["favorable_count"]) is int
        assert type(group["selection_rate"]) is float


def test_evidence_is_deterministic():
    preds = [1, 0, 1, 0, 1, 1]
    sens = ["A", "A", "B", "B", "C", "C"]

    assert fairness_evidence(preds, sens, favorable_label=1) == fairness_evidence(
        preds, sens, favorable_label=1
    )


def test_real_attribute_9_shape_end_to_end():
    """A realistic Attribute 9 population produces coherent evidence."""
    sens = pd.Series(
        ["A93", "A92", "A93", "A94", "A91", "A92"] * 4,
        name="personal_status_and_sex",
    )
    preds = [0, 1, 0, 1, 0, 1] * 4

    records = fairness_evidence(preds, sens, favorable_label=0)
    groups = _groups(records)

    assert [g["group"] for g in groups] == ["A93", "A92", "A94", "A91"]
    assert sum(g["group_count"] for g in groups) == len(preds)
    assert _summary(records)["status"] in VALID_STATUSES
    assert _summary(records)["protected_attribute"] == "personal_status_and_sex"
