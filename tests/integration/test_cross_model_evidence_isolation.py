"""Phase 5: can two models' fairness evidence be told apart? Currently, no.

KNOWN LIMITATION — this whole file documents a gap, not correct behaviour.

Phase 5 requires evidence to be isolated by evidence type, model identity, and
assurance run identity. Today only the first of those three exists:

- ``fairness_evidence()`` records carry ``evidence_type``,
  ``protected_attribute``, ``group``, ``group_count``, ``favorable_count``,
  ``selection_rate``, ``favorable_label`` and ``is_mock`` -- and no model or
  run identity at all.
- ``app/report/generate.py``'s ``EVIDENCE_SECTION_BY_TYPE`` routes records by
  ``evidence_type`` alone.

So two models evaluated into one report have their ``fairness_group`` records
placed in the same ``fairness`` section, interleaved, with nothing in the data
distinguishing which model produced which record. The 5A/5B design spec flags
this as a pre-5D concern (B7: routing must move to
``(evidence_type, model_id)``).

These tests PASS today. They are tripwires: the day identity is added they will
start failing, which is the signal to update them and delete the parts that
assert absence. They deliberately do not assert that the current state is
correct, and nothing here modifies ``build_evidence_records()`` or the routing
table.

Owner of the fix: report/orchestration integration (Khushi), 5D.
"""
import pytest

from app.api.orchestration import (
    build_evidence_records,
    compute_real_explainability,
)
from app.models.model import RandomForestAdapter, predict_batch

FAIRNESS_GROUP = "fairness_group"
FAIRNESS_SUMMARY = "fairness_summary"

# The fields a fairness_group record actually carries today.
EXPECTED_GROUP_FIELDS = {
    "evidence_type",
    "protected_attribute",
    "group",
    "group_count",
    "favorable_count",
    "selection_rate",
    "favorable_label",
    "is_mock",
}

IDENTITY_FIELDS = ("model_id", "model_version", "assurance_run_id", "adapter_id")


@pytest.fixture(scope="module")
def lr_evidence() -> list:
    """Fairness evidence records from a real Logistic Regression run."""
    model_output = predict_batch()
    explanation = compute_real_explainability(model_output, method="shap")
    return build_evidence_records(model_output, explanation, method="shap")


@pytest.fixture(scope="module")
def rf_evidence() -> list:
    """Fairness evidence records from a real Random Forest run."""
    model_output = predict_batch(adapter=RandomForestAdapter.load_default())
    explanation = compute_real_explainability(model_output, method="shap")
    return build_evidence_records(model_output, explanation, method="shap")


def _fairness_groups(records: list) -> list:
    return [r for r in records if r.get("evidence_type") == FAIRNESS_GROUP]


# ---------------------------------------------------------------------------
# The gap itself
# ---------------------------------------------------------------------------


def test_known_limitation_fairness_group_records_carry_no_model_identity(
    lr_evidence: list,
):
    """No model_id, model_version, assurance_run_id or adapter_id on a record."""
    groups = _fairness_groups(lr_evidence)
    assert groups, "expected at least one fairness_group record"

    for record in groups:
        assert set(record.keys()) == EXPECTED_GROUP_FIELDS
        for field in IDENTITY_FIELDS:
            assert field not in record


def test_known_limitation_fairness_summary_records_carry_no_model_identity(
    lr_evidence: list,
):
    summaries = [
        r for r in lr_evidence if r.get("evidence_type") == FAIRNESS_SUMMARY
    ]
    assert summaries, "expected a fairness_summary record"

    for record in summaries:
        for field in IDENTITY_FIELDS:
            assert field not in record


def test_known_limitation_two_models_evidence_is_indistinguishable_by_identity(
    lr_evidence: list, rf_evidence: list
):
    """The consequence: identity alone cannot separate the two models' records.

    Both runs produce the same evidence_type, the same protected attribute, and
    the same group labels and counts (the models scored the same rows). Only
    the selection rates differ -- and a rate is a measurement, not an
    identifier, so it cannot be used to attribute a record to a model.
    """
    lr_groups = _fairness_groups(lr_evidence)
    rf_groups = _fairness_groups(rf_evidence)

    assert lr_groups and rf_groups

    # Identical key sets, so no structural discriminator exists.
    assert {frozenset(r.keys()) for r in lr_groups} == {
        frozenset(r.keys()) for r in rf_groups
    }

    # Identical evidence_type and group identity across both models.
    assert [r["group"] for r in lr_groups] == [r["group"] for r in rf_groups]
    assert [r["group_count"] for r in lr_groups] == [
        r["group_count"] for r in rf_groups
    ]

    # Pooling both models' records loses the distinction entirely: the merged
    # list contains no field that says which model any record came from.
    merged = lr_groups + rf_groups
    assert len(merged) == len(lr_groups) + len(rf_groups)
    for field in IDENTITY_FIELDS:
        assert not any(field in record for record in merged)


def test_known_limitation_models_did_produce_different_measurements(
    lr_evidence: list, rf_evidence: list
):
    """Why the missing identity matters: the two runs are genuinely different.

    If the models agreed, mixing their evidence would be harmless. They do not
    -- the selection rates differ -- so an interleaved fairness section would
    present two models' rates as one population's.
    """
    lr_rates = [r["selection_rate"] for r in _fairness_groups(lr_evidence)]
    rf_rates = [r["selection_rate"] for r in _fairness_groups(rf_evidence)]

    assert lr_rates != rf_rates


def test_known_limitation_routing_table_keys_on_evidence_type_alone(
    lr_evidence: list, rf_evidence: list
):
    """Both models' fairness records route to the same report section.

    Read from the real routing table rather than restating it, so this tracks
    ``app/report/generate.py`` if the mapping changes.
    """
    from app.report.generate import EVIDENCE_SECTION_BY_TYPE

    sections = {
        EVIDENCE_SECTION_BY_TYPE[r["evidence_type"]]
        for r in _fairness_groups(lr_evidence) + _fairness_groups(rf_evidence)
    }

    # One destination for both models -- the interleaving risk, stated plainly.
    assert sections == {"fairness"}


# ---------------------------------------------------------------------------
# What IS correctly isolated today, so the gap is not overstated
# ---------------------------------------------------------------------------


def test_population_level_fairness_evidence_still_carries_no_instance_id(
    lr_evidence: list, rf_evidence: list
):
    """The Phase 3 rule holds for both models: group evidence is not per-record."""
    for record in _fairness_groups(lr_evidence) + _fairness_groups(rf_evidence):
        assert "instance_id" not in record
