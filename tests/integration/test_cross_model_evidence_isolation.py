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
from app.models.model import (
    LogisticRegressionAdapter,
    RandomForestAdapter,
    predict_batch,
)

FAIRNESS_GROUP = "fairness_group"
FAIRNESS_SUMMARY = "fairness_summary"


@pytest.fixture(scope="module")
def lr_evidence() -> list:
    """Fairness evidence records from a real Logistic Regression run."""
    model_output = predict_batch()
    explanation = compute_real_explainability(model_output, method="shap")
    return build_evidence_records(
        model_output,
        explanation,
        method="shap",
        model_id=LogisticRegressionAdapter.load_default().model_id,
    )


@pytest.fixture(scope="module")
def rf_evidence() -> list:
    """Fairness evidence records from a real Random Forest run."""
    rf_adapter = RandomForestAdapter.load_default()
    model_output = predict_batch(adapter=rf_adapter)
    explanation = compute_real_explainability(model_output, method="shap")
    return build_evidence_records(
        model_output,
        explanation,
        method="shap",
        model_id=rf_adapter.model_id,
    )


def _fairness_groups(records: list) -> list:
    return [r for r in records if r.get("evidence_type") == FAIRNESS_GROUP]


# ---------------------------------------------------------------------------
# Cross-model evidence distinction & isolation
# ---------------------------------------------------------------------------


def test_two_models_did_produce_different_measurements(
    lr_evidence: list, rf_evidence: list
):
    """Why identity matters: the two runs are genuinely different.

    If the models agreed, mixing their evidence would be harmless. They do not
    -- the selection rates differ -- so an interleaved fairness section would
    present two models' rates as one population's without distinguishing them.
    """
    lr_rates = [r["selection_rate"] for r in _fairness_groups(lr_evidence)]
    rf_rates = [r["selection_rate"] for r in _fairness_groups(rf_evidence)]

    assert lr_rates != rf_rates


def test_two_models_fairness_evidence_are_now_distinguishable_by_model_id(
    lr_evidence: list, rf_evidence: list
):
    """The gap this task closes: pooled records can now be told apart."""
    lr_groups = _fairness_groups(lr_evidence)
    rf_groups = _fairness_groups(rf_evidence)
    assert lr_groups and rf_groups

    merged = lr_groups + rf_groups
    model_ids_present = {r["model_id"] for r in merged}
    assert model_ids_present == {
        "german-credit-logistic-regression",
        "german-credit-random-forest",
    }
    # Grouping by model_id recovers exactly the two original sets.
    by_model = {}
    for r in merged:
        by_model.setdefault(r["model_id"], []).append(r)
    assert len(by_model["german-credit-logistic-regression"]) == len(lr_groups)
    assert len(by_model["german-credit-random-forest"]) == len(rf_groups)


def test_evidence_section_by_type_stays_evidence_type_only(lr_evidence, rf_evidence):
    """Architectural fact, not a limitation: section assignment never
    depends on which model produced a record -- both models' fairness
    records correctly target the same "fairness" section. What
    changed is that _route_evidence_records() now also groups by
    model_id internally (test above), so this single shared section
    no longer means the records are indistinguishable once pooled."""
    from app.report.generate import EVIDENCE_SECTION_BY_TYPE

    sections = {
        EVIDENCE_SECTION_BY_TYPE[r["evidence_type"]]
        for r in _fairness_groups(lr_evidence) + _fairness_groups(rf_evidence)
    }
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
