"""Fairness on an arbitrary feature space (owner: Arushi).

WHY THIS FILE EXISTS
    ``tests/fairness/test_fairness.py`` proves the arithmetic is correct on
    German Credit's Attribute 9. This file proves something different: that the
    fairness layer carries no German Credit assumption at all, and computes a
    correct result for a model whose feature space shares nothing with it --
    PROVIDED the protected attribute is supplied explicitly.

    That proviso is the rule, not a limitation to be engineered away. Which
    attribute is protected is a regulatory and governance decision. A fairness
    module that picked one by scanning column names would be inventing a
    compliance judgement, and it would pick differently for every model.

THE RULE, STATED ONCE
    - The protected attribute is ALWAYS supplied by the caller.
    - It is NEVER inferred from feature names, dtypes, or cardinality.
    - ``personal_status_and_sex`` is German Credit's attribute, not a global
      default for every model the platform evaluates.
    - ``favorable_label`` is likewise domain-specific and never inferred.

    ``app/monitoring/monitor.py`` enforces the same rule at the monitoring
    layer: omit ``protected_attribute`` and the fairness channel reports
    PENDING rather than guessing.

NOT DUPLICATED HERE
    Threshold bands, boundary behaviour, rounding, PENDING semantics, null
    handling, positional pairing and the group schema -- all covered by
    ``test_fairness.py`` and ``test_fairness_evidence.py``. This file only
    varies the FEATURE SPACE and the LABEL SPACE.
"""
import pandas as pd
import pytest

from app.config.thresholds import STATUS_PENDING, VALID_STATUSES
from app.fairness.evidence import fairness_evidence
from app.fairness.fairness import DEFAULT_PROTECTED_ATTRIBUTE, fairness_report

# A feature space with nothing in common with German Credit. Deliberately
# written out here rather than imported, so this file keeps testing "some
# arbitrary model" even if the synthetic bank's schema later changes.
REGIONS = ["north", "south", "east", "west"]


def _bank_like_frame(n_per_region: int = 10) -> pd.DataFrame:
    """A small applicant table in a non-German-Credit feature space."""
    rows = []
    for index, region in enumerate(REGIONS):
        for row in range(n_per_region):
            rows.append(
                {
                    "region": region,
                    "employment_type": "salaried" if row % 2 == 0 else "self_employed",
                    "annual_income": 30_000 + 1_000 * row,
                    "credit_utilization_ratio": 0.1 + 0.05 * row,
                    # Region 0 is approved most often, region 3 least -- a real,
                    # deliberate disparity for the metric to find.
                    "prediction": 0 if row >= index else 1,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Structural: fairness CANNOT infer an attribute, because it never sees one
# ---------------------------------------------------------------------------


def test_fairness_report_never_receives_a_feature_matrix():
    """The strongest form of "never inferred": it has nothing to infer from.

    ``fairness_report()`` takes predictions and ONE sensitive feature. There is
    no feature matrix, no dataset, and no model in its signature, so there is no
    mechanism by which it could select a protected attribute even if it wanted
    to. Any selection has already happened in the caller.
    """
    import inspect

    parameters = set(inspect.signature(fairness_report).parameters)
    assert parameters == {"predictions", "sensitive_feature", "favorable_label"}
    for forbidden in ("feature_matrix", "data", "model", "adapter", "dataset"):
        assert forbidden not in parameters


def test_fairness_module_reads_no_dataset_and_loads_no_model():
    """No import path from fairness to the model or dataset layers."""
    from pathlib import Path

    source = Path("app/fairness/fairness.py").read_text(encoding="utf-8")
    for forbidden in (
        "load_dataset",
        "predict_batch",
        "from app.models",
        "import app.models",
        "read_csv",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Behavioural: a correct result on a feature space fairness has never seen
# ---------------------------------------------------------------------------


def test_fairness_computes_on_an_arbitrary_protected_attribute():
    frame = _bank_like_frame()

    result = fairness_report(
        predictions=frame["prediction"],
        sensitive_feature=frame["region"],
        favorable_label=0,
    )

    assert result["protected_attribute"] == "region"
    assert result["status"] in VALID_STATUSES
    assert {group["group"] for group in result["groups"]} == set(REGIONS)
    assert len(result["groups"]) == 4
    # The deliberate disparity is found: north is favoured over west.
    rates = {group["group"]: group["selection_rate"] for group in result["groups"]}
    assert rates["north"] > rates["west"]


def test_a_second_attribute_in_the_same_space_is_evaluated_independently():
    """Which attribute you ask about changes the answer -- as it must."""
    frame = _bank_like_frame()

    by_region = fairness_report(
        predictions=frame["prediction"],
        sensitive_feature=frame["region"],
        favorable_label=0,
    )
    by_employment = fairness_report(
        predictions=frame["prediction"],
        sensitive_feature=frame["employment_type"],
        favorable_label=0,
    )

    assert by_region["protected_attribute"] == "region"
    assert by_employment["protected_attribute"] == "employment_type"
    assert by_region["disparate_impact_ratio"] != by_employment["disparate_impact_ratio"]


def test_fairness_works_on_a_non_numeric_label_space():
    """Neither the prediction values nor the favourable one need be 0/1.

    A model outside this project's credit convention may emit string decisions.
    Nothing in fairness requires integers -- but ``favorable_label`` must then
    be supplied, because the default (0) would match nothing.
    """
    decisions = pd.Series(
        ["approve", "decline", "approve", "decline", "approve", "approve"],
        name="decision",
    )
    branch = pd.Series(["A", "A", "A", "B", "B", "B"], name="branch_code")

    result = fairness_report(
        predictions=decisions,
        sensitive_feature=branch,
        favorable_label="approve",
    )

    assert result["protected_attribute"] == "branch_code"
    assert result["status"] in VALID_STATUSES
    # Rates are reported rounded to 4dp (the module's documented precision),
    # so the tolerance matches that rather than floating-point exactness.
    rates = {group["group"]: group["selection_rate"] for group in result["groups"]}
    assert rates["A"] == pytest.approx(2 / 3, abs=1e-4)
    assert rates["B"] == pytest.approx(2 / 3, abs=1e-4)


def test_the_wrong_favorable_label_is_pending_not_a_silent_pass():
    """Defaulting to 0 against a string label space measures nothing.

    It must not report PASS -- that would be a clean bill of health for a
    comparison that never happened.
    """
    decisions = pd.Series(["approve", "decline", "approve", "decline"], name="decision")
    branch = pd.Series(["A", "A", "B", "B"], name="branch_code")

    result = fairness_report(predictions=decisions, sensitive_feature=branch)

    assert result["status"] == STATUS_PENDING
    assert result["disparate_impact_ratio"] == 1.0


def test_more_than_two_groups_are_all_reported():
    """Nothing collapses an arbitrary attribute to a binary comparison."""
    frame = _bank_like_frame()

    result = fairness_report(
        predictions=frame["prediction"],
        sensitive_feature=frame["region"],
        favorable_label=0,
    )

    assert len(result["groups"]) == 4
    # The aggregates are taken across all four, not a chosen pair.
    rates = [group["selection_rate"] for group in result["groups"]]
    assert result["demographic_parity_diff"] == pytest.approx(
        max(rates) - min(rates), abs=1e-4
    )


def test_evidence_carries_the_generic_attribute_name():
    """The evidence layer must not relabel a non-German-Credit attribute."""
    frame = _bank_like_frame()

    records = fairness_evidence(
        predictions=frame["prediction"],
        sensitive_feature=frame["region"],
        favorable_label=0,
        model_id="some-external-model",
        assurance_run_id="run-42",
    )

    assert records
    for record in records:
        assert record["protected_attribute"] == "region"
        assert record["model_id"] == "some-external-model"
        assert record["assurance_run_id"] == "run-42"
    assert DEFAULT_PROTECTED_ATTRIBUTE not in repr(records)


# ---------------------------------------------------------------------------
# KNOWN HAZARD -- characterisation, not endorsement
# ---------------------------------------------------------------------------


def test_an_unnamed_sensitive_feature_is_labelled_german_credits_attribute():
    """TRIPWIRE. Documents current behaviour that is wrong for a generic model.

    ``_resolve_protected_attribute_name()`` falls back to
    ``DEFAULT_PROTECTED_ATTRIBUTE`` ("personal_status_and_sex") whenever the
    sensitive feature carries no pandas name. That was harmless when the
    platform had exactly one model and one dataset. It is not harmless now: a
    caller who passes an external model's region column as a plain list gets a
    result claiming it measured German Credit's Attribute 9.

    The measurement itself is correct -- only the LABEL on it is wrong -- but a
    wrong label on fairness evidence is exactly the kind of misattribution the
    Phase 5 identity work exists to prevent.

    NOT fixed by ``ModelAdapter.protected_attribute``. That upstream field
    (None = "not declared") decides WHICH attribute a caller asks about, and
    ``app.api.orchestration.compute_real_fairness()`` now uses it instead of a
    hardcoded column name. This fallback is a different layer: it decides what
    name is REPORTED once an unnamed sequence has already been handed to
    ``fairness_report()``. An adapter declaring its attribute correctly still
    hits this fallback if the caller passes the column as a plain list.

    This test asserts what the code does TODAY so the behaviour cannot change
    silently, and so the team can see it. It is NOT an endorsement. Changing
    the fallback would alter the frozen six-key contract and break
    ``test_fairness.py::test_non_series_sensitive_feature_uses_default_name``,
    so it needs a team decision rather than a unilateral edit from this lane.

    Safe usage, pinned by the test below: pass a NAMED Series -- which is what
    selecting a column from a DataFrame gives you for free, and what
    ``app/monitoring/monitor.py`` always does.
    """
    regions_without_a_name = ["north", "south", "north", "south"]
    predictions = [0, 1, 0, 1]

    result = fairness_report(
        predictions=predictions,
        sensitive_feature=regions_without_a_name,
        favorable_label=0,
    )

    assert result["protected_attribute"] == DEFAULT_PROTECTED_ATTRIBUTE
    assert result["protected_attribute"] != "region"


def test_selecting_the_column_from_a_frame_labels_it_correctly():
    """The safe usage, and the reason monitoring is not exposed to the hazard.

    ``frame[column]`` yields a Series whose ``name`` is the column name, so the
    attribute labels itself. ``app/monitoring/monitor.py::_fairness_channel``
    selects exactly this way.
    """
    frame = _bank_like_frame()

    for attribute in ("region", "employment_type"):
        result = fairness_report(
            predictions=frame["prediction"],
            sensitive_feature=frame[attribute],
            favorable_label=0,
        )
        assert result["protected_attribute"] == attribute


def test_reset_index_preserves_the_attribute_name():
    """Monitoring calls ``.reset_index(drop=True)`` before handing the column over.

    If that dropped the Series name, every monitored fairness result would be
    mislabelled via the fallback above. It does not -- pinned here because the
    monitoring layer depends on it.
    """
    frame = _bank_like_frame()
    sensitive = frame["region"].reset_index(drop=True)

    assert sensitive.name == "region"
    result = fairness_report(
        predictions=frame["prediction"],
        sensitive_feature=sensitive,
        favorable_label=0,
    )
    assert result["protected_attribute"] == "region"
