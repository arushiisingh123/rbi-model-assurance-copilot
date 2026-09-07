"""Fairness evaluation module (owner: Arushi).

Calculates demographic parity difference and disparate impact ratio across
categories using pure pandas and numpy. Thresholds are imported from
app.config.thresholds -- this module never defines its own.

Note on sensitive feature representation:
The sensitive feature reflects raw combined categories (e.g. Attribute 9 of
German Credit data). It is not a standalone gender/sex column. No derived
sex grouping is applied here; the observed categories are used as-is.

Note on the favourable outcome:
Which prediction value counts as "favourable" is domain-specific and is never
inferred from the data. It is supplied by the caller via ``favorable_label``.
The default matches the current credit model, where the target is encoded
0 = GOOD (favourable) and 1 = BAD. See docs/module-interfaces.md.

Note on is_mock:
``is_mock=False`` means the metric calculation itself is real (real arithmetic
on real inputs). It does NOT mean the result is verified regulatory evidence.
Mapping a metric to any RBI requirement is the compliance module's job, and the
thresholds used here are project/industry conventions, not RBI requirements
(docs/thresholds.md).
"""

from typing import Any

import pandas as pd

from app.config.thresholds import (
    STATUS_PENDING,
    classify_disparate_impact,
)
from app.fairness.grouping import validate_fairness_inputs

DEFAULT_PROTECTED_ATTRIBUTE = "personal_status_and_sex"

# The favourable outcome for the current credit model.
# The model target is encoded 0 = GOOD (favourable) and 1 = BAD, so the
# favourable prediction value is 0. Callers evaluating a model with different
# label semantics must pass favorable_label explicitly.
DEFAULT_FAVORABLE_LABEL = 0

# Number of decimal places the reported metrics are rounded to. The status is
# derived from the rounded (reported) ratio so the number shown in a report and
# the status attached to it can never disagree.
_ROUNDING_DP = 4


def _resolve_protected_attribute_name(sensitive_feature: Any) -> str:
    """Derive the reported protected-attribute name.

    Uses a pandas Series name when available. The literal name "gender" is
    rejected: Attribute 9 combines personal/marital status with sex and must
    never be reported as a standalone gender field (docs/decisions.md,
    "Phase 1 credit-scoring dataset and fairness attribute").
    """
    if isinstance(sensitive_feature, pd.Series) and sensitive_feature.name:
        attr_name = str(sensitive_feature.name)
        if attr_name.lower() in ("gender", "sex"):
            return DEFAULT_PROTECTED_ATTRIBUTE
        return attr_name
    return DEFAULT_PROTECTED_ATTRIBUTE


def _analyse_fairness(
    predictions: Any,
    sensitive_feature: Any,
    favorable_label: Any,
) -> dict:
    """Run the shared fairness arithmetic once.

    Single source of truth for both ``fairness_report()`` (the Phase 2 public
    contract) and ``app.fairness.evidence.fairness_evidence()`` (the Phase 3
    evidence layer), so the two can never disagree about a rate, a metric, or a
    status. Neither caller recomputes anything.

    Returns:
        protected_attribute (str): resolved reported name.
        groups (list[dict]): per-group ``group`` / ``group_count`` /
            ``favorable_count`` / ``selection_rate``, in first-observed order.
        demographic_parity_diff (float): rounded aggregate.
        disparate_impact_ratio (float): rounded aggregate.
        status (str): PASS, WARNING, FAIL, or PENDING.

    ``groups`` is populated whenever the inputs validate -- including both
    PENDING cases. A comparison that could not be made still has observable
    group counts, and withholding them would leave a report with nothing to
    show for an input that was perfectly readable.
    """
    if favorable_label is None:
        raise ValueError(
            "favorable_label must not be None. The favourable outcome is "
            "domain-specific and is never inferred from the data; for the "
            "current credit model 0 = GOOD (favourable) and 1 = BAD."
        )

    # Resolve the reported name before cleaning, so a Series name survives.
    attr_name = _resolve_protected_attribute_name(sensitive_feature)

    clean_preds, clean_sens = validate_fairness_inputs(predictions, sensitive_feature)

    # pd.unique preserves first-observed order; metrics and evidence share it.
    observed_groups = pd.unique(clean_sens)

    groups = []
    selection_rates = []
    for group in observed_groups:
        member_preds = clean_preds[clean_sens == group]
        group_count = int(len(member_preds))
        favorable_count = int((member_preds == favorable_label).sum())
        # Selection rate: P(pred == favorable_label | group), i.e. favourable
        # predictions over group size.
        rate = favorable_count / group_count
        selection_rates.append(rate)
        groups.append(
            {
                "group": group,
                "group_count": group_count,
                "favorable_count": favorable_count,
                "selection_rate": round(rate, _ROUNDING_DP),
            }
        )

    max_rate = max(selection_rates)
    min_rate = min(selection_rates)

    # Two ways the comparison cannot be made: fewer than two groups to compare,
    # or no group receives the favourable outcome at all (leaving the ratio
    # undefined). Both report neutral aggregates with PENDING -- never PASS.
    if len(observed_groups) < 2 or max_rate == 0.0:
        return {
            "protected_attribute": attr_name,
            "groups": groups,
            "demographic_parity_diff": 0.0,
            "disparate_impact_ratio": 1.0,
            "status": STATUS_PENDING,
        }

    dp_diff = round(float(max_rate - min_rate), _ROUNDING_DP)
    di_ratio = round(float(min_rate / max_rate), _ROUNDING_DP)

    return {
        "protected_attribute": attr_name,
        "groups": groups,
        "demographic_parity_diff": dp_diff,
        "disparate_impact_ratio": di_ratio,
        # Classify the reported value, not the pre-rounding value, so the ratio
        # shown in a report always matches the status shown beside it.
        "status": classify_disparate_impact(di_ratio),
    }


def fairness_report(
    predictions: Any = None,
    sensitive_feature: Any = None,
    favorable_label: Any = DEFAULT_FAVORABLE_LABEL,
) -> dict:
    """Calculate demographic parity difference and disparate impact ratio.

    Args:
        predictions: Model predictions (binary or discrete labels). Paired with
            ``sensitive_feature`` by position: element i of each describes the
            same record. A pandas input whose index is not ``0..n-1`` is
            rejected rather than silently re-paired by index -- pass
            ``.reset_index(drop=True)`` or ``.to_numpy()``.
        sensitive_feature: Protected / sensitive attribute categories. Used
            as-is; no codebook or derived sex grouping is applied. Same
            positional pairing rule as ``predictions``.
        favorable_label: The prediction value that counts as the favourable
            outcome. Defaults to ``DEFAULT_FAVORABLE_LABEL`` (0), matching the
            current credit model where 0 = GOOD and 1 = BAD. Pass explicitly
            when evaluating a model with different label semantics.

    Returns:
        Dictionary with keys:
            protected_attribute (str): Name of the attribute evaluated.
            demographic_parity_diff (float): max selection rate - min selection rate.
            disparate_impact_ratio (float): min selection rate / max selection rate.
            status (str): PASS, WARNING, FAIL, or PENDING, derived solely from
                the reported disparate impact ratio via
                app.config.thresholds.classify_disparate_impact.
            is_mock (bool): False -- the calculation is real.

    Raises:
        ValueError: if favorable_label is None, if either input is None, if
            either pandas input has a non-positional index, if the inputs have
            mismatched lengths, or if no valid rows remain.

    Notes:
        PENDING is returned when the assessment cannot be performed meaningfully:
        fewer than two groups are present, or no group receives the favourable
        outcome at all (a zero maximum selection rate makes the ratio undefined).
        Demographic parity difference is reported as a metric only; it is never
        classified, because no threshold is defined for it (docs/thresholds.md).
    """
    analysis = _analyse_fairness(predictions, sensitive_feature, favorable_label)

    # Exactly the five approved keys -- the Phase 3 evidence layer
    # (app.fairness.evidence) exposes the per-group detail instead, so this
    # contract stays unchanged.
    return {
        "protected_attribute": analysis["protected_attribute"],
        "demographic_parity_diff": analysis["demographic_parity_diff"],
        "disparate_impact_ratio": analysis["disparate_impact_ratio"],
        "status": analysis["status"],
        "is_mock": False,
    }
