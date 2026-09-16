"""Phase 3 fairness evidence layer (owner: Arushi).

Turns one fairness assessment into flat, self-describing records suitable for
downstream reporting. ``fairness_report()`` stays exactly as approved in Phase 2
-- five keys, no additions. This module is the place the per-group detail lives.

All arithmetic comes from ``app.fairness.fairness._analyse_fairness``: the same
validation, the same positional pairing, the same null handling, the same
rounding, and the same status. Nothing is recomputed here, so evidence and the
Phase 2 report can never disagree.

Evidence is POPULATION-LEVEL, never per-record. A fairness metric describes a
group, not an individual: a selection rate is a property of a set of records,
and attaching a record identifier to it would invite a reader to treat a group
statistic as a statement about one applicant. There is deliberately no
``instance_id`` field anywhere in this module.

Two record types are produced:

- ``fairness_group``   one per observed group, with its counts and rate.
- ``fairness_summary`` one per assessment, carrying the aggregate metrics and
  status exactly as ``fairness_report()`` reports them.

``is_mock`` is ``False`` because the arithmetic is real. That says nothing about
regulatory standing: mapping any of this to an RBI requirement is the compliance
module's job, and the thresholds behind ``status`` are project conventions, not
RBI requirements (docs/thresholds.md).
"""

from typing import Any, Dict, List, Optional

from app.fairness.fairness import DEFAULT_FAVORABLE_LABEL, _analyse_fairness

__all__ = ["fairness_evidence"]

EVIDENCE_TYPE_GROUP = "fairness_group"
EVIDENCE_TYPE_SUMMARY = "fairness_summary"


def fairness_evidence(
    predictions: Any = None,
    sensitive_feature: Any = None,
    favorable_label: Any = DEFAULT_FAVORABLE_LABEL,
    *,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build group-level fairness evidence records.

    Args:
        predictions: Model predictions. Same contract as ``fairness_report()``:
            paired with ``sensitive_feature`` by position, and a pandas input
            whose index is not ``0..n-1`` is rejected.
        sensitive_feature: Protected / sensitive attribute categories, used
            as-is. Original category values and first-observed order are
            preserved in the output.
        favorable_label: The prediction value counting as favourable. Defaults
            to ``DEFAULT_FAVORABLE_LABEL`` (0), matching the current credit
            model where 0 = GOOD and 1 = BAD. Echoed on every record so a
            reader never has to guess which polarity produced a rate.
        model_id: Additive (Phase 5D). Included on every returned record only
            when provided.
        assurance_run_id: Additive (Phase 5D). Included on every returned
            record only when provided.

    Returns:
        One ``fairness_group`` record per observed group, in first-observed
        order, followed by a single ``fairness_summary`` record.

        In the PENDING cases -- fewer than two groups, or no group receiving the
        favourable outcome -- the group records are still returned, and the
        summary carries the same PENDING status and neutral aggregates that
        ``fairness_report()`` returns. No disparity is invented.

        model_id / assurance_run_id are additive (Phase 5D): included on
        every returned record only when provided, using the same
        None-filtering already required elsewhere in this codebase for
        an identical reason -- test_exact_record_key_sets asserts an
        EXACT key set on every record when called without these
        arguments, so an always-present None field would break it.
        Omitted, output is byte-identical to before this parameter
        existed.

    Raises:
        ValueError: on exactly the same conditions as ``fairness_report()``
            (``favorable_label`` of None, a None input, a non-positional pandas
            index, mismatched lengths, empty input, or no valid rows).
    """
    analysis = _analyse_fairness(predictions, sensitive_feature, favorable_label)
    protected_attribute = analysis["protected_attribute"]

    identity = {
        k: v
        for k, v in {"model_id": model_id, "assurance_run_id": assurance_run_id}.items()
        if v is not None
    }

    records: List[Dict[str, Any]] = [
        {
            "evidence_type": EVIDENCE_TYPE_GROUP,
            "protected_attribute": protected_attribute,
            "group": group["group"],
            "group_count": group["group_count"],
            "favorable_count": group["favorable_count"],
            "selection_rate": group["selection_rate"],
            "favorable_label": favorable_label,
            "is_mock": False,
            **identity,
        }
        for group in analysis["groups"]
    ]

    records.append(
        {
            "evidence_type": EVIDENCE_TYPE_SUMMARY,
            "protected_attribute": protected_attribute,
            "demographic_parity_diff": analysis["demographic_parity_diff"],
            "disparate_impact_ratio": analysis["disparate_impact_ratio"],
            "status": analysis["status"],
            "favorable_label": favorable_label,
            "is_mock": False,
            **identity,
        }
    )

    return records
