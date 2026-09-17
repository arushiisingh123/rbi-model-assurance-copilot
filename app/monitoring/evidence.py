"""Monitoring evidence records with model identity (owner: Arushi).

Turns a ``monitor_run()`` result into flat evidence records shaped like the
project's existing producers (``app.fairness.evidence.fairness_evidence``,
``app.explainability.evidence``): one dict per finding, each carrying its own
``evidence_type``, its own identity, and ``is_mock``.

NOT A SECOND EVIDENCE SYSTEM
    Same record shape, same ``evidence_type``-per-record convention, same
    identity fields (``model_id``, ``assurance_run_id``), same rule that
    identity keys are omitted rather than set to None. This adds monitoring
    records to the existing scheme; it does not introduce a parallel one.

    Nothing is calculated here. Every value is copied from the
    ``monitor_run()`` result, which copied it from the module that owns the
    calculation. No rounding, no re-classification, no re-derivation.

IDENTITY COMES FROM THE RESULT, NOT FROM THE CALLER
    ``model_id``, ``model_version`` and ``assurance_run_id`` are read from the
    result's own ``context`` -- the one ``monitor_run()`` validated and threaded
    through. There is deliberately no way to pass a different identity
    alongside a result: an override is exactly how a Random Forest measurement
    acquires a Logistic Regression label, and prediction drift is the channel
    whose value genuinely depends on which model produced it.

    ``model_version`` is included here although the existing producers carry
    only ``model_id`` and ``assurance_run_id``. Monitoring compares one model
    ACROSS TIME, so distinguishing two versions of the same ``model_id`` is not
    optional for this evidence -- a drift result from v1 and one from v2 are
    different measurements even though the ``model_id`` matches.

!!! NOT YET ROUTABLE INTO THE REPORT !!!
    The report DOES already have a drift section. ``app.report.generate``
    builds a section keyed ``"drift"``, titled "Data & Prediction Drift
    Detection", with its own RAG query and relevance keywords, populated by
    ``_extract_drift_finding()`` from the flat ``drift_report()`` output as a
    Layer 1 technical finding. So the gap is NOT a missing section.

    The gap is the SUPPORTING-EVIDENCE routing table.
    ``app.report.generate.EVIDENCE_SECTION_BY_TYPE`` maps each
    ``evidence_type`` to a section and RAISES ``ValueError`` on a type it does
    not cover; it currently covers only the four explainability/fairness types.
    No evidence_type maps to ``"drift"`` at all -- so that section today
    carries a Layer 1 finding and retrieved evidence, but no per-record
    supporting evidence.

    So these records must NOT be appended to the list handed to
    ``build_report()`` until that table gains entries for them. That is a
    one-table change in ``app/report/generate.py`` (Nidhi's module) rather
    than a new report section, but it is still not this module's call: a lane
    does not extend another module's routing table unilaterally, and which
    section each of the five types belongs to -- ``"drift"`` for the drift
    channels, ``"drift"`` or ``"fairness"`` for ``fairness_monitor_summary``
    -- is a decision for the report's owner.

    The records are useful now regardless: they are the monitoring evidence
    contract, consumable directly by the API and dashboard layers, and they
    are what the existing drift section's supporting evidence would be
    populated from once the table is extended.
    ``tests/monitoring/test_monitoring_identity.py`` pins this constraint so it
    is discovered by a named test rather than by a crash inside report
    generation.
"""

from typing import Any, Dict, List, Mapping, Optional

from app.monitoring.monitor import (
    CHANNEL_FAIRNESS,
    CHANNEL_FEATURE_DRIFT,
    CHANNEL_PREDICTION_DRIFT,
)

__all__ = [
    "EVIDENCE_TYPE_FEATURE_DRIFT",
    "EVIDENCE_TYPE_FAIRNESS_MONITOR",
    "EVIDENCE_TYPE_MONITORING_SUMMARY",
    "EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL",
    "EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE",
    "MONITORING_EVIDENCE_TYPES",
    "monitoring_evidence",
]

EVIDENCE_TYPE_FEATURE_DRIFT = "feature_drift_summary"
EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL = "prediction_drift_label"
EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE = "prediction_drift_score"
EVIDENCE_TYPE_FAIRNESS_MONITOR = "fairness_monitor_summary"
EVIDENCE_TYPE_MONITORING_SUMMARY = "monitoring_summary"

# The label and score channels of prediction drift get SEPARATE records rather
# than one combined record. They are measured with different metrics over
# different representations (categorical class frequencies vs a continuous score
# distribution), and one may be unavailable while the other is measured. A
# single record would have to either merge two incomparable PSI values or
# silently drop one.
MONITORING_EVIDENCE_TYPES = (
    EVIDENCE_TYPE_FEATURE_DRIFT,
    EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL,
    EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE,
    EVIDENCE_TYPE_FAIRNESS_MONITOR,
    EVIDENCE_TYPE_MONITORING_SUMMARY,
)


def _identity(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Identity fields every record in one monitoring run carries.

    Absent fields are omitted rather than set to None, matching
    ``fairness_evidence()``: the existing evidence tests assert an EXACT key
    set per record, so an always-present None field would be a contract break.
    ``monitor_run()`` has already refused a context missing any of these, so in
    practice all three are present.
    """
    identity = {
        "model_id": context.get("model_id"),
        "model_version": context.get("model_version"),
        "assurance_run_id": context.get("assurance_run_id"),
    }
    adapter_id = context.get("adapter_id")
    if adapter_id is not None:
        identity["adapter_id"] = adapter_id
    return {k: v for k, v in identity.items() if v is not None}


def _window_labels(result: Mapping[str, Any]) -> Dict[str, Any]:
    """The two window labels, so a record names the periods it compared."""
    return {
        "reference_window_id": result.get("reference_window_id"),
        "current_window_id": result.get("current_window_id"),
    }


def monitoring_evidence(monitor_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build evidence records from one ``monitor_run()`` result.

    Args:
        monitor_result: A ``monitor_run()`` output. Its ``context`` supplies the
            identity on every record; see the module docstring for why identity
            cannot be passed separately.

    Returns:
        A list of flat evidence records, one per measured channel, followed by a
        single ``monitoring_summary`` record. Every record carries
        ``evidence_type``, the run's identity, both window labels, and
        ``is_mock``.

        A channel that did not run produces NO record. An absent measurement is
        not evidence, and emitting a zero-valued record for it would state a
        finding that was never measured. The ``monitoring_summary`` record still
        reports that channel's PENDING status, so the gap stays visible.

        Read-only with respect to its input: no key of ``monitor_result`` or of
        any nested channel result is added, renamed, or modified.

    Raises:
        ValueError: if ``monitor_result`` is not a mapping or carries no
            ``context``.
    """
    if not isinstance(monitor_result, Mapping):
        raise ValueError(
            "monitor_result must be a monitor_run() output mapping, got "
            f"{type(monitor_result).__name__}."
        )

    context = monitor_result.get("context")
    if not isinstance(context, Mapping) or not context:
        raise ValueError(
            "monitor_result carries no 'context', so its records could not be "
            "attributed to a model or run. Pass a monitor_run() output."
        )

    identity = _identity(context)
    windows = _window_labels(monitor_result)
    base = {**identity, **windows, "is_mock": False}

    records: List[Dict[str, Any]] = []

    feature_drift = monitor_result.get(CHANNEL_FEATURE_DRIFT)
    if feature_drift is not None:
        records.append(
            {
                "evidence_type": EVIDENCE_TYPE_FEATURE_DRIFT,
                "psi": feature_drift.get("psi"),
                "ks_statistic": feature_drift.get("ks_statistic"),
                "status": feature_drift.get("status"),
                "features_evaluated": feature_drift.get("features_evaluated"),
                **base,
            }
        )

    prediction_drift = monitor_result.get(CHANNEL_PREDICTION_DRIFT)
    if prediction_drift is not None:
        records.append(
            {
                "evidence_type": EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL,
                # Named to state the metric, not just "psi": this is a
                # categorical PSI over class frequencies and is NOT the same
                # measurement as the quantile PSI in the two records around it.
                "metric": "categorical_psi",
                "label_psi": prediction_drift.get("label_psi"),
                "status": prediction_drift.get("label_status"),
                "classes_evaluated": prediction_drift.get("classes_evaluated"),
                "per_class": prediction_drift.get("per_class"),
                **base,
            }
        )
        # Emitted only when a score distribution was actually measured. A model
        # with no probability capability gets no score record at all, rather
        # than a record whose PSI is None or zero.
        if prediction_drift.get("score_psi") is not None:
            records.append(
                {
                    "evidence_type": EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE,
                    "metric": "quantile_psi",
                    "score_psi": prediction_drift.get("score_psi"),
                    "score_ks_statistic": prediction_drift.get("score_ks_statistic"),
                    "status": prediction_drift.get("score_status"),
                    "score_availability": prediction_drift.get("score_availability"),
                    **base,
                }
            )

    fairness = monitor_result.get(CHANNEL_FAIRNESS)
    if fairness is not None:
        records.append(
            {
                "evidence_type": EVIDENCE_TYPE_FAIRNESS_MONITOR,
                "protected_attribute": fairness.get("protected_attribute"),
                "disparate_impact_ratio": fairness.get("disparate_impact_ratio"),
                "demographic_parity_diff": fairness.get("demographic_parity_diff"),
                "status": fairness.get("status"),
                **base,
            }
        )

    records.append(
        {
            "evidence_type": EVIDENCE_TYPE_MONITORING_SUMMARY,
            "status": monitor_result.get("monitoring_status"),
            "channel_status": monitor_result.get("channel_status"),
            "alert_count": len(monitor_result.get("alerts") or []),
            **base,
        }
    )

    return records
