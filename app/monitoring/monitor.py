"""Monitoring run: feature drift, prediction drift, and fairness over two windows (owner: Arushi).

WHAT THIS DOES, AND WHAT IT REFUSES TO DO
    ``monitor_run()`` evaluates the three monitoring channels over one
    reference/current window pair and reports one overall status. It is pure
    coordination: every number it reports is produced by the module that already
    owns that calculation --

        feature/data drift   app.drift.drift.drift_report()
        prediction drift     app.drift.prediction_drift.prediction_drift_report()
        fairness             app.fairness.fairness.fairness_report()

    Nothing is recalculated, re-rounded, re-classified, or reinterpreted here,
    and no threshold is defined here. ``monitoring_status`` is
    ``worst_status()`` over statuses the three modules already produced.

THE THREE CHANNELS STAY SEPARATE
    Feature drift and prediction drift answer different questions and are never
    merged into one figure:

        feature drift    = has the INPUT population changed?    MODEL-INDEPENDENT
        prediction drift = has the MODEL's OUTPUT changed?       MODEL-SPECIFIC

    Two models scoring one identical feature matrix therefore report identical
    feature drift and different prediction drift. That is correct behaviour, not
    a defect -- feature drift is a property of the data, so making it disagree
    between models would mean measuring something other than the data.

    Each channel keeps its own result dict and its own status under
    ``channel_status``, so a reader can always see which question produced which
    verdict. ``monitoring_status`` is an additional summary, never a
    replacement.

MODEL IDENTITY
    ``monitor_run()`` does not mint identity and does not define an identity
    format. It takes an ``AssuranceRunContext``-shaped dict -- built by the
    existing ``app.api.orchestration.build_assurance_run_context()`` -- and
    threads it through unchanged onto the result. There is exactly one identity
    mechanism in this project and this reuses it.

    This matters most for prediction drift, which is the one channel whose value
    depends on WHICH model produced the outputs. A prediction-drift result
    attributed to the wrong model is a wrong statement about that model's
    behaviour, not a cosmetic mislabel.

API-LAYER INDEPENDENCE
    Imports nothing from ``app.api``, matching ``app.drift.comparability`` and
    ``app.drift.prediction_drift``: the context is handled as a plain dict, so
    the monitoring layer acquires no dependency on the API layer. A caller
    holding a pydantic ``AssuranceRunContext`` passes ``context.model_dump()``.

Note on is_mock:
    ``is_mock=False`` means the arithmetic is real. It says nothing about where
    the window data came from. A window built over a dataset from
    ``app.drift.scenario.build_drift_scenario`` is SYNTHETIC and must never be
    presented as observed drift in a real lending population.
"""

from typing import Any, Dict, List, Mapping, Optional

from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PENDING,
    STATUS_WARNING,
    worst_status,
)
from app.drift.drift import drift_report
from app.drift.prediction_drift import prediction_drift_report
from app.fairness.fairness import DEFAULT_FAVORABLE_LABEL, fairness_report
from app.monitoring.windows import MonitoringWindow

__all__ = [
    "CHANNEL_FAIRNESS",
    "CHANNEL_FEATURE_DRIFT",
    "CHANNEL_PREDICTION_DRIFT",
    "MONITORING_CHANNELS",
    "REQUIRED_CONTEXT_FIELDS",
    "monitor_run",
]

CHANNEL_FEATURE_DRIFT = "feature_drift"
CHANNEL_PREDICTION_DRIFT = "prediction_drift"
CHANNEL_FAIRNESS = "fairness"

# Fixed channel order, so a result's channel_status keys and its alert ordering
# are stable across runs rather than dependent on dict insertion accidents.
MONITORING_CHANNELS = (
    CHANNEL_FEATURE_DRIFT,
    CHANNEL_PREDICTION_DRIFT,
    CHANNEL_FAIRNESS,
)

# The identity fields AssuranceRunContext requires. Declared here rather than
# imported so this module stays free of app.api (see the module docstring);
# tests assert this tuple against the real schema so the two cannot drift apart.
REQUIRED_CONTEXT_FIELDS = ("model_id", "model_version", "assurance_run_id")

# Statuses that raise an alert. Not a threshold and not a new status: this is a
# projection of statuses the analytical modules already produced.
_ALERTING_STATUSES = (STATUS_WARNING, STATUS_FAIL)


def _validate_context(context: Any) -> Dict[str, Any]:
    """Return a copy of ``context``, refusing one that cannot identify a model.

    A monitoring result whose model identity is absent cannot be kept apart
    from another model's result once the two are pooled -- which is precisely
    the cross-model contamination the Phase 5 evidence-isolation tests exist to
    prevent. Refusing here is cheaper than discovering it in a report.
    """
    if not isinstance(context, Mapping):
        raise ValueError(
            "context must be an AssuranceRunContext-shaped mapping with "
            f"{list(REQUIRED_CONTEXT_FIELDS)}, got {type(context).__name__}. "
            "Build it with app.api.orchestration.build_assurance_run_context() "
            "and pass .model_dump()."
        )

    missing = [f for f in REQUIRED_CONTEXT_FIELDS if not context.get(f)]
    if missing:
        raise ValueError(
            f"context is missing required identity field(s): {missing}. "
            "A monitoring result must be attributable to the model and run that "
            "produced it, otherwise two models' results cannot be told apart."
        )

    return dict(context)


def _feature_drift_channel(
    reference: MonitoringWindow,
    current: MonitoringWindow,
) -> Optional[Dict[str, Any]]:
    """Feature/data drift, or None when a window carries no features.

    Delegates wholly to ``drift_report()``. Note that this channel is given the
    two windows' FEATURES only -- no predictions and no model identity reach it,
    which is what keeps it model-independent.
    """
    if not (reference.has_features and current.has_features):
        return None
    return drift_report(reference.features, current.features)


def _prediction_drift_channel(
    reference: MonitoringWindow,
    current: MonitoringWindow,
) -> Optional[Dict[str, Any]]:
    """Prediction/output drift, or None when a window carries no predictions.

    Scores are passed only when BOTH windows have them: a PSI between a scored
    window and an unscored one is not a measurement of anything. When either
    side lacks scores the label channel is still evaluated and
    ``prediction_drift_report()`` marks the score channel unavailable.
    """
    if not (reference.has_predictions and current.has_predictions):
        return None

    both_scored = reference.has_scores and current.has_scores
    return prediction_drift_report(
        reference.predictions,
        current.predictions,
        reference_scores=reference.scores if both_scored else None,
        current_scores=current.scores if both_scored else None,
    )


def _fairness_channel(
    current: MonitoringWindow,
    protected_attribute: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Fairness over the CURRENT window, or None when it cannot be evaluated.

    Evaluated on the current window because fairness asks how the model is
    behaving in the period being monitored; the reference window's fairness is a
    separate question a caller can ask with a second ``monitor_run``.

    ``protected_attribute`` is REQUIRED and caller-supplied. It is deliberately
    not inferred: which attribute is protected is a regulatory and governance
    decision, and a monitoring module that guessed one would be inventing a
    compliance judgement. When it is None, fairness is simply not evaluated --
    reported as PENDING, never as PASS.

    ``favorable_label`` comes from the window's own model contract
    (``label_semantics``), falling back to the fairness module's documented
    default only when the contract stated none.
    """
    if protected_attribute is None or not current.has_predictions:
        return None
    if not current.has_features or protected_attribute not in current.features.columns:
        return None

    favorable_label = (
        current.favorable_label
        if current.favorable_label is not None
        else DEFAULT_FAVORABLE_LABEL
    )

    # reset_index(drop=True): fairness pairs predictions and the sensitive
    # feature by POSITION and rejects a non-positional pandas index rather than
    # re-pairing by index. The window's frame may carry the original row labels.
    sensitive = current.features[protected_attribute].reset_index(drop=True)

    return fairness_report(
        predictions=current.predictions,
        sensitive_feature=sensitive,
        favorable_label=favorable_label,
    )


def _window_descriptor(window: MonitoringWindow) -> Dict[str, Any]:
    """The window's own metadata, copied onto the result.

    Additive: ``reference_window_id`` / ``current_window_id`` stay exactly where
    they were, so no existing consumer changes. This carries the rest of what
    the window declared about ITSELF -- where its data came from and which
    period it describes -- so a downstream reader is not forced back to the
    window object to answer "was this observed or generated?".

    Timestamps are emitted as ISO-8601 strings rather than ``datetime``
    objects: these records travel into evidence and JSON responses, and a
    ``datetime`` is not JSON-serializable. ``None`` stays ``None`` -- an
    unstated bound is not converted into a value.

    ``provenance`` is likewise passed through untouched. ``None`` means the
    caller did not state where the data came from and is NEVER upgraded to
    ``"observed"``.
    """
    return {
        "window_id": window.window_id,
        "provenance": window.provenance,
        "window_start": (
            window.window_start.isoformat() if window.window_start else None
        ),
        "window_end": window.window_end.isoformat() if window.window_end else None,
        "record_count": len(window.predictions) if window.predictions else None,
    }


def _channel_status(result: Optional[Mapping[str, Any]]) -> str:
    """The status a channel reported, or PENDING when it did not run.

    Read from the channel's own ``status`` key -- never re-derived from its
    metric values, so a channel's status here is always the status its owning
    module assigned.
    """
    if result is None:
        return STATUS_PENDING
    return result.get("status", STATUS_PENDING)


def _build_alerts(
    channel_status: Mapping[str, str],
    results: Mapping[str, Optional[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """One alert per channel currently at WARNING or FAIL.

    Detection only -- no delivery, no routing, no suppression state. An alert
    carries no severity of its own: it echoes the channel's existing status, so
    an alert can never disagree with the result it was raised from. Channels at
    PASS or PENDING raise nothing; in particular a PENDING channel raises no
    alert, because an unmeasured channel is not a finding.
    """
    alerts: List[Dict[str, Any]] = []
    for channel in MONITORING_CHANNELS:
        status = channel_status[channel]
        if status not in _ALERTING_STATUSES:
            continue
        alerts.append(
            {
                "channel": channel,
                "status": status,
                "detail": _alert_detail(channel, results[channel]),
            }
        )
    return alerts


def _alert_detail(channel: str, result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The metric values behind an alert, copied from the channel's own result.

    Each channel reports different metrics, so the detail names the metrics that
    channel actually produced instead of forcing them into one shared shape.
    Values are copied by reference from the result; none is recomputed.
    """
    if result is None:
        return {}
    if channel == CHANNEL_FEATURE_DRIFT:
        return {
            "psi": result.get("psi"),
            "ks_statistic": result.get("ks_statistic"),
            "features_evaluated": result.get("features_evaluated"),
        }
    if channel == CHANNEL_PREDICTION_DRIFT:
        return {
            "label_psi": result.get("label_psi"),
            "label_status": result.get("label_status"),
            "score_psi": result.get("score_psi"),
            "score_status": result.get("score_status"),
            "score_availability": result.get("score_availability"),
        }
    return {
        "protected_attribute": result.get("protected_attribute"),
        "disparate_impact_ratio": result.get("disparate_impact_ratio"),
        "demographic_parity_diff": result.get("demographic_parity_diff"),
    }


def monitor_run(
    reference: MonitoringWindow,
    current: MonitoringWindow,
    *,
    context: Mapping[str, Any],
    protected_attribute: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate all monitoring channels over one reference/current window pair.

    Args:
        reference: The baseline window.
        current: The monitored window.
        context: ``AssuranceRunContext``-shaped mapping identifying the model
            and run these results belong to. Must carry non-empty
            ``model_id``, ``model_version`` and ``assurance_run_id``; built by
            ``app.api.orchestration.build_assurance_run_context()``. Threaded
            onto the result unchanged.
        protected_attribute: Column name in ``current.features`` to evaluate
            fairness over. Required for the fairness channel and never
            inferred -- see ``_fairness_channel``. When omitted, fairness is
            reported as PENDING rather than skipped silently.

    Returns:
        Dictionary with keys:
            context (dict): the identity mapping passed in, copied unchanged.
            reference_window_id / current_window_id (str): the two windows'
                labels, so a result names the periods it compared.
            windows (dict): additive. ``{"reference": {...}, "current": {...}}``,
                each carrying that window's own ``window_id``, ``provenance``,
                ``window_start``/``window_end`` (ISO-8601 strings or None) and
                ``record_count``. Copied from the window, never inferred --
                a ``provenance`` of None stays None.
            feature_drift (dict or None): ``drift_report()`` output, or None
                when either window carries no features.
            prediction_drift (dict or None): ``prediction_drift_report()``
                output, or None when either window carries no predictions.
            fairness (dict or None): ``fairness_report()`` output over the
                current window, or None when no protected attribute was given
                or the column is absent.
            channel_status (dict): one status per channel in
                ``MONITORING_CHANNELS``, always all three keys, PENDING for a
                channel that did not run. Read from each channel's own
                ``status``, never re-derived.
            monitoring_status (str): ``worst_status()`` over the three channel
                statuses. PENDING inputs are skipped, so an unmeasured channel
                neither hides nor invents a finding; PENDING overall means
                nothing at all could be measured.
            alerts (list of dict): one entry per channel at WARNING or FAIL,
                in ``MONITORING_CHANNELS`` order.
            is_mock (bool): False -- the calculations are real.

    Raises:
        ValueError: if either window is not a ``MonitoringWindow``, or if
            ``context`` is missing required identity fields.
    """
    for name, window in (("reference", reference), ("current", current)):
        if not isinstance(window, MonitoringWindow):
            raise ValueError(
                f"{name} must be a MonitoringWindow, got "
                f"{type(window).__name__}. Build one with "
                "app.monitoring.windows.window_from_model_output()."
            )

    validated_context = _validate_context(context)

    results: Dict[str, Optional[Dict[str, Any]]] = {
        CHANNEL_FEATURE_DRIFT: _feature_drift_channel(reference, current),
        CHANNEL_PREDICTION_DRIFT: _prediction_drift_channel(reference, current),
        CHANNEL_FAIRNESS: _fairness_channel(current, protected_attribute),
    }

    channel_status = {
        channel: _channel_status(results[channel]) for channel in MONITORING_CHANNELS
    }

    return {
        "context": validated_context,
        "reference_window_id": reference.window_id,
        "current_window_id": current.window_id,
        # Additive: each window's own declared metadata (provenance, period,
        # size). See _window_descriptor -- nothing here is inferred.
        "windows": {
            "reference": _window_descriptor(reference),
            "current": _window_descriptor(current),
        },
        CHANNEL_FEATURE_DRIFT: results[CHANNEL_FEATURE_DRIFT],
        CHANNEL_PREDICTION_DRIFT: results[CHANNEL_PREDICTION_DRIFT],
        CHANNEL_FAIRNESS: results[CHANNEL_FAIRNESS],
        "channel_status": channel_status,
        "monitoring_status": worst_status(
            *(channel_status[channel] for channel in MONITORING_CHANNELS)
        ),
        "alerts": _build_alerts(channel_status, results),
        "is_mock": False,
    }
