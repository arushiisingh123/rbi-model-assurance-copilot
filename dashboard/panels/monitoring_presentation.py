"""Pure presentation logic for the Monitoring panel (owner: Arushi).

WHAT THIS MODULE DOES
---------------------
Selects and reshapes values that ALREADY EXIST in a ``MonitoringAssuranceResult``
(``GET``/``POST /monitoring``) into table- and caption-ready structures.

WHAT IT DOES NOT DO
-------------------
It calculates nothing. No PSI, KS statistic, selection rate, disparate impact
ratio, class frequency, or status is computed, re-derived, re-rounded, or
re-classified here. Every number and every status it returns is the one the
monitoring API supplied.

In particular it never compares a value against a threshold. The four statuses
(PASS / WARNING / FAIL / PENDING) arrive already assigned by
``app/config/thresholds.py`` via the analytical modules, and a panel that
re-derived one could disagree with the result it is displaying.

TWO THINGS THIS MODULE IS CAREFUL ABOUT
---------------------------------------
``provenance`` of ``None`` means "the caller did not state where this data came
from". It is rendered as *not stated* and NEVER as "observed" -- collapsing the
two is exactly how generated scenario data gets read as real observed drift.

An unavailable channel is not a passing channel. A ``None`` result or a
``PENDING`` status is surfaced as an explicit limitation rather than omitted,
so a reader cannot mistake "not measured" for "measured and fine".

NO STREAMLIT HERE
-----------------
Deliberately free of any UI import so it can be tested headlessly. Rendering
lives in ``monitoring_panel.py``.
"""

from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "CHANNEL_LABELS",
    "PER_FEATURE_FIELDS",
    "PER_CLASS_FIELDS",
    "PROVENANCE_NOT_STATED",
    "channel_status_rows",
    "feature_drift_rows",
    "identity_rows",
    "limitations",
    "monitoring_summary_row",
    "per_class_rows",
    "prediction_drift_rows",
    "window_rows",
]

# Display names for the three analytical channels, in the order monitoring
# reports them. Keys are the API's own channel keys.
CHANNEL_LABELS = {
    "feature_drift": "Feature drift",
    "prediction_drift": "Prediction drift",
    "fairness": "Fairness",
}

PER_FEATURE_FIELDS = ("feature", "psi", "ks_statistic")
PER_CLASS_FIELDS = ("class", "reference_rate", "current_rate")

PROVENANCE_NOT_STATED = "not stated"


def _result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The inner monitoring result, or an empty mapping."""
    result = payload.get("result") if isinstance(payload, Mapping) else None
    return result if isinstance(result, Mapping) else {}


def identity_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Model and run identity, exactly as the API reported it.

    Shown because a monitoring finding that cannot name the model it describes
    cannot be told apart from another model's once the two are pooled.
    """
    context = _result(payload).get("context")
    context = context if isinstance(context, Mapping) else {}
    rows = [
        {"field": "model_id", "value": context.get("model_id")},
        {"field": "model_version", "value": context.get("model_version")},
        {"field": "assurance_run_id", "value": context.get("assurance_run_id")},
    ]
    adapter_id = context.get("adapter_id")
    if adapter_id is not None:
        rows.append({"field": "adapter_id", "value": adapter_id})
    return rows


def window_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per window, carrying its declared metadata.

    ``provenance`` of None renders as ``PROVENANCE_NOT_STATED``; it is never
    shown as "observed". Absent period bounds render as an em dash rather than
    an invented date.
    """
    windows = _result(payload).get("windows")
    windows = windows if isinstance(windows, Mapping) else {}

    rows: List[Dict[str, Any]] = []
    for side in ("reference", "current"):
        info = windows.get(side)
        info = info if isinstance(info, Mapping) else {}
        provenance = info.get("provenance")
        rows.append(
            {
                "window": side,
                "window_id": info.get("window_id"),
                "provenance": provenance if provenance else PROVENANCE_NOT_STATED,
                "window_start": info.get("window_start") or "—",
                "window_end": info.get("window_end") or "—",
                "record_count": info.get("record_count"),
            }
        )
    return rows


def channel_status_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per analytical channel with the status the API assigned it.

    Always all three channels, in ``CHANNEL_LABELS`` order, so a channel that
    did not run stays visible as PENDING instead of vanishing from the table.
    """
    result = _result(payload)
    channel_status = result.get("channel_status")
    channel_status = channel_status if isinstance(channel_status, Mapping) else {}

    return [
        {
            "channel": label,
            "status": channel_status.get(key, "PENDING"),
            "measured": result.get(key) is not None,
        }
        for key, label in CHANNEL_LABELS.items()
    ]


def monitoring_summary_row(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The headline: overall status, the windows compared, and alert count."""
    result = _result(payload)
    alerts = result.get("alerts")
    return {
        "monitoring_status": result.get("monitoring_status", "PENDING"),
        "reference_window_id": result.get("reference_window_id"),
        "current_window_id": result.get("current_window_id"),
        "alert_count": len(alerts) if isinstance(alerts, list) else 0,
    }


def feature_drift_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Per-feature PSI/KS as reported. Empty when the channel did not run.

    No per-feature status is produced: the API deliberately attaches none, so
    neither does this.
    """
    feature_drift = _result(payload).get("feature_drift")
    if not isinstance(feature_drift, Mapping):
        return []
    per_feature = feature_drift.get("per_feature")
    if not isinstance(per_feature, list):
        return []
    return [
        {field: entry.get(field) for field in PER_FEATURE_FIELDS}
        for entry in per_feature
        if isinstance(entry, Mapping)
    ]


def prediction_drift_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The two prediction-drift channels as separate, labelled rows.

    Label and score are reported separately because they are different
    measurements -- a categorical PSI over class frequencies and a quantile PSI
    over a continuous score. Merging them into one number would average two
    incomparable quantities.

    The score row is included even when unavailable, carrying its
    ``score_availability`` reason, so a missing measurement is never read as a
    measured zero.
    """
    prediction_drift = _result(payload).get("prediction_drift")
    if not isinstance(prediction_drift, Mapping):
        return []

    return [
        {
            "channel": "Label (categorical PSI)",
            "psi": prediction_drift.get("label_psi"),
            "ks_statistic": None,
            "status": prediction_drift.get("label_status"),
            "availability": "computed",
        },
        {
            "channel": "Score (quantile PSI)",
            "psi": prediction_drift.get("score_psi"),
            "ks_statistic": prediction_drift.get("score_ks_statistic"),
            "status": prediction_drift.get("score_status"),
            "availability": prediction_drift.get("score_availability"),
        },
    ]


def per_class_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Reference/current frequency per predicted class, as reported."""
    prediction_drift = _result(payload).get("prediction_drift")
    if not isinstance(prediction_drift, Mapping):
        return []
    per_class = prediction_drift.get("per_class")
    if not isinstance(per_class, list):
        return []
    return [
        {field: entry.get(field) for field in PER_CLASS_FIELDS}
        for entry in per_class
        if isinstance(entry, Mapping)
    ]


def limitations(payload: Mapping[str, Any]) -> List[str]:
    """Plain-language notes about what this run could NOT measure.

    Surfaced rather than hidden: an unmeasured channel is not a passing one,
    and a reader who cannot see the gap will read the overall status as
    covering everything.
    """
    result = _result(payload)
    notes: List[str] = []

    if result.get("feature_drift") is None:
        notes.append(
            "Feature drift was not evaluated: at least one window carried no "
            "feature data."
        )

    prediction_drift = result.get("prediction_drift")
    if prediction_drift is None:
        notes.append(
            "Prediction drift was not evaluated: at least one window carried "
            "no predictions."
        )
    elif isinstance(prediction_drift, Mapping):
        if prediction_drift.get("score_availability") == "unavailable_no_scores":
            notes.append(
                "The score channel was not measured: this model provides no "
                "probability output. The label channel was measured and the "
                "prediction-drift status reflects it alone."
            )

    if result.get("fairness") is None:
        attribute: Optional[str] = payload.get("protected_attribute")
        if attribute:
            notes.append(
                f"Fairness was not evaluated: '{attribute}' is not a column in "
                "this model's feature space."
            )
        else:
            notes.append(
                "Fairness was not evaluated: this model declares no protected "
                "attribute and none was supplied. It is never inferred."
            )

    for side, info in zip(("reference", "current"), window_rows(payload)):
        if info["provenance"] == PROVENANCE_NOT_STATED:
            notes.append(
                f"The {side} window's provenance was not stated, so this run "
                "cannot be presented as observed production monitoring."
            )

    return notes
