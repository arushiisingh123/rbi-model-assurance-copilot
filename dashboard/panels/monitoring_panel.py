"""Monitoring presentation panel (owner: Arushi).

Renders the ``MonitoringAssuranceResult`` payload returned by
``GET``/``POST /monitoring``. The caller fetches the data and passes it in,
matching the convention the other panels follow.

WHAT THIS MODULE IS NOT
    - Not the monitoring module (``app/monitoring/``), the drift module
      (``app/drift/``), or the fairness module (``app/fairness/``). It never
      computes a PSI, a KS statistic, a selection rate, a disparate impact
      ratio, or a class frequency. Every number shown arrives already
      calculated.
    - Not a classifier. It never compares a value against a threshold and never
      derives a PASS/WARNING/FAIL/PENDING verdict. Statuses are displayed
      exactly as the API supplied them, through the shared
      ``dashboard.api_client.render_status`` helper.
    - Not a source of per-feature or per-class severity. The API attaches no
      status to an individual feature or class, so neither does this panel.

    All reshaping lives in ``monitoring_presentation.py``, which is
    streamlit-free and tested headlessly.

THREE THINGS THIS PANEL IS CAREFUL ABOUT
    A window whose ``provenance`` is unstated is labelled *not stated*, never
    "observed". The whole purpose of the field is that generated scenario data
    can never read as real observed drift.

    Label drift and score drift are shown as two separate rows. They are two
    different measurements -- a categorical PSI over class frequencies and a
    quantile PSI over a continuous score -- and one may be unavailable while
    the other is measured.

    Channels that could not be measured are surfaced as explicit limitations.
    An unmeasured channel is not a passing channel, and the overall status must
    not be read as covering it.
"""
from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from dashboard.api_client import render_status
from dashboard.panels.monitoring_presentation import (
    channel_status_rows,
    feature_drift_rows,
    identity_rows,
    limitations,
    monitoring_summary_row,
    per_class_rows,
    prediction_drift_rows,
    window_rows,
)


def render_monitoring_panel(data: Mapping[str, Any], source: str) -> None:
    """Render the monitoring panel.

    Parameters
    ----------
    data
        The ``MonitoringAssuranceResult``-shaped mapping already fetched by the
        caller.
    source
        Human-readable provenance of the payload itself (e.g. "API" or a
        fallback label), displayed so a reader always knows whether the
        dashboard reached the live service.
    """
    st.header("Monitoring")
    st.caption(f"Data source: {source}")

    if not isinstance(data, Mapping) or not data.get("result"):
        st.warning(
            "No monitoring result available. Monitoring runs on demand via "
            "`/monitoring`; it is not collected or scheduled."
        )
        return

    summary = monitoring_summary_row(data)

    st.subheader("Overall")
    col_status, col_windows, col_alerts = st.columns(3)
    col_status.metric("Monitoring status", render_status(summary["monitoring_status"]))
    col_windows.metric(
        "Windows compared",
        f"{summary['reference_window_id']} → {summary['current_window_id']}",
    )
    col_alerts.metric("Alerts raised", summary["alert_count"])
    st.caption(
        "Overall status is the most severe MEASURED channel. PENDING channels "
        "are skipped rather than ranked, so an unmeasured channel neither "
        "hides nor invents a finding."
    )

    st.subheader("Model identity")
    st.table(identity_rows(data))

    st.subheader("Windows")
    st.table(window_rows(data))
    st.caption(
        "`window_start`/`window_end` are monitoring-window boundaries — the "
        "period the records describe — not collection or scoring timestamps. "
        "A provenance of *not stated* means the caller did not declare where "
        "the data came from; it does not mean observed."
    )

    st.subheader("Channels")
    channel_rows = [
        {
            "channel": row["channel"],
            "status": render_status(row["status"]),
            "measured": "yes" if row["measured"] else "no",
        }
        for row in channel_status_rows(data)
    ]
    st.table(channel_rows)

    _render_prediction_drift(data)
    _render_feature_drift(data)
    _render_fairness(data)
    _render_alerts(data)
    _render_evidence(data)
    _render_limitations(data)


def _render_prediction_drift(data: Mapping[str, Any]) -> None:
    """Prediction drift: the model's OWN output, not its inputs."""
    rows = prediction_drift_rows(data)
    st.subheader("Prediction drift")
    if not rows:
        st.info("Prediction drift was not evaluated for this run.")
        return

    # psi/ks_statistic are passed through as numbers or None -- never replaced
    # with a placeholder string. Mixing str and float in one column produces an
    # object column that Arrow cannot serialize, and an absent measurement is
    # better shown as blank than as a symbol that could be read as a value.
    st.table(
        [
            {
                "channel": row["channel"],
                "psi": row["psi"],
                "ks_statistic": row["ks_statistic"],
                "status": render_status(row["status"]),
                "availability": row["availability"],
            }
            for row in rows
        ]
    )
    st.caption(
        "Prediction drift asks whether the MODEL'S OUTPUT changed, so it is "
        "model-specific — unlike feature drift, which asks whether the input "
        "population changed and is the same for any model over the same rows. "
        "No KS threshold exists; KS is reported as a metric only."
    )

    per_class = per_class_rows(data)
    if per_class:
        st.markdown("**Predicted class frequencies**")
        st.table(per_class)


def _render_feature_drift(data: Mapping[str, Any]) -> None:
    rows = feature_drift_rows(data)
    st.subheader("Feature drift")
    if not rows:
        st.info("Feature drift was not evaluated for this run.")
        return
    st.table(rows)
    st.caption(
        "Per-feature values carry no status of their own; the channel status "
        "comes from the aggregate PSI. A monitoring window of only a few dozen "
        "records can report drift that is not there — PSI bins into 10 "
        "reference quantiles."
    )


def _render_fairness(data: Mapping[str, Any]) -> None:
    result = data.get("result") or {}
    fairness = result.get("fairness")
    st.subheader("Fairness monitoring")

    if not isinstance(fairness, Mapping):
        attribute = data.get("protected_attribute")
        if attribute:
            st.info(
                f"Fairness was not evaluated: '{attribute}' is not a column in "
                "this model's feature space."
            )
        else:
            st.info(
                "Fairness was not evaluated: this model declares no protected "
                "attribute and none was supplied. It is never inferred — which "
                "attribute is protected is a governance decision."
            )
        return

    col_attr, col_di, col_status = st.columns(3)
    col_attr.metric("Protected attribute", str(fairness.get("protected_attribute")))
    col_di.metric("Disparate impact ratio", fairness.get("disparate_impact_ratio"))
    col_status.metric("Status", render_status(fairness.get("status")))

    groups = fairness.get("groups")
    if isinstance(groups, list) and groups:
        st.table(groups)
    st.caption(
        "Group labels are the raw observed categories, shown as-is. Fairness "
        "is evaluated on the CURRENT window."
    )


def _render_alerts(data: Mapping[str, Any]) -> None:
    result = data.get("result") or {}
    alerts = result.get("alerts")
    st.subheader("Alerts")
    if not isinstance(alerts, list) or not alerts:
        st.success("No channel is at WARNING or FAIL.")
        return
    for alert in alerts:
        if not isinstance(alert, Mapping):
            continue
        st.warning(
            f"{alert.get('channel')}: {render_status(alert.get('status'))}"
        )
        st.json(alert.get("detail") or {})
    st.caption(
        "Detection only — no delivery, routing, or suppression. An alert "
        "echoes its channel's existing status and can never disagree with it."
    )


def _render_evidence(data: Mapping[str, Any]) -> None:
    evidence = data.get("evidence")
    st.subheader("Evidence")
    if not isinstance(evidence, list) or not evidence:
        st.info("No monitoring evidence records were produced.")
        return

    st.table(
        [
            {
                "evidence_type": record.get("evidence_type"),
                "status": render_status(record.get("status")),
                "model_id": record.get("model_id"),
            }
            for record in evidence
            if isinstance(record, Mapping)
        ]
    )
    st.caption(
        "A channel that did not run produces no record — an absent measurement "
        "is not evidence. The `monitoring_summary` record still reports that "
        "channel's PENDING status, so the gap stays visible."
    )
    with st.expander("Raw evidence records"):
        st.json(evidence)


def _render_limitations(data: Mapping[str, Any]) -> None:
    notes = limitations(data)
    if not notes:
        return
    st.subheader("Limitations")
    for note in notes:
        st.warning(note)
