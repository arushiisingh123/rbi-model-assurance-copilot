"""Fairness and drift presentation panel (Owner: Arushi — Phase 4).

Renders the ``FairnessDriftResult`` payload returned by ``GET /fairness-drift``
(``dashboard.api_client.get_fairness_drift()``). The caller fetches the data and
passes it in, matching the convention the other Phase 4 panels follow.

WHAT THIS MODULE IS NOT
    - Not the fairness module (`app/fairness/`) and not the drift module
      (`app/drift/`). It never computes a selection rate, a demographic parity
      difference, a disparate impact ratio, a PSI, or a KS statistic. Every
      number shown here arrives in ``data`` already calculated.
    - Not a classifier. It never compares a value against a threshold and never
      derives a PASS/WARNING/FAIL/PENDING verdict. Statuses are displayed
      exactly as the API supplied them, through the shared
      ``dashboard.api_client.render_status`` helper.
    - Not a source of per-group or per-feature severity. The API deliberately
      attaches no status to an individual group or feature, so neither does
      this panel.

TWO THINGS THIS PANEL IS CAREFUL ABOUT
    ``disparate_impact_ratio`` is displayed, never recomputed from the group
    selection rates. Dividing two 4-decimal rounded rates can disagree with the
    reported ratio in the last digit (0.3854 / 0.4545 rounds to 0.848 while the
    full-precision ratio is 0.8479), so the reported field is the only correct
    thing to show.

    ``personal_status_and_sex`` (Attribute 9 of the German Credit data) combines
    personal/marital status with sex. Group labels are shown as the raw observed
    category codes and are never relabelled as a standalone gender or sex.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import streamlit as st

from dashboard.api_client import render_status

# The exact fields the API contract supplies per group and per feature. Used for
# the table column order; a payload carrying anything else is shown as-is by the
# raw-values expander rather than being reshaped here.
GROUP_FIELDS = ("group", "count", "favorable_count", "selection_rate")
PER_FEATURE_FIELDS = ("feature", "psi", "ks_statistic")

# Matched case-insensitively against a drift note. A note that says "Synthetic"
# or "synthetic" must be labelled just as clearly as one shouting "SYNTHETIC":
# the whole point of the check is that synthetic drift can never read as
# observed drift, so it must not depend on how the producer capitalised it.
_SYNTHETIC_MARKER = "synthetic"


def render_fairness_drift_panel(data: dict[str, Any], source: str) -> None:
    """Render the fairness and drift panel.

    Parameters
    ----------
    data
        The ``FairnessDriftResult``-shaped dict already fetched by the caller
        (for example via ``dashboard.api_client.get_fairness_drift()``).
    source
        ``"api"`` or ``"fallback"`` -- where ``data`` came from, as already
        determined by the caller.
    """
    fairness = data.get("fairness") or {}
    drift = data.get("drift") or {}

    _render_fairness(fairness, source)
    st.divider()
    _render_drift(drift, source, payload_note=data.get("note"))


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------


def _render_fairness(fairness: Mapping[str, Any], source: str) -> None:
    """Aggregate fairness metrics, then the per-group breakdown behind them."""
    st.subheader("Fairness Evaluation")
    st.caption(
        f"Data source: **{source}** | "
        f"Mock data: **{fairness.get('is_mock', True)}**"
    )

    status = fairness.get("status")
    st.markdown(f"**Fairness Status:** {render_status(status)}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Protected Attribute", _shown(fairness.get("protected_attribute")))
    col2.metric(
        "Demographic Parity Diff", _shown(fairness.get("demographic_parity_diff"))
    )
    col3.metric(
        "Disparate Impact Ratio", _shown(fairness.get("disparate_impact_ratio"))
    )
    st.caption(
        "Both metrics are reported by the fairness module. The disparate impact "
        "ratio is shown as supplied and is not re-derived from the group rates "
        "below. The thresholds behind the status are project conventions, not "
        "RBI requirements."
    )

    if status == "PENDING":
        st.info(
            "Status is PENDING: the fairness module reported that the "
            "comparison between groups could not be made -- fewer than two "
            "groups were observed, or no group received the favourable "
            "outcome. Any observed groups are still shown below. This is "
            "neither a pass nor a failure."
        )

    _render_fairness_groups(fairness)


def _render_fairness_groups(fairness: Mapping[str, Any]) -> None:
    """Per-group selection rates: chart first, exact values alongside."""
    st.markdown("**Selection rate by observed group**")

    groups = fairness.get("groups") or []
    if not groups:
        st.info(
            "No per-group fairness breakdown was returned for this payload."
        )
        return

    favorable_label = fairness.get("favorable_label")
    label_note = (
        f" Favourable outcome label: `{favorable_label}`."
        if favorable_label is not None
        else ""
    )
    st.caption(
        "Selection rate is the share of each group receiving the favourable "
        "outcome, as calculated by the fairness module." + label_note
        + " Group labels are the raw observed categories; for "
        "`personal_status_and_sex` they combine personal/marital status with "
        "sex and are not a standalone gender field."
    )

    rows = _rows(groups, GROUP_FIELDS)
    st.bar_chart(
        rows,
        x="group",
        y="selection_rate",
        x_label="Observed group",
        y_label="Selection rate (as reported)",
        sort=False,  # preserve the order the API supplied
        use_container_width=True,
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def _render_drift(
    drift: Mapping[str, Any], source: str, payload_note: Any = None
) -> None:
    """Aggregate PSI/KS, provenance caveats, then the per-feature breakdown."""
    st.subheader("Drift Detection")
    is_mock = drift.get("is_mock", True)
    st.caption(f"Data source: **{source}** | Mock data: **{is_mock}**")

    _render_drift_provenance(drift, source, payload_note, is_mock)

    status = drift.get("status")
    st.markdown(f"**Drift Status:** {render_status(status)}")

    col1, col2 = st.columns(2)
    col1.metric("Population Stability Index (PSI)", _shown(drift.get("psi")))
    col2.metric("Kolmogorov-Smirnov (KS)", _shown(drift.get("ks_statistic")))
    st.caption(
        "Both aggregates are the maximum across evaluated features, computed by "
        "the drift module and shown as supplied. They are evaluated "
        "independently, so the feature behind the maximum PSI need not be the "
        "feature behind the maximum KS. Status follows the reported PSI only; "
        "no KS threshold exists."
    )

    if status == "PENDING":
        st.info(
            "Status is PENDING: the drift module reported that no feature "
            "could be evaluated. Absent or unusable data is not evidence of "
            "drift, so this is neither a pass nor a failure."
        )

    features_evaluated = drift.get("features_evaluated") or []
    st.markdown("**Features evaluated**")
    if features_evaluated:
        st.caption(
            f"{len(features_evaluated)} feature(s) were actually evaluated: "
            + ", ".join(f"`{name}`" for name in features_evaluated)
            + ". Features not listed here are not covered by the values above."
        )
    else:
        st.caption("No features were evaluated for this payload.")

    _render_per_feature(drift)


def _render_drift_provenance(
    drift: Mapping[str, Any], source: str, payload_note: Any, is_mock: bool
) -> None:
    """State what the drift comparison actually is, before showing any number.

    A synthetic scenario and a development train/test comparison are both
    legitimate, and neither is observed production drift. Whichever applies
    must be visible next to the numbers rather than inferred by the reader.

    ``is_mock`` is passed in rather than re-read from ``drift`` so the warning
    and the caption above it can never disagree: a payload with no ``is_mock``
    key must not be captioned as mock while the matching warning is withheld.
    """
    notes = [note for note in (drift.get("note"), payload_note) if note]

    for note in notes:
        if _SYNTHETIC_MARKER in str(note).lower():
            st.warning(
                "SYNTHETIC DRIFT SCENARIO — not observed production drift. "
                f"{note}"
            )
        else:
            st.info(str(note))

    if source == "fallback" or is_mock:
        st.warning(
            "This drift result is mock or fallback data, not a live "
            "measurement. It must not be presented as observed drift in a "
            "real lending population."
        )

    if not notes:
        st.caption(
            "Production drift compares the development training split with the "
            "held-out test split of one static dataset. Nothing here observes a "
            "live lending population."
        )


def _render_per_feature(drift: Mapping[str, Any]) -> None:
    """Per-feature PSI and KS, charted separately because they are independent."""
    st.markdown("**Per-feature drift detail**")

    per_feature = drift.get("per_feature") or []
    if not per_feature:
        st.info("No per-feature drift breakdown was returned for this payload.")
        return

    st.caption(
        "Per-feature values as calculated by the drift module. They carry no "
        "status and no threshold of their own -- only the aggregate PSI above "
        "is classified. PSI and KS are charted separately because their maxima "
        "are evaluated independently."
    )

    rows = _rows(per_feature, PER_FEATURE_FIELDS)

    st.markdown("Per-feature PSI")
    st.bar_chart(
        rows,
        x="feature",
        y="psi",
        x_label="Feature",
        y_label="PSI (as reported)",
        sort=False,  # preserve the order the API supplied
        use_container_width=True,
    )

    st.markdown("Per-feature KS statistic")
    st.bar_chart(
        rows,
        x="feature",
        y="ks_statistic",
        x_label="Feature",
        y_label="KS statistic (as reported)",
        sort=False,
        use_container_width=True,
    )

    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------


def _rows(
    records: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    """Order each record's known fields for display, copying values verbatim.

    Only reorders keys and fills a genuinely absent one with ``None``; no value
    is converted, rounded, or derived. A record carrying extra keys keeps them,
    appended after the known fields, so nothing supplied is silently dropped.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {field: record.get(field) for field in fields}
        row.update(
            {key: value for key, value in record.items() if key not in fields}
        )
        rows.append(row)
    return rows


def _shown(value: Any) -> str:
    """Render a metric value as supplied, or say plainly that it is absent."""
    return "not reported" if value is None else str(value)
