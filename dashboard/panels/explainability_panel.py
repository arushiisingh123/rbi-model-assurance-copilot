"""Explainability dashboard panel (owner: Manas, Phase 4).

Presents results the explainability module already computed. It recalculates
nothing: every number rendered here comes from
``app.explainability.explain()`` via the API, unchanged.

All sorting, selection, top-N and scale-label logic lives in
``explainability_presentation.py`` so it can be tested without Streamlit or a
running API. This file is the rendering shell only.

CONTAINER VS CONTENT
--------------------
This module owns the *content* of the Explainability panel. The dashboard
shell, tab placement, navigation, and data fetching belong to Khushi
(``dashboard/dashboard_app.py``, ``dashboard/api_client.py``). The panel
therefore takes a fetch callable rather than importing the API client, so the
two can be wired together without either owner editing the other's file.
"""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import streamlit as st

from dashboard.panels.explainability_presentation import (
    DEFAULT_TOP_N,
    SCALE_CAPTION,
    full_contribution_rows,
    global_importance_rows,
    instance_contribution_rows,
    instance_options,
    scale_caption,
    scale_label,
)

SUPPORTED_METHODS = ("shap", "lime")

# Guard rail for the collapsed full-matrix expander. 200 records x 20 features
# is 4,000 rows; Streamlit will render it, but building the list on every
# rerun is wasteful, so the expander only materialises when opened.
FULL_MATRIX_WARN_ROWS = 2000


def render(
    fetch_explainability: Callable[[str], Tuple[Mapping[str, Any], str]],
    *,
    instance_ids: Optional[Sequence[Any]] = None,
    key_prefix: str = "explainability",
) -> None:
    """Render the Explainability panel.

    Args:
        fetch_explainability: callable taking a method name ("shap"/"lime")
            and returning ``(explanation, source)`` — the same shape
            ``dashboard.api_client.get_explainability`` already returns.
            Injected rather than imported so this panel does not depend on
            Khushi's client module.
        instance_ids: optional identifiers for the explained rows, in the
            same order. When supplied they label the record selector; when
            omitted the selector is labelled by position and makes no claim
            of applicant identity. The model layer produces these
            (``predict_batch()['instance_ids']``), but the current
            ``/explainability`` response does not carry them.
        key_prefix: Streamlit widget key prefix, so the panel can appear more
            than once without key collisions.
    """
    method = st.radio(
        "Explainability method",
        list(SUPPORTED_METHODS),
        horizontal=True,
        index=0,
        key=f"{key_prefix}_method",
    )

    explanation, source = fetch_explainability(method)

    is_mock = explanation.get("is_mock", False) if isinstance(explanation, Mapping) else False
    st.caption(f"Data source: **{source}** | Mock data: **{is_mock}**")
    if is_mock:
        st.warning(
            "This explanation is MOCK data, not a real model explanation. "
            "It must not be read as evidence about the credit model."
        )

    try:
        unit = scale_label(method)
        axis_label = scale_caption(method)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.info(f"**{method.upper()} contribution scale: {unit}.** {SCALE_CAPTION}")

    try:
        _render_global(explanation, method, axis_label)
        st.divider()
        _render_instance(explanation, method, axis_label, instance_ids, key_prefix)
        st.divider()
        _render_full_data(explanation, instance_ids)
    except ValueError as exc:
        # A malformed explanation is reported, never patched over with
        # substitute numbers.
        st.error(f"Could not present this explanation: {exc}")


def _render_global(explanation: Mapping[str, Any], method: str, axis_label: str) -> None:
    """Model-level view: which features drive the model overall."""
    st.subheader(f"Global feature importance — {method.upper()}")
    st.caption(
        "MODEL-LEVEL. Averaged across every explained record — this describes "
        "the model, not any individual applicant."
    )

    rows = global_importance_rows(explanation)
    if not rows:
        st.info("No global importance values were returned.")
        return

    st.bar_chart(
        rows,
        x="feature",
        y="importance",
        x_label="Feature",
        y_label=f"Mean absolute contribution ({scale_label(method)})",
        horizontal=True,
        sort=None,  # keep the helper's descending order
        use_container_width=True,
    )
    with st.expander("Global importance values", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_instance(
    explanation: Mapping[str, Any],
    method: str,
    axis_label: str,
    instance_ids: Optional[Sequence[Any]],
    key_prefix: str,
) -> None:
    """Record-level view: why one prediction came out as it did."""
    st.subheader(f"Single-record explanation — {method.upper()}")

    options = instance_options(explanation, instance_ids)
    if not options:
        st.info("No explained records were returned.")
        return

    has_identity = options[0]["is_identity"]
    if has_identity:
        st.caption(
            "RECORD-LEVEL. Signed contributions for the selected record. "
            "Records are identified by their stable instance_id."
        )
    else:
        st.caption(
            "RECORD-LEVEL. Signed contributions for the selected record. "
            "The current /explainability response carries no instance_id, so "
            "records are selected by POSITION in this explained set. A "
            "position is not an applicant identity and does not remain "
            "stable across runs."
        )

    labels = [option["label"] for option in options]
    selected_label = st.selectbox(
        "Record", labels, index=0, key=f"{key_prefix}_record"
    )
    position = labels.index(selected_label)

    top_n = st.slider(
        "Features shown",
        min_value=1,
        max_value=max(1, len(explanation["per_instance"][position]["contributions"])),
        value=min(DEFAULT_TOP_N, len(explanation["per_instance"][position]["contributions"])),
        key=f"{key_prefix}_top_n",
    )

    rows = instance_contribution_rows(explanation, position, top_n=top_n)
    if not rows:
        st.info("This record has no contributions to display.")
        return

    # Signed values go to the chart: direction is the point of a local
    # explanation. Ordering used abs(); the plotted number never does.
    st.bar_chart(
        [{"feature": r["feature"], "contribution": r["contribution"]} for r in rows],
        x="feature",
        y="contribution",
        x_label="Feature",
        y_label=axis_label,
        horizontal=True,
        sort=None,
        use_container_width=True,
    )
    st.caption(
        "Positive values push the prediction toward class 1 (BAD / higher "
        "credit risk); negative values push toward class 0 (GOOD). Ordered by "
        "absolute size; the values shown are the original signed numbers."
    )

    st.markdown(f"**Top {len(rows)} contributions**")
    st.dataframe(
        [
            {"Feature": r["feature"], f"Contribution ({scale_label(method)})": r["contribution"]}
            for r in rows
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_full_data(
    explanation: Mapping[str, Any], instance_ids: Optional[Sequence[Any]]
) -> None:
    """The complete matrix, collapsed by default."""
    per_instance = explanation.get("per_instance") or []
    features = len(per_instance[0].get("contributions", {})) if per_instance else 0
    total = len(per_instance) * features

    with st.expander(
        f"Full explanation data ({len(per_instance)} records x {features} features "
        f"= {total} contributions)",
        expanded=False,
    ):
        st.caption(
            "Complete per-record contributions, unmodified. Collapsed by "
            "default because this is an audit view, not a reading view."
        )
        if total > FULL_MATRIX_WARN_ROWS:
            st.caption(
                f"Large table ({total} rows) — it may take a moment to render."
            )
        st.dataframe(
            full_contribution_rows(explanation, instance_ids),
            use_container_width=True,
            hide_index=True,
        )
