"""Model result presentation panel (Owner: Namitha — Phase 4).

Renders the ``ModelResult`` payload returned by ``GET /model``
(``dashboard.api_client.get_model()``). The caller fetches the data and passes
it in, matching the convention the other Phase 4 panels follow.

WHAT THIS MODULE IS NOT
    - Not the model module (`app/models/`). It never trains, predicts, scores,
      or evaluates. Every prediction, probability and performance metric shown
      here arrives in ``data`` already calculated by
      ``app.models.model.evaluate()`` / ``predict_batch()``.
    - Not a metric calculator. Accuracy, precision, recall, F1 and ROC-AUC are
      DISPLAYED exactly as the API supplied them. None is derived, rounded into
      a different value, or recomputed from ``predictions`` — doing so could
      disagree with the owning module in the last digit, and the reported field
      is the only correct thing to show.
    - Not a classifier of model quality. The panel attaches no PASS/WARNING/FAIL
      verdict to a metric; the API supplies no per-metric status, so neither
      does this panel.

ON THE TWO DISTRIBUTION CHARTS
    Counting how many predictions fall in each class, and how many probabilities
    fall in each 0.1-wide band, is DISPLAY AGGREGATION — the same kind of
    operation as sorting a table. It derives no new analytical metric: the
    inputs are the already-computed ``predictions`` and ``probabilities``, and
    the outputs are frequencies used only to draw a chart. No threshold is
    applied and no verdict is produced.

CLASS LABELS
    ``0 = GOOD`` / ``1 = BAD`` is read from ``model_metadata.label_semantics``
    when the API supplies it, rather than hardcoded here, so the panel cannot
    drift from the model module's own definition. If it is absent the raw class
    value is shown unlabelled instead of guessing.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import streamlit as st

# The metric fields the API contract supplies, in display order. A payload
# carrying anything else is shown by the raw-values expander rather than being
# reshaped here.
METRIC_FIELDS = ("accuracy", "precision", "recall", "f1", "roc_auc")

# Width of each probability display band. Ten bands over [0, 1] is readable at
# desktop width and fine enough to show skew without implying precision the
# underlying probabilities do not have.
PROBABILITY_BAND_WIDTH = 0.1

# Rows shown in the prediction table before it is capped. The full record set is
# available in the feature-matrix expander; 200 rows of scrolling is not a
# reading view.
MAX_TABLE_ROWS = 200


def render_model_panel(data: dict[str, Any], source: str) -> None:
    """Render the model panel.

    Parameters
    ----------
    data
        The ``ModelResult``-shaped dict already fetched by the caller (for
        example via ``dashboard.api_client.get_model()``).
    source
        ``"api"`` or ``"fallback"`` -- where ``data`` came from, as already
        determined by the caller.
    """
    data = data or {}
    st.caption(
        f"Data source: **{source}** | Mock data: **{data.get('is_mock', False)}**"
    )

    _render_metadata(data.get("model_metadata") or {})
    st.divider()
    _render_metrics(data.get("model_metrics"))
    st.divider()
    _render_prediction_distribution(data)
    st.divider()
    _render_probability_distribution(data)
    st.divider()
    _render_prediction_table(data)
    _render_feature_matrix(data)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _render_metadata(meta: Mapping[str, Any]) -> None:
    st.subheader("Model metadata")
    if not meta:
        st.info("No model metadata was returned.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Model type", meta.get("model_type", "N/A"))
    col2.metric("Version", meta.get("version", "N/A"))

    st.text(f"Trained on: {meta.get('trained_on', 'N/A')}")
    features = meta.get("feature_names") or []
    st.text(f"Features ({len(features)}): {', '.join(features) if features else 'N/A'}")

    label_semantics = meta.get("label_semantics")
    if label_semantics:
        st.info(
            f"**Target semantics** — 0 = {label_semantics.get('0', 'N/A')}; "
            f"1 = {label_semantics.get('1', 'N/A')}. "
            f"{label_semantics.get('probabilities_represent', '')}"
        )


# ---------------------------------------------------------------------------
# Performance metrics (P4-02)
# ---------------------------------------------------------------------------


def _render_metrics(metrics: Mapping[str, Any] | None) -> None:
    """Display held-out evaluation metrics exactly as the API reported them."""
    st.subheader("Model performance (held-out test set)")

    if not metrics:
        st.info(
            "No model metrics were returned by the API. Performance metrics "
            "come from `app.models.model.evaluate()` via `GET /model`; they are "
            "never computed in the dashboard."
        )
        return

    if metrics.get("is_mock"):
        st.warning(
            "These performance metrics are MOCK data, not a real evaluation of "
            "the trained model."
        )

    present = [field for field in METRIC_FIELDS if metrics.get(field) is not None]
    if present:
        columns = st.columns(len(present))
        for column, field in zip(columns, present):
            column.metric(field.replace("_", " ").upper(), _format_metric(metrics[field]))
    else:
        st.info("The metrics payload contained no recognised metric fields.")

    n_test = metrics.get("n_test_samples")
    if n_test is not None:
        st.caption(
            f"Evaluated on {n_test} held-out test record(s). Values are shown as "
            "reported by the model module and are not recalculated here."
        )

    with st.expander("Raw metric values", expanded=False):
        st.json(dict(metrics))


def _format_metric(value: Any) -> str:
    """Format for display only — the underlying value is never altered."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}"
    return str(value)


# ---------------------------------------------------------------------------
# Prediction distribution
# ---------------------------------------------------------------------------


def _class_label(value: Any, label_semantics: Mapping[str, Any] | None) -> str:
    """Label a class using the model's own semantics, never a hardcoded map."""
    if not label_semantics:
        return f"Class {value}"
    described = label_semantics.get(str(value))
    return f"{value} — {described}" if described else f"Class {value}"


def _render_prediction_distribution(data: Mapping[str, Any]) -> None:
    st.subheader("Prediction distribution")
    predictions = data.get("predictions") or []
    if not predictions:
        st.info("No predictions were returned.")
        return

    label_semantics = (data.get("model_metadata") or {}).get("label_semantics")

    counts: dict[Any, int] = {}
    for prediction in predictions:
        counts[prediction] = counts.get(prediction, 0) + 1

    rows = [
        {"class": _class_label(value, label_semantics), "count": counts[value]}
        for value in sorted(counts, key=lambda v: (isinstance(v, str), v))
    ]

    st.bar_chart(
        rows,
        x="class",
        y="count",
        x_label="Predicted class",
        y_label="Records",
        sort=False,
        use_container_width=True,
    )
    st.caption(
        f"How the {len(predictions)} predicted labels are distributed across "
        "classes. A frequency count of values the model already produced — no "
        "metric is derived here."
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Probability distribution
# ---------------------------------------------------------------------------


def _band_label(index: int) -> str:
    low = index * PROBABILITY_BAND_WIDTH
    high = low + PROBABILITY_BAND_WIDTH
    return f"{low:.1f}-{high:.1f}"


def _render_probability_distribution(data: Mapping[str, Any]) -> None:
    st.subheader("Probability distribution")
    probabilities = data.get("probabilities") or []
    if not probabilities:
        st.info("No probabilities were returned.")
        return

    numeric = [p for p in probabilities if isinstance(p, (int, float)) and not isinstance(p, bool)]
    if not numeric:
        st.info("The probabilities payload contained no numeric values.")
        return

    band_count = int(round(1 / PROBABILITY_BAND_WIDTH))
    counts = [0] * band_count
    for probability in numeric:
        index = int(probability / PROBABILITY_BAND_WIDTH)
        index = min(max(index, 0), band_count - 1)  # 1.0 falls in the last band
        counts[index] += 1

    rows = [{"band": _band_label(i), "count": counts[i]} for i in range(band_count)]

    st.bar_chart(
        rows,
        x="band",
        y="count",
        x_label="Predicted probability band",
        y_label="Records",
        sort=False,
        use_container_width=True,
    )

    label_semantics = (data.get("model_metadata") or {}).get("label_semantics")
    represents = (label_semantics or {}).get("probabilities_represent")
    st.caption(
        f"How the {len(numeric)} predicted probabilities fall across "
        f"{band_count} equal bands. "
        + (f"{represents} " if represents else "")
        + "Banding is a display aggregation of values the model already "
        "produced; no probability is modified."
    )


# ---------------------------------------------------------------------------
# Per-record detail
# ---------------------------------------------------------------------------


def _record_identifiers(data: Mapping[str, Any], count: int) -> tuple[list[str], bool]:
    """Identifiers for the prediction table, and whether they are real identity.

    Uses ``instance_ids`` when the API supplies one per record. It never pads,
    truncates, or invents an id: if the lengths disagree the panel falls back to
    positional labels and says so, because a silently trimmed id list is how one
    record acquires another record's identity.
    """
    instance_ids = data.get("instance_ids")
    if isinstance(instance_ids, Sequence) and not isinstance(instance_ids, (str, bytes)):
        if len(instance_ids) == count:
            return [str(identifier) for identifier in instance_ids], True
    return [f"Row {index}" for index in range(count)], False


def _render_prediction_table(data: Mapping[str, Any]) -> None:
    st.subheader("Prediction details")
    predictions = data.get("predictions") or []
    probabilities = data.get("probabilities") or []

    if not predictions:
        st.info("No predictions were returned.")
        return

    count = min(len(predictions), len(probabilities)) if probabilities else len(predictions)
    identifiers, is_identity = _record_identifiers(data, count)

    if is_identity:
        st.caption("Records are identified by their stable `instance_id`.")
    else:
        st.caption(
            "No usable `instance_ids` were supplied for these records, so rows "
            "are labelled by POSITION. A position is not an applicant identity "
            "and is not stable across runs."
        )

    rows = [
        {
            "record": identifiers[index],
            "prediction": predictions[index],
            "probability": probabilities[index] if probabilities else None,
        }
        for index in range(min(count, MAX_TABLE_ROWS))
    ]

    st.dataframe(rows, use_container_width=True, hide_index=True)
    if count > MAX_TABLE_ROWS:
        st.caption(f"Showing the first {MAX_TABLE_ROWS} of {count} records.")


def _render_feature_matrix(data: Mapping[str, Any]) -> None:
    """The scored input records, collapsed — an audit view, not a reading view."""
    feature_matrix = data.get("feature_matrix") or []
    with st.expander(
        f"Feature matrix — scored input records ({len(feature_matrix)})",
        expanded=False,
    ):
        if not feature_matrix:
            st.info("No feature matrix was returned.")
            return
        st.caption("The exact records scored by the model, unmodified.")
        st.dataframe(feature_matrix, use_container_width=True, hide_index=True)
