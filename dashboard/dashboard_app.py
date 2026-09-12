"""Phase 4 Streamlit Dashboard (Owner: Khushi).

Shell and integration layer only. Fetches each analytical domain's result via
``dashboard/api_client.py`` and hands the already-fetched data to that
domain's own dashboard panel (``dashboard/panels/``) to render. This module
owns tab placement, navigation, and failure isolation between tabs -- it does
not calculate, recalculate, or reshape any analytical value itself.

Panel ownership (Phase 4, docs/decisions.md "Phase 4 allocation," D2):
    - Explainability -- Manas (``dashboard/panels/explainability_panel.py``)
    - Fairness & Drift -- Arushi (``dashboard/panels/fairness_drift_panel.py``)
    - Compliance, Report -- Nidhi (``dashboard/panels/compliance_panel.py``,
      ``dashboard/panels/report_panel.py``)
    - Model -- Namitha's panel does not exist yet. The Model tab below is the
      original inline rendering from earlier development, kept as-is (while
      ``model_metrics`` is now exposed by the API as of this change, dedicated
      panel visualization remains to be wired when Namitha's panel lands);
      it will be replaced by a delegated panel call once available.
"""
import streamlit as st

from dashboard.api_client import (
    get_compliance,
    get_explainability,
    get_fairness_drift,
    get_health,
    get_model,
    get_report,
)
from dashboard.panels.compliance_panel import render_compliance_panel
from dashboard.panels.explainability_panel import render as render_explainability_panel
from dashboard.panels.fairness_drift_panel import render_fairness_drift_panel
from dashboard.panels.report_panel import render_report_panel

st.set_page_config(page_title="AI Model Risk & Assurance Copilot", layout="centered")

st.title("AI Model Risk & Assurance Copilot")
st.caption(
    "Phase 4 — Dashboard + UX. Each tab is rendered by its owning analytical "
    "module's panel from the data this shell fetches; nothing is calculated "
    "or reshaped here."
)

health_data, health_source = get_health()
if health_source == "api" and health_data.get("status") == "ok":
    st.success("API Backend: Connected (http://127.0.0.1:8000)")
else:
    st.warning("API Backend: Unreachable — Operating in offline mock/fallback mode")

tab_model, tab_explain, tab_fair_drift, tab_compliance, tab_report = st.tabs(
    ["Model", "Explainability", "Fairness & Drift", "Compliance", "Report"]
)


def _safe_render(label: str, render_fn) -> None:
    """Run ``render_fn()``, isolating any failure to this tab only.

    A malformed API response or an unexpected exception inside one panel
    must not blank the rest of the dashboard -- the other tabs still need to
    render on the same page load.
    """
    try:
        render_fn()
    except Exception as exc:  # noqa: BLE001 - deliberately broad: last-resort tab isolation
        st.error(f"{label} tab failed to render: {exc}")


# -----------------------------------------------------------------------------
# 1. Model Tab (inline -- no dedicated panel exists yet, see module docstring)
# -----------------------------------------------------------------------------
with tab_model:
    def _render_model_tab() -> None:
        data, source = get_model()
        st.caption(f"Data source: **{source}** | Mock data: **{data.get('is_mock', False)}**")

        st.subheader("Model Metadata")
        meta = data.get("model_metadata", {})
        col1, col2 = st.columns(2)
        col1.metric("Model Type", meta.get("model_type", "N/A"))
        col2.metric("Version", meta.get("version", "N/A"))
        st.text(f"Trained on: {meta.get('trained_on', 'N/A')}")
        st.text(f"Features: {', '.join(meta.get('feature_names', []))}")

        label_sem = meta.get("label_semantics")
        if label_sem:
            st.info(
                f"**Target Semantics**: 0 = GOOD (favorable outcome), 1 = BAD (positive class) | "
                f"{label_sem.get('probabilities_represent', '')}"
            )

        st.subheader("Predictions & Probabilities")
        preds = data.get("predictions", [])
        probs = data.get("probabilities", [])
        pred_records = [
            {"Row": i, "Prediction": pred, "Probability": prob}
            for i, (pred, prob) in enumerate(zip(preds, probs))
        ]
        st.dataframe(pred_records, use_container_width=True)

        st.subheader("Feature Matrix (Input Records)")
        st.dataframe(data.get("feature_matrix", []), use_container_width=True)

    _safe_render("Model", _render_model_tab)

# -----------------------------------------------------------------------------
# 2. Explainability Tab (owner: Manas)
# -----------------------------------------------------------------------------
with tab_explain:
    def _render_explainability_tab() -> None:
        # The panel owns method selection and fetching -- it re-fetches on
        # every shap/lime change, so the fetch callable is injected rather
        # than pre-fetched here. instance_ids is intentionally omitted:
        # GET /explainability does not carry them yet (see
        # docs/module-interfaces.md, Phase 4 explainability section), and the
        # panel's documented, tested fallback is honest positional labelling.
        render_explainability_panel(fetch_explainability=get_explainability)

    _safe_render("Explainability", _render_explainability_tab)

# -----------------------------------------------------------------------------
# 3. Fairness & Drift Tab (owner: Arushi)
# -----------------------------------------------------------------------------
with tab_fair_drift:
    def _render_fairness_drift_tab() -> None:
        data, source = get_fairness_drift()
        render_fairness_drift_panel(data=data, source=source)

    _safe_render("Fairness & Drift", _render_fairness_drift_tab)

# -----------------------------------------------------------------------------
# 4. Compliance Tab (owner: Nidhi)
# -----------------------------------------------------------------------------
with tab_compliance:
    def _render_compliance_tab() -> None:
        data, source = get_compliance()
        render_compliance_panel(data=data, source=source)

    _safe_render("Compliance", _render_compliance_tab)

# -----------------------------------------------------------------------------
# 5. Report Tab (owner: Nidhi)
# -----------------------------------------------------------------------------
with tab_report:
    def _render_report_tab() -> None:
        data, source = get_report()
        render_report_panel(data=data, source=source)

    _safe_render("Report", _render_report_tab)

st.divider()
st.caption(
    "AI Model Risk & Assurance Copilot — Phase 4. Each tab displays results "
    "exactly as its owning analytical module produced them; no value is "
    "recalculated, overridden, or fabricated in the dashboard. Data source "
    "and mock/fallback status are shown at the top of each tab."
)
