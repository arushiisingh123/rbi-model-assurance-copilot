"""Phase 2 Streamlit Dashboard (Owner: Khushi).

Consumes the Phase 2 API via api_client helper and displays structured evaluation
evidence for Model, Explainability, Fairness & Drift, and RBI Compliance.
"""
import streamlit as st

from dashboard.api_client import (
    get_compliance,
    get_explainability,
    get_fairness_drift,
    get_model,
    render_status,
)

st.set_page_config(page_title="AI Model Risk & Assurance Copilot", layout="centered")

st.title("AI Model Risk & Assurance Copilot")
st.caption("Phase 2 — Integration. Connected analytical modules for RBI compliance assurance.")

tab_model, tab_explain, tab_fair_drift, tab_compliance = st.tabs(
    ["Model", "Explainability", "Fairness & Drift", "Compliance"]
)

# -----------------------------------------------------------------------------
# 1. Model Tab
# -----------------------------------------------------------------------------
with tab_model:
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

# -----------------------------------------------------------------------------
# 2. Explainability Tab
# -----------------------------------------------------------------------------
with tab_explain:
    method = st.radio(
        "Explainability Method",
        ["shap", "lime"],
        horizontal=True,
        index=0,
    )
    data, source = get_explainability(method=method)
    st.caption(f"Data source: **{source}** | Mock data: **{data.get('is_mock', False)}**")

    st.subheader(f"{method.upper()} Global Feature Importance")
    importance_dict = data.get("global_importance", {})
    imp_records = [
        {"Feature": k, "Importance": v} for k, v in importance_dict.items()
    ]
    st.dataframe(imp_records, use_container_width=True)

    st.subheader(f"{method.upper()} Per-Instance Feature Contributions")
    st.json(data.get("per_instance", []))

# -----------------------------------------------------------------------------
# 3. Fairness & Drift Tab
# -----------------------------------------------------------------------------
with tab_fair_drift:
    data, source = get_fairness_drift()
    st.caption(f"Data source: **{source}**")

    fairness = data.get("fairness", {})
    drift = data.get("drift", {})

    st.subheader("Fairness Evaluation")
    st.markdown(f"**Fairness Status:** {render_status(fairness.get('status'))} *(Mock: {fairness.get('is_mock', False)})*")
    col1, col2, col3 = st.columns(3)
    col1.metric("Protected Attribute", fairness.get("protected_attribute", "N/A"))
    col2.metric("Demographic Parity Diff", fairness.get("demographic_parity_diff", "N/A"))
    col3.metric("Disparate Impact Ratio", fairness.get("disparate_impact_ratio", "N/A"))

    st.divider()

    st.subheader("Drift Detection")
    st.caption(
        "Note: Synthetic / controlled drift scenario per team-approved decision (2026-08-27). "
        "This demonstrates detection capability and is NOT observed production drift."
    )
    if drift.get("note"):
        st.info(f"ℹ️ {drift.get('note')}")

    st.markdown(f"**Drift Status:** {render_status(drift.get('status'))} *(Mock: {drift.get('is_mock', False)})*")
    col1, col2 = st.columns(2)
    col1.metric("Population Stability Index (PSI)", drift.get("psi", "N/A"))
    col2.metric("Kolmogorov-Smirnov (KS)", drift.get("ks_statistic", "N/A"))
    st.text(f"Features Evaluated: {', '.join(drift.get('features_evaluated', []))}")

# -----------------------------------------------------------------------------
# 4. Compliance Tab
# -----------------------------------------------------------------------------
with tab_compliance:
    data, source = get_compliance()
    st.caption(f"Data source: **{source}** | Mock data: **{data.get('is_mock', True)}** (illustrative sample rules)")

    st.subheader("RBI Compliance Findings")
    findings = data.get("findings", [])
    if not findings:
        st.info("No compliance findings returned.")
    for finding in findings:
        rule_title = f"{finding.get('rule_id', 'Rule')} — {render_status(finding.get('status'))}"
        with st.expander(rule_title, expanded=True):
            st.write(f"**Description:** {finding.get('rule_description', '')}")
            st.write(f"**Technical Reference:** `{finding.get('technical_finding_ref', '')}`")
            evidence = finding.get("evidence_chunks", [])
            if evidence:
                st.write("**Evidence Chunks:**")
                st.json(evidence)
            else:
                st.caption("No evidence chunks populated yet (Phase 3 RAG integration).")

st.divider()
st.caption(
    "AI Model Risk & Assurance Copilot — Phase 2 Integration. Model, Explainability, and Fairness evaluations run on real pipeline data; Drift evaluation runs on a controlled synthetic scenario; Compliance reflects illustrative sample rules."
)
