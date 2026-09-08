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
    get_report,
    render_status,
)

st.set_page_config(page_title="AI Model Risk & Assurance Copilot", layout="centered")

st.title("AI Model Risk & Assurance Copilot")
st.caption("Phase 3 — RAG + LLM Reporting. Connected analytical modules and evidence-grounded assurance.")

tab_model, tab_explain, tab_fair_drift, tab_compliance, tab_report = st.tabs(
    ["Model", "Explainability", "Fairness & Drift", "Compliance", "Report"]
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
        "Note: Production assurance evaluates drift by comparing the development training split "
        "with the held-out test split. The fallback (used only if the API is unreachable) is a "
        "static mock fixture (not a build_drift_scenario() run)."
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

# -----------------------------------------------------------------------------
# 5. Report Tab (Phase 3 — LLM Reporting Wiring)
# -----------------------------------------------------------------------------
with tab_report:
    data, source = get_report()
    st.caption(f"Data source: **{source}** | Mock data: **{data.get('is_mock', True)}**")

    if data.get("is_mock", True):
        st.warning(
            "⚠️ **PROVISIONAL MOCK REPORT**: This report is synthetic mock data for "
            "API and Dashboard Phase 3 integration testing. No real LLM generation or "
            "live vector database retrieval was executed."
        )

    cov = data.get("evidence_coverage", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Report ID", data.get("report_id", "N/A"))
    col2.metric("Model Version", data.get("model_version", "N/A"))
    col3.metric("Evidence Retrieved", f"{cov.get('retrieved', 0)} / {cov.get('total', 0)}")
    col4.metric("Evidence Not Found", cov.get("not_found", 0))

    st.subheader("Assurance Findings & Evidence-Grounded Report")

    sections = data.get("sections", [])
    if not sections:
        st.info("No report sections available.")

    for sec in sections:
        heading = sec.get("heading", "Report Section")
        st.markdown(f"### {heading}")

        # Layer 1: Technical Finding
        tf = sec.get("technical_finding", {})
        with st.container():
            st.markdown(
                f"**Layer 1 — Technical Finding** | Status: {render_status(tf.get('status'))} "
                f"| Ref: `{tf.get('ref', 'N/A')}` *(Source: `{tf.get('source_module', 'N/A')}`)*"
            )
            st.json(tf.get("value", {}))

        # Layer 2: Retrieved Evidence
        re = sec.get("retrieved_evidence", {})
        status = re.get("evidence_status", "NOT_ATTEMPTED")
        with st.container():
            st.markdown(f"**Layer 2 — Retrieved Regulatory Evidence** (Status: `{status}`)")
            if status == "NOT_FOUND":
                st.error("⚠️ **No verified RBI regulatory evidence retrieved.** (Safety fallback: no regulatory requirement claimed)")
            elif status == "RETRIEVED":
                citations = re.get("citations", [])
                if citations:
                    for cit in citations:
                        st.markdown(
                            f"- **Source:** {cit.get('source', '')} | **Locator:** `{cit.get('locator', '')}` "
                            f"*(Provenance: `{cit.get('provenance', '')}`)*\n"
                            f"  > \"{cit.get('quote', '')}\""
                        )
                else:
                    st.caption("No citations listed.")
            else:
                st.caption("Evidence retrieval not attempted.")

        # Layer 3: LLM Interpretation
        llm = sec.get("llm_interpretation", {})
        with st.container():
            basis = llm.get("regulatory_basis", "none")
            st.markdown(f"**Layer 3 — LLM Interpretation** *(Regulatory Basis: `{basis}`)*")
            st.info(llm.get("text", "No interpretation generated."))
            grounded = llm.get("grounded_in", [])
            if grounded:
                st.caption(f"Grounded strictly in: {', '.join(f'`{g}`' for g in grounded)}")

        st.divider()

    disclaimers = data.get("disclaimers", [])
    if disclaimers:
        st.subheader("Disclaimers & Notes")
        for d in disclaimers:
            st.caption(f"• {d}")

st.divider()
st.caption(
    "AI Model Risk & Assurance Copilot — Phase 3. Model, Explainability, and Fairness evaluations run on real pipeline data; Drift detection compares dev train/test splits; Compliance reflects illustrative sample rules; Report wiring is mock-backed (PROVISIONAL)."
)
