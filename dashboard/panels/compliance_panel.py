"""RBI compliance presentation panel (Owner: Nidhi — Phase 4).

Renders the ``ComplianceResult`` payload returned by ``GET /compliance``
(``dashboard.api_client.get_compliance()``). This module presents the
result faithfully -- it does not call the compliance engine, does not
recompute a status, does not classify a finding, and does not populate
``evidence_chunks`` from any other source. It fetches nothing itself: the
caller (the dashboard shell) fetches the data and passes it in, matching
the project's existing data-fetching-outside-the-panel convention (see
``dashboard/dashboard_app.py``'s current tab bodies, which already follow
``data, source = get_compliance()`` before rendering).

WHAT THIS MODULE IS NOT
    - Not the compliance engine (`app/compliance/`). It never evaluates a
      rule or derives a PASS/WARNING/FAIL/PENDING status; it only displays
      the status the engine already produced.
    - Not a RAG evidence renderer. `ComplianceFinding.evidence_chunks` is a
      distinct, currently-always-empty field on the illustrative rule
      engine's own output -- it has never been wired to the RAG pipeline.
      The real RAG-grounded evidence for the compliance domain is rendered
      by ``dashboard/panels/report_panel.py`` (via `GET /report`'s
      compliance section, `retrieved_evidence.citations`), not here. This
      panel says so explicitly rather than implying the two are the same
      channel.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.api_client import render_status


def render_compliance_panel(data: dict[str, Any], source: str) -> None:
    """Render the RBI compliance findings panel.

    Parameters
    ----------
    data
        The ``ComplianceResult``-shaped dict already fetched by the caller
        (for example via ``dashboard.api_client.get_compliance()``).
    source
        ``"api"`` or ``"fallback"`` -- where ``data`` came from, as already
        determined by the caller.
    """
    is_mock = data.get("is_mock", True)
    st.caption(
        f"Data source: **{source}** | Mock data: **{is_mock}** "
        "(illustrative sample RBI rules -- not verified regulatory text)"
    )

    st.subheader("RBI Compliance Findings")

    findings = data.get("findings", [])
    if not findings:
        st.info("No compliance findings returned.")
        return

    for finding in findings:
        _render_finding(finding)


def _render_finding(finding: dict[str, Any]) -> None:
    """One compliance finding: rule identity, status, and its evidence state."""
    rule_id = finding.get("rule_id", "Rule")
    status = finding.get("status")
    title = f"{rule_id} — {render_status(status)}"

    with st.expander(title, expanded=True):
        st.write(f"**Description:** {finding.get('rule_description', 'not stated')}")
        st.write(f"**Technical Reference:** `{finding.get('technical_finding_ref', 'not stated')}`")

        evidence_chunks = finding.get("evidence_chunks", [])
        st.markdown("**Evidence:**")
        if evidence_chunks:
            # Rendered exactly as returned -- this panel never invents,
            # merges, or reformats an evidence_chunks entry.
            for chunk in evidence_chunks:
                st.markdown(f"- {chunk}")
        else:
            st.caption(
                "No evidence chunks are populated for this finding. This is "
                "the compliance engine's own `evidence_chunks` field, which "
                "is not currently populated by any producer -- it is not "
                "the same channel as the RAG-retrieved regulatory evidence "
                "shown on the Report tab."
            )
