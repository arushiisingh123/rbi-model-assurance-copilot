"""Assurance report presentation panel (Owner: Nidhi — Phase 4).

Renders the ``ReportResult`` payload returned by ``GET /report``
(``dashboard.api_client.get_report()``). This module presents the report
faithfully -- it does not call the LLM, does not call RAG retrieval, does
not compute or reclassify any technical finding, and does not merge the
report's layers into one another. It fetches nothing itself: the caller
(the dashboard shell) fetches the data and passes it in, matching the
project's existing data-fetching-outside-the-panel convention.

REPORT LAYERS RENDERED SEPARATELY, IN THIS ORDER, PER SECTION
    1. Technical finding      -- the deterministic analytical result (Layer 1)
    2. LLM interpretation     -- the narrative layer, where available (Layer 3)
    3. Supporting evidence    -- structured records from Manas's/Arushi's
                                 evidence builders, routed to this section
    4. Retrieved evidence     -- RAG citations for this section (Layer 2),
                                 including full source attribution

These are never collapsed into one another: interpretation text is never
folded into the technical finding, and a citation's fields are never
copied into the interpretation.

EVIDENCE HONESTY RULES ENFORCED HERE
    - A missing/`None` citation metadata field renders as "not stated" --
      never guessed, never left silently blank in a way that could be
      misread as "not applicable" or "no".
    - `is_excerpt` / `is_current` are always rendered explicitly when a
      citation exists, precisely because the one approved RBI source is a
      limited 2014 excerpt (`is_excerpt: True`, `is_current: False`).
      Retrieving it must never be presented as current or binding
      regulation.
    - `NOT_FOUND` always renders as "no verified evidence was retrieved
      from the current approved/indexed corpus" -- never as an absence of
      RBI regulation.
    - `NOT_ATTEMPTED` (declared in the schema, not currently produced by
      `generate_report()`) is handled distinctly from `NOT_FOUND` rather
      than folded into the same branch.
    - The citation-level `provenance` field (a short classification string:
      "verified" / "illustrative" / "interim_single_document") is labelled
      as evidence classification, not confused with the unrelated
      `provenance` dict that lives on `RBIEvidence` / `TechnicalFinding`
      elsewhere in the codebase.
"""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st

_NOT_STATED = "not stated"


def _or_not_stated(value: Optional[Any]) -> str:
    """String form of ``value``, or the literal ``"not stated"`` for ``None``.

    Never guesses a replacement value -- an absent field stays visibly
    absent.
    """
    return _NOT_STATED if value is None else str(value)


def render_report_panel(data: dict[str, Any], source: str) -> None:
    """Render the full assurance report panel.

    Parameters
    ----------
    data
        The ``ReportResult``-shaped dict already fetched by the caller (for
        example via ``dashboard.api_client.get_report()``).
    source
        ``"api"`` or ``"fallback"`` -- where ``data`` came from, as already
        determined by the caller.
    """
    is_mock = data.get("is_mock", True)
    st.caption(f"Data source: **{source}** | Report is_mock: **{is_mock}**")

    if is_mock:
        st.warning(
            "⚠️ This report is marked **is_mock: True**. Treat every section "
            "below as illustrative, not a verified regulatory assessment."
        )

    coverage = data.get("evidence_coverage", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Report ID", data.get("report_id", _NOT_STATED))
    col2.metric("Model Version", data.get("model_version", _NOT_STATED))
    col3.metric("Evidence Retrieved", f"{coverage.get('retrieved', 0)} / {coverage.get('total', 0)}")
    col4.metric("Evidence Not Found", coverage.get("not_found", 0))

    st.subheader("Assurance Findings & Evidence-Grounded Report")

    sections = data.get("sections", [])
    if not sections:
        st.info("No report sections available.")

    for section in sections:
        _render_section(section)

    disclaimers = data.get("disclaimers", [])
    if disclaimers:
        st.subheader("Disclaimers & Notes")
        for disclaimer in disclaimers:
            st.caption(f"• {disclaimer}")


def _render_section(section: dict[str, Any]) -> None:
    heading = section.get("heading", "Report Section")
    st.markdown(f"### {heading}")

    # 1. Technical finding (Layer 1) -- the analytical result, verbatim.
    _render_technical_finding(section.get("technical_finding", {}))

    # 2. LLM interpretation (Layer 3) -- narrative layer, where available.
    _render_llm_interpretation(section.get("llm_interpretation", {}))

    # 3. Supporting evidence -- structured records routed to this section.
    _render_supporting_evidence(section.get("supporting_evidence", []))

    # 4. Retrieved evidence / citations (Layer 2).
    _render_retrieved_evidence(section.get("retrieved_evidence", {}))

    st.divider()


def _render_technical_finding(finding: dict[str, Any]) -> None:
    from dashboard.api_client import render_status

    with st.container():
        st.markdown(
            f"**1. Technical Finding** | Status: {render_status(finding.get('status'))} "
            f"| Ref: `{finding.get('ref', _NOT_STATED)}` "
            f"*(Source module: `{finding.get('source_module', _NOT_STATED)}`, "
            f"provenance: `{finding.get('provenance', _NOT_STATED)}`)*"
        )
        value = finding.get("value")
        if isinstance(value, dict) and value:
            # Structured, generic key/value listing -- no domain-specific
            # interpretation of what a model/fairness/drift/compliance value
            # means, and no raw st.json dump.
            for key, val in value.items():
                st.markdown(f"- **{key}:** {val}")
        elif value not in (None, {}, []):
            st.write(value)
        else:
            st.caption("No finding value recorded.")


def _render_llm_interpretation(interpretation: dict[str, Any]) -> None:
    with st.container():
        basis = interpretation.get("regulatory_basis", _NOT_STATED)
        is_mock = interpretation.get("is_mock")
        st.markdown(
            f"**2. Interpretation** *(Regulatory basis: `{basis}`, "
            f"is_mock: **{is_mock}**)*"
        )
        text = interpretation.get("text")
        if text:
            st.info(text)
        else:
            st.caption("No interpretation available for this section.")
        grounded_in = interpretation.get("grounded_in", [])
        if grounded_in:
            st.caption(f"Grounded strictly in: {', '.join(f'`{ref}`' for ref in grounded_in)}")


def _render_supporting_evidence(records: list[dict[str, Any]]) -> None:
    st.markdown("**3. Supporting Evidence**")
    if not records:
        st.caption(
            "No supporting evidence records are attached to this section. "
            "This does not mean none exists elsewhere -- only that none was "
            "routed to this section."
        )
        return
    for record in records:
        if not isinstance(record, dict):
            continue
        evidence_type = record.get("evidence_type", _NOT_STATED)
        with st.expander(f"Evidence record: `{evidence_type}`", expanded=False):
            # Generic key/value listing -- carried through verbatim, no
            # reinterpretation of what a fairness/explainability evidence
            # record's fields mean.
            for key, val in record.items():
                st.markdown(f"- **{key}:** {val}")


def _render_retrieved_evidence(retrieved: dict[str, Any]) -> None:
    status = retrieved.get("evidence_status", "NOT_ATTEMPTED")
    st.markdown(f"**4. Retrieved Regulatory Evidence** (Status: `{status}`)")

    if status == "NOT_FOUND":
        st.error(
            "No verified evidence was retrieved from the current "
            "approved/indexed corpus. This does not mean that no RBI rule "
            "exists -- only that nothing was retrieved for this query."
        )
        return

    if status == "NOT_ATTEMPTED":
        st.caption(
            "Evidence retrieval was not attempted for this section "
            "(distinct from NOT_FOUND, which means retrieval ran and found "
            "nothing verified)."
        )
        return

    if status != "RETRIEVED":
        # Defensive: an evidence_status outside the known vocabulary is
        # shown as-is rather than silently treated as any of the above.
        st.caption(f"Unrecognised evidence status: `{status}`.")
        return

    citations = retrieved.get("citations", [])
    if not citations:
        st.caption("Status is RETRIEVED but no citations were provided.")
        return

    for citation in citations:
        _render_citation(citation)


def _render_citation(citation: dict[str, Any]) -> None:
    is_excerpt = citation.get("is_excerpt")
    is_current = citation.get("is_current")

    if is_excerpt is True:
        excerpt_line = "**Excerpt status:** Limited excerpt — not the complete source document."
    elif is_excerpt is False:
        excerpt_line = "**Excerpt status:** Full document (not an excerpt)."
    else:
        excerpt_line = f"**Excerpt status:** {_NOT_STATED}."

    if is_current is True:
        currency_line = "**Currency:** Current, per source metadata."
    elif is_current is False:
        currency_line = "**Currency:** NOT current — historical/superseded. Do not treat as binding current regulation."
    else:
        currency_line = f"**Currency:** {_NOT_STATED}."

    source_url = citation.get("source_url")
    if source_url:
        source_url_line = f"**Source URL:** [{source_url}]({source_url})"
    else:
        source_url_line = f"**Source URL:** {_NOT_STATED}"

    with st.container():
        st.markdown(
            f"- **Source:** {citation.get('source', _NOT_STATED)} "
            f"| **Locator:** `{citation.get('locator', _NOT_STATED)}` "
            f"| **Evidence classification (provenance):** `{citation.get('provenance', _NOT_STATED)}`\n"
            f"  > \"{citation.get('quote', _NOT_STATED)}\""
        )
        st.markdown(
            f"  {excerpt_line}  \n"
            f"  {currency_line}  \n"
            f"  **Document type:** {_or_not_stated(citation.get('document_type'))}  \n"
            f"  **Publication date:** {_or_not_stated(citation.get('publication_date'))}  \n"
            f"  {source_url_line}"
        )
