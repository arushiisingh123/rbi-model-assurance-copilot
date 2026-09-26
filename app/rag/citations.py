"""Turn retrieved RBI evidence into a structured citation (owner: Nidhi / RAG).

WHAT THIS IS FOR
    ``app.rag.evidence.RBIEvidence`` already carries everything a citation
    needs, but carries some of it inside a ``provenance`` dict. A consumer --
    the API, the report, the dashboard -- should not have to know which keys
    live where, nor repeat the "is this page real or did I imagine it" check.
    This module does that lifting once.

TWO CLAUSE NUMBERS, NEVER MERGED
    ``source_clause`` is what the cited PDF prints. ``register_clause`` is
    what ``app/rbi/verified_requirements.py`` cites for the same provision,
    read from the RBI website. They differ for fourteen of the sixteen
    verified requirements. Both are carried; neither is derived from the other; the
    pairing comes from ``app/rbi/clause_crossref.py``, whose rows were
    established by locating the register's own quote inside the document.

    A caller rendering these MUST label them separately. Showing "16.13" as
    though the cited PDF printed it would be a fabricated reference, because
    that PDF prints "16(m)".

THE NULL RULE
    A field the evidence does not carry stays ``None``. The 2014 text excerpt
    has no pages and no clause numbering, so a citation to it has
    ``page=None`` and ``source_clause=None`` -- not page 1, not clause 1.
"""
from __future__ import annotations

from typing import Any, Optional

from app.api.schemas import Citation


def _locator(page: Optional[int], source_clause: Optional[str], chunk_index) -> str:
    """A human-readable pointer to where a quote came from.

    Prefers what a reviewer can look up -- "page 16, clause 16(m)" -- and
    falls back to the chunk index only when the source has neither. Each part
    appears only when known, so the locator never implies precision it lacks.
    """
    parts = []
    if isinstance(page, int):
        parts.append(f"page {page}")
    if source_clause:
        parts.append(f"clause {source_clause}")
    if parts:
        return ", ".join(parts)
    return f"chunk #{chunk_index}"


def citation_from_evidence(
    evidence: Any,
    *,
    provenance_label: str = "interim_multi_document",
) -> Citation:
    """Build a structured ``Citation`` from one ``RBIEvidence`` record.

    ``evidence`` is duck-typed rather than imported so this module does not
    pull ``app.rag.evidence`` (and ChromaDB with it) into every consumer.
    """
    provenance = dict(getattr(evidence, "provenance", None) or {})

    page = provenance.get("page")
    if not isinstance(page, int):
        # Chroma stores metadata as scalars and a round-trip can stringify an
        # int. A page that cannot be read back as a number is reported as
        # unknown rather than coerced into a plausible one.
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None

    source_clause = provenance.get("section_id") or None

    from app.rbi.clause_crossref import register_clause_for_source

    register_clause = register_clause_for_source(
        getattr(evidence, "doc_id", None), source_clause
    )

    return Citation(
        source=getattr(evidence, "title", None) or getattr(evidence, "doc_id", "") or "",
        locator=_locator(page, source_clause, getattr(evidence, "chunk_index", 0)),
        quote=(getattr(evidence, "text", "") or "").strip(),
        provenance=provenance_label,
        source_url=getattr(evidence, "source_url", None),
        publication_date=getattr(evidence, "publication_date", None),
        document_type=getattr(evidence, "document_type", None),
        is_excerpt=_as_bool(getattr(evidence, "is_excerpt", None)),
        is_current=_as_bool(getattr(evidence, "is_current", None)),
        page=page,
        source_clause=source_clause,
        section_title=provenance.get("section_title") or None,
        register_clause=register_clause,
        reference_number=provenance.get("reference_number") or None,
        chunk_id=getattr(evidence, "chunk_id", None),
    )


def _as_bool(value: Any) -> Optional[bool]:
    """Chroma metadata round-trips booleans as strings; restore them.

    Returns ``None`` for anything unrecognised, because "we could not tell
    whether this source is current" must not collapse into "it is current".
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def citations_from_evidence(records: Any) -> list[Citation]:
    """Structured citations for a list of evidence records (may be empty)."""
    return [citation_from_evidence(record) for record in (records or [])]


def attach_citations_to_findings(
    findings: Any,
    evidence_by_rule: Any,
    *,
    as_dict: bool = True,
) -> list:
    """Return ``findings`` with structured ``citations`` attached per rule.

    ONE PLACE, TWO CALLERS
        ``GET /compliance`` and ``build_assurance_result()`` both retrieve RBI
        evidence per rule, and both must present it the same way. Before this
        helper only the first attached structured citations, so the same
        finding carried a full citation on one endpoint and nothing but
        chunk-ids on another -- and the downloadable PDF, which is built from
        the second path, silently lost the regulatory grounding the JSON
        already had.

    The findings are COPIED, never mutated: the rule engine's own output shape
    is asserted exactly by its tests, and this must stay additive to it.

    Attaching a citation NEVER changes a status. The rule engine decides every
    status from technical findings alone, before this is called; an empty
    citation list means no supporting text was retrieved, not that a check
    failed.

    ``as_dict`` returns plain dicts (what a FastAPI route hands back);
    ``False`` keeps ``Citation`` models, for a caller that wants them typed.
    """
    evidence_by_rule = evidence_by_rule or {}
    out = []
    for finding in findings or []:
        citations = citations_from_evidence(evidence_by_rule.get(finding["rule_id"]))
        out.append(
            {
                **finding,
                "citations": [c.model_dump() for c in citations]
                if as_dict
                else citations,
            }
        )
    return out


__all__ = [
    "citation_from_evidence",
    "citations_from_evidence",
    "attach_citations_to_findings",
]


def attach_rule_provenance(findings: list) -> list:
    """Stamp each technical finding with its rule's own ``rbi_source``.

    Done HERE, at the API boundary, rather than inside
    ``app.compliance.compliance``: that module's finding dict has an approved
    key set locked by its own tests, so it is left exactly as agreed.

    Every rule in ``app/rbi/rules`` carries the ILLUSTRATIVE marker, and
    ``ComplianceFinding``'s docstring has always said findings carry it -- but
    the field was absent from the schema and never populated, so the API
    reported ``None``. That left the dashboard with nothing to distinguish a
    project-threshold FAIL from a regulatory breach. Purely additive: no
    status, metric or citation is touched.
    """
    from app.rbi.rules import load_rules

    source_by_rule = {
        rule["rule_id"]: rule.get("rbi_source") for rule in load_rules()
    }
    stamped = []
    for finding in findings:
        record = dict(finding)
        record["rbi_source"] = source_by_rule.get(record.get("rule_id"))
        stamped.append(record)
    return stamped
