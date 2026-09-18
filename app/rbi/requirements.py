"""Regulatory requirements and deterministic evidence matching.

WHY THIS IS NOT "19 DOCUMENTS = 19 RULES"
    A source document contains many obligations. The unit of compliance is a
    REQUIREMENT ("the model must be independently validated"), traceable to
    the document, section and page it came from. One document therefore
    yields many requirements, and every requirement carries its own source
    location so a reviewer can go and read it.

    No requirement can be authored here. A requirement's text belongs to an
    RBI document, and none of the 19 sources is present -- so
    ``requirements_for_document()`` returns nothing but an explicit
    BLOCKED/EVIDENCE_MISSING record explaining why. Writing plausible
    requirement text from general knowledge would be fabricating regulation.

A PARALLEL STATUS VOCABULARY, ON PURPOSE
    The existing six-rule engine (``app/compliance/engine.py``) uses
    PASS/WARNING/FAIL/PENDING from ``app/config/thresholds.py``, shared with
    fairness and drift. That vocabulary stays exactly as it is. Requirement
    assessment needs distinctions it does not have -- "we could not tell if
    this applies" and "this applies but we have no evidence" are different
    answers, and both differ from "this failed".

    So this module defines its own ``RequirementStatus``. The two coexist;
    neither is converted into the other silently.

THE RULE THAT MATTERS MOST
    Absence of evidence is never PASS. A requirement with no matching
    evidence is EVIDENCE_MISSING; one whose applicability is unsettled is
    APPLICABILITY_UNCLEAR. Both are visible gaps. A compliance tool that
    reports "no problems found" when it simply looked at nothing is worse
    than one that reports nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from app.rbi.applicability import (
    APPLICABILITY_UNCLEAR as CTX_UNCLEAR,
    APPLIES,
    DOES_NOT_APPLY,
    ApplicabilityDecision,
    AssessmentContext,
    evaluate_applicability,
)
from app.rbi.manifest import RBIDocument

# ---------------------------------------------------------------------------
# Requirement status vocabulary (parallel to the six-rule engine's statuses)
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
PARTIAL = "PARTIAL"
NOT_ASSESSED = "NOT_ASSESSED"
APPLICABILITY_UNCLEAR = "APPLICABILITY_UNCLEAR"
EVIDENCE_MISSING = "EVIDENCE_MISSING"

REQUIREMENT_STATUSES = (
    PASS,
    FAIL,
    PARTIAL,
    NOT_ASSESSED,
    APPLICABILITY_UNCLEAR,
    EVIDENCE_MISSING,
)

# Statuses that assert a verified outcome. Everything else is a visible gap
# and must never be presented, counted or summarised as compliance.
CONCLUSIVE_STATUSES = (PASS, FAIL, PARTIAL)


@dataclass(frozen=True)
class SourceLocation:
    """Where in an RBI document a requirement came from.

    ``page`` / ``section_id`` are Optional because they are only knowable by
    reading the document. They are NOT filled with placeholders: a citation
    pointing at an invented page is worse than one that admits it has no
    page, because it looks checkable and is not.
    """

    document_id: str
    document_title: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    page: Optional[int] = None
    paragraph: Optional[int] = None
    chunk_id: Optional[str] = None

    def is_locatable(self) -> bool:
        """Whether this points at a specific place a reviewer could open."""
        return self.page is not None or bool(self.section_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "page": self.page,
            "paragraph": self.paragraph,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class EvidenceRequirement:
    """What the project must produce for a requirement to be assessable."""

    evidence_type: str
    description: str
    # Dotted path into the assurance evidence bundle, e.g.
    # "explainability.available" -- resolved deterministically, never fuzzily.
    finding_ref: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "description": self.description,
            "finding_ref": self.finding_ref,
        }


@dataclass(frozen=True)
class RegulatoryRequirement:
    """One obligation, traceable to the document section it came from."""

    requirement_id: str
    document_id: str
    requirement: str
    source: SourceLocation
    evidence_required: tuple[EvidenceRequirement, ...] = ()
    domain: tuple[str, ...] = ()
    provisional: bool = False
    provisional_note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "document_id": self.document_id,
            "requirement": self.requirement,
            "source": self.source.to_dict(),
            "evidence_required": [e.to_dict() for e in self.evidence_required],
            "domain": list(self.domain),
            "provisional": self.provisional,
            "provisional_note": self.provisional_note,
        }


@dataclass(frozen=True)
class RequirementFinding:
    """The assessment of one requirement against one project's evidence."""

    requirement_id: str
    document_id: str
    requirement: str
    applicability: str
    status: str
    reason: str
    source: SourceLocation
    evidence_required: tuple[str, ...] = ()
    evidence_found: tuple[dict[str, Any], ...] = ()
    provisional: bool = False
    provisional_note: Optional[str] = None
    citable: bool = False
    model_id: Optional[str] = None
    assurance_run_id: Optional[str] = None
    notes: tuple[str, ...] = field(default=())

    def is_conclusive(self) -> bool:
        return self.status in CONCLUSIVE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "document_id": self.document_id,
            "requirement": self.requirement,
            "applicability": self.applicability,
            "status": self.status,
            "reason": self.reason,
            "source": self.source.to_dict(),
            "evidence_required": list(self.evidence_required),
            "evidence_found": [dict(e) for e in self.evidence_found],
            "provisional": self.provisional,
            "provisional_note": self.provisional_note,
            "citable": self.citable,
            "model_id": self.model_id,
            "assurance_run_id": self.assurance_run_id,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Requirement extraction
# ---------------------------------------------------------------------------


class RequirementsUnavailable(Exception):
    """Requirements were asked for from a document that is not present."""


def requirements_for_document(document: RBIDocument) -> list[RegulatoryRequirement]:
    """Requirements extracted from one document's own text.

    Returns [] for any document that is not downloaded, because there is no
    text to extract from. It deliberately does NOT fall back to generating
    requirements from the document's title, domain tags or general knowledge
    of what such a circular usually says -- that would produce regulation
    this project invented and then attributed to the RBI.

    When sources are ingested, this is the single place requirement
    extraction is wired in; every requirement it returns must carry a
    SourceLocation pointing at the section it was read from.
    """
    if not document.is_citable():
        return []
    # Extraction from ingested text is implemented alongside ingestion; no
    # source document is present, so this path is unreachable today and is
    # deliberately left unwritten rather than stubbed with invented output.
    return []


def blocked_finding(
    document: RBIDocument,
    decision: ApplicabilityDecision,
    *,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> RequirementFinding:
    """A document-level finding recording that it could not be assessed.

    This is what an absent source produces: a visible NOT_ASSESSED entry
    naming the document and the blocking reason. It is emitted rather than
    the document being skipped, because a silently skipped regulation looks
    exactly like a regulation that passed.
    """
    return RequirementFinding(
        requirement_id=f"{document.document_id}:UNASSESSED",
        document_id=document.document_id,
        requirement=(
            f"Requirements from '{document.title}' could not be extracted or "
            "assessed."
        ),
        applicability=decision.applicability,
        status=NOT_ASSESSED,
        reason=document.blocking_reason()
        or "The source document is not available for assessment.",
        source=SourceLocation(
            document_id=document.document_id, document_title=document.title
        ),
        provisional=document.is_provisional(),
        provisional_note=document.provisional_note,
        citable=False,
        model_id=model_id,
        assurance_run_id=assurance_run_id,
        notes=tuple(decision.reasons),
    )


# ---------------------------------------------------------------------------
# Deterministic evidence matching
# ---------------------------------------------------------------------------


def resolve_evidence(evidence_bundle: Any, ref: Optional[str]) -> tuple[bool, Any]:
    """Look up a dotted path in the assurance evidence. (found, value).

    Exact-path only. No fuzzy or semantic matching: evidence must be the
    thing the requirement actually asked for, not something that resembles
    it.
    """
    if not ref:
        return False, None
    current = evidence_bundle
    for part in str(ref).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


_EMPTY = (None, "", [], {}, ())


def assess_requirement(
    requirement: RegulatoryRequirement,
    document: RBIDocument,
    context: AssessmentContext,
    evidence_bundle: Any,
    *,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> RequirementFinding:
    """Assess one requirement. Deterministic; no model involved.

    Decision order, and why:
      1. Applicability first -- an inapplicable requirement is not a failure,
         and an unclear one must not be assessed at all.
      2. Evidence presence second -- with nothing to assess, the answer is
         EVIDENCE_MISSING, never PASS.
      3. Only then is a verdict formed, and only from evidence that is
         actually present.
    """
    decision = evaluate_applicability(document, context)
    base = {
        "requirement_id": requirement.requirement_id,
        "document_id": requirement.document_id,
        "requirement": requirement.requirement,
        "source": requirement.source,
        "evidence_required": tuple(e.evidence_type for e in requirement.evidence_required),
        "provisional": requirement.provisional or document.is_provisional(),
        "provisional_note": requirement.provisional_note or document.provisional_note,
        "citable": document.is_citable(),
        "model_id": model_id,
        "assurance_run_id": assurance_run_id,
        "notes": tuple(decision.reasons),
    }

    if decision.applicability == DOES_NOT_APPLY:
        return RequirementFinding(
            applicability=DOES_NOT_APPLY,
            status=NOT_ASSESSED,
            reason=(
                "This requirement does not apply to the assessed entity/context, "
                "so it was not assessed. " + (decision.reasons[-1] if decision.reasons else "")
            ).strip(),
            **base,
        )

    if decision.applicability == CTX_UNCLEAR:
        return RequirementFinding(
            applicability=CTX_UNCLEAR,
            status=APPLICABILITY_UNCLEAR,
            reason=(
                "Applicability could not be determined from the document's "
                "metadata, so no compliance conclusion was drawn. "
                + (decision.reasons[-1] if decision.reasons else "")
            ).strip(),
            **base,
        )

    # APPLIES -- now look for the evidence it demands.
    found: list[dict[str, Any]] = []
    missing: list[str] = []
    for needed in requirement.evidence_required:
        present, value = resolve_evidence(evidence_bundle, needed.finding_ref)
        if present and value not in _EMPTY:
            found.append(
                {
                    "evidence_type": needed.evidence_type,
                    "finding_ref": needed.finding_ref,
                    "value": value,
                }
            )
        else:
            missing.append(needed.evidence_type)

    if not requirement.evidence_required:
        return RequirementFinding(
            applicability=APPLIES,
            status=NOT_ASSESSED,
            reason=(
                "The requirement applies but declares no evidence contract, so "
                "there is nothing to assess it against."
            ),
            evidence_found=(),
            **base,
        )

    if not found:
        return RequirementFinding(
            applicability=APPLIES,
            status=EVIDENCE_MISSING,
            reason=(
                "The requirement applies, but none of the evidence it requires "
                f"({', '.join(missing)}) is present in this assurance run. "
                "Absence of evidence is not compliance."
            ),
            evidence_found=(),
            **base,
        )

    if missing:
        return RequirementFinding(
            applicability=APPLIES,
            status=PARTIAL,
            reason=(
                f"{len(found)} of {len(requirement.evidence_required)} required "
                f"evidence items are present; still missing: {', '.join(missing)}."
            ),
            evidence_found=tuple(found),
            **base,
        )

    return RequirementFinding(
        applicability=APPLIES,
        status=PASS,
        reason=(
            "Every evidence item this requirement declares is present in the "
            "assurance run."
        ),
        evidence_found=tuple(found),
        **base,
    )


def assess_corpus(
    documents: Iterable[RBIDocument],
    context: AssessmentContext,
    evidence_bundle: Any,
    *,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> list[RequirementFinding]:
    """Assess every requirement of every in-scope document.

    A document that yields no requirements (because its source is absent)
    produces one explicit NOT_ASSESSED finding rather than silently
    contributing nothing.
    """
    findings: list[RequirementFinding] = []
    for document in documents:
        decision = evaluate_applicability(document, context)
        if decision.applicability == DOES_NOT_APPLY:
            continue

        requirements = requirements_for_document(document)
        if not requirements:
            findings.append(
                blocked_finding(
                    document,
                    decision,
                    model_id=model_id,
                    assurance_run_id=assurance_run_id,
                )
            )
            continue

        for requirement in requirements:
            findings.append(
                assess_requirement(
                    requirement,
                    document,
                    context,
                    evidence_bundle,
                    model_id=model_id,
                    assurance_run_id=assurance_run_id,
                )
            )
    return findings


def summarise(findings: Iterable[RequirementFinding]) -> dict[str, Any]:
    """Counts per status, plus how much of the corpus is actually assessed.

    ``assessed_fraction`` counts only CONCLUSIVE statuses. A run where every
    requirement is EVIDENCE_MISSING scores 0.0, which is the honest number.
    """
    findings = list(findings)
    counts = {status: 0 for status in REQUIREMENT_STATUSES}
    for finding in findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1
    conclusive = sum(counts[s] for s in CONCLUSIVE_STATUSES)
    return {
        "total": len(findings),
        "by_status": counts,
        "conclusive": conclusive,
        "assessed_fraction": (conclusive / len(findings)) if findings else 0.0,
        "provisional_findings": sum(1 for f in findings if f.provisional),
        "citable_findings": sum(1 for f in findings if f.citable),
    }


__all__ = [
    "PASS",
    "FAIL",
    "PARTIAL",
    "NOT_ASSESSED",
    "APPLICABILITY_UNCLEAR",
    "EVIDENCE_MISSING",
    "REQUIREMENT_STATUSES",
    "CONCLUSIVE_STATUSES",
    "SourceLocation",
    "EvidenceRequirement",
    "RegulatoryRequirement",
    "RequirementFinding",
    "RequirementsUnavailable",
    "requirements_for_document",
    "blocked_finding",
    "resolve_evidence",
    "assess_requirement",
    "assess_corpus",
    "summarise",
]
