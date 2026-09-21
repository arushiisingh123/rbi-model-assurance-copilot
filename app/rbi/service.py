"""Adapter between technical assurance findings and the RBI requirement layer.

    technical findings
        -> assessment context
        -> deterministic applicability   (BEFORE retrieval)
        -> applicable requirements
        -> deterministic evidence matching
        -> RequirementStatus findings
        -> evidence records -> report

WHY A SEPARATE SERVICE RATHER THAN WIDENING THE SIX-RULE ENGINE
    The existing engine maps four technical findings onto six illustrative
    rules and answers PASS/WARNING/FAIL/PENDING. It works, it is wired into
    /compliance and the report, and ~1,600 tests depend on it. It is left
    exactly as it is.

    This layer answers a different question -- "which RBI obligations apply
    to this institution, and does the assurance run evidence them?" -- and
    needs statuses the old vocabulary does not have. The two run side by
    side and produce separately-identified findings; neither converts into
    the other.

APPLICABILITY RUNS BEFORE RETRIEVAL, NOT AFTER
    A document that does not bind this entity is filtered out before its
    requirements are ever fetched. Retrieving first and filtering later
    would mean an inapplicable obligation could be ranked, surfaced and
    cited on the strength of a good text match. Applicability is a gate.

NO LLM IS REACHABLE FROM THIS MODULE
    Every status here is produced by explicit comparison in
    ``app/rbi/requirements.py``. This module imports no LLM client and calls
    no generator. Narrative comes later, in the report, and only ever
    describes findings that were already decided here.

WHERE REQUIREMENTS COME FROM
    ``requirements_for_document`` -- which returns [] for every document in
    the current manifest, because none of the 19 sources has been obtained.
    The ``requirement_source`` parameter exists so integration tests can
    inject clearly-marked synthetic fixtures; its default is the real
    extractor, so production behaviour cannot silently become fixture
    behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from app.rbi.applicability import (
    APPLICABILITY_UNCLEAR as CTX_UNCLEAR,
    APPLIES,
    DOES_NOT_APPLY,
    ApplicabilityDecision,
    AssessmentContext,
    evaluate_applicability,
    rank_by_authority,
)
from app.rbi.manifest import RBIDocument, RBIManifest, load_manifest
from app.rbi.requirements import (
    APPLICABILITY_UNCLEAR,
    EVIDENCE_MISSING,
    NOT_ASSESSED,
    RegulatoryRequirement,
    RequirementFinding,
    assess_requirement,
    blocked_finding,
    requirements_for_document,
    summarise,
)

# The evidence_type under which requirement findings travel to the report.
# Distinct from every legacy type so the two compliance layers stay
# separable in a pooled evidence list.
EVIDENCE_TYPE_RBI_REQUIREMENT = "rbi_requirement_finding"

# The four outcomes a caller must be able to tell apart (constraint 5).
OUTCOME_NO_APPLICABLE_REQUIREMENT = "no_applicable_requirement"
OUTCOME_APPLICABILITY_UNCLEAR = "applicability_unclear"
OUTCOME_EVIDENCE_MISSING = "evidence_missing"
OUTCOME_ASSESSED = "assessed_with_evidence"

RequirementSource = Callable[[RBIDocument], Sequence[RegulatoryRequirement]]


@dataclass(frozen=True)
class RBIAssessment:
    """The result of assessing one assurance run against the RBI corpus."""

    findings: tuple[RequirementFinding, ...]
    applicability: tuple[ApplicabilityDecision, ...]
    context: AssessmentContext
    corpus_coverage: dict[str, Any]
    model_id: Optional[str] = None
    assurance_run_id: Optional[str] = None
    notes: tuple[str, ...] = field(default=())

    # -- the four distinguishable outcomes ---------------------------------

    def outcome(self) -> str:
        """One label for what this assessment actually established.

        Ordered from weakest to strongest claim, so a run that established
        nothing can never be described as one that did.
        """
        if not self.findings:
            return OUTCOME_NO_APPLICABLE_REQUIREMENT
        if any(f.status not in (NOT_ASSESSED, APPLICABILITY_UNCLEAR, EVIDENCE_MISSING)
               for f in self.findings):
            return OUTCOME_ASSESSED
        if any(f.status == EVIDENCE_MISSING for f in self.findings):
            return OUTCOME_EVIDENCE_MISSING
        if any(f.status == APPLICABILITY_UNCLEAR for f in self.findings):
            return OUTCOME_APPLICABILITY_UNCLEAR
        return OUTCOME_NO_APPLICABLE_REQUIREMENT

    def summary(self) -> dict[str, Any]:
        base = summarise(self.findings)
        base["outcome"] = self.outcome()
        base["documents_in_scope"] = len(
            [d for d in self.applicability if d.applicability != DOES_NOT_APPLY]
        )
        base["documents_excluded"] = len(
            [d for d in self.applicability if d.applicability == DOES_NOT_APPLY]
        )
        base["corpus_ready"] = self.corpus_coverage.get("corpus_ready_for_compliance", False)
        return base

    # -- report handoff ----------------------------------------------------

    def evidence_records(self) -> list[dict[str, Any]]:
        """Requirement findings as evidence records for the report layer.

        Each record is the finding's own dict plus an ``evidence_type`` and
        top-level ``model_id`` (which is what
        ``app/report/generate.py::_route_evidence_records`` buckets on).
        Structured provenance is carried verbatim under ``source`` -- there
        is no free-text locator, so a consumer reads
        ``source.document_id`` / ``source.section_id`` / ``source.page``
        rather than parsing a string.
        """
        records: list[dict[str, Any]] = []
        for finding in self.findings:
            record = finding.to_dict()
            record["evidence_type"] = EVIDENCE_TYPE_RBI_REQUIREMENT
            if finding.model_id is not None:
                record["model_id"] = finding.model_id
            if finding.assurance_run_id is not None:
                record["assurance_run_id"] = finding.assurance_run_id
            records.append(record)
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome(),
            "summary": self.summary(),
            "context": self.context.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "applicability": [d.to_dict() for d in self.applicability],
            "corpus_coverage": self.corpus_coverage,
            "model_id": self.model_id,
            "assurance_run_id": self.assurance_run_id,
            "notes": list(self.notes),
        }


def build_assessment_context(
    *,
    regulated_entity_type: Optional[str] = None,
    product: Optional[str] = None,
    model_use_case: Optional[str] = None,
    customer_population: Optional[str] = None,
    loan_type: Optional[str] = None,
    digital_vs_physical: Optional[str] = None,
    third_party_dependency: Optional[bool] = None,
    data_type: Optional[str] = None,
    geography: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> AssessmentContext:
    """Build the applicability context from DECLARED facts only.

    Every dimension is optional and nothing is inferred. In particular the
    context is NOT derived from the model under assurance: which entity type
    is being regulated, whether lending is digital, and whether a third
    party is involved are facts about the INSTITUTION, not about an
    sklearn pipeline. Guessing any of them from a model_id would invent a
    regulatory premise.

    An undeclared dimension makes dependent documents APPLICABILITY_UNCLEAR,
    which is the honest answer and is visible in the output.
    """
    return AssessmentContext(
        regulated_entity_type=regulated_entity_type,
        product=product,
        model_use_case=model_use_case,
        customer_population=customer_population,
        loan_type=loan_type,
        digital_vs_physical=digital_vs_physical,
        third_party_dependency=third_party_dependency,
        data_type=data_type,
        geography=geography,
        as_of_date=as_of_date,
    )


def assess_rbi_requirements(
    technical_findings: Any,
    *,
    context: AssessmentContext,
    manifest: Optional[RBIManifest] = None,
    requirement_source: RequirementSource = requirements_for_document,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
    include_unassessable: bool = True,
) -> RBIAssessment:
    """Assess an assurance run against the applicable RBI corpus.

    Parameters
    ----------
    technical_findings
        The assurance evidence bundle. Requirements address it by dotted
        path (``"explainability.available"``), resolved exactly -- never
        fuzzily, so evidence must be the thing the requirement asked for.
    context
        Who is being assessed. Built by ``build_assessment_context()``.
    requirement_source
        How a document's requirements are obtained. Defaults to the real
        extractor, which returns [] for every unresolved document. Tests
        inject synthetic fixtures here; production never does.
    include_unassessable
        When True (default) a document that yields no requirements produces
        an explicit NOT_ASSESSED finding naming its blocking reason. Set
        False only where the caller genuinely wants assessed requirements
        alone -- a silently skipped regulation is indistinguishable from a
        regulation that passed.
    """
    manifest = manifest if manifest is not None else load_manifest()

    decisions: list[ApplicabilityDecision] = []
    findings: list[RequirementFinding] = []
    notes: list[str] = []
    # Documents that ARE present and DO apply, yet yielded no requirement.
    # Counted separately from absent documents because the two say different
    # things about the corpus and call for different remedies.
    obtained_but_unmapped: list[str] = []

    # Authority order first, so that when several documents cover the same
    # ground the current one is assessed (and reported) ahead of a draft or
    # historical one.
    for document in rank_by_authority(manifest.all()):
        decision = evaluate_applicability(document, context)
        decisions.append(decision)

        # GATE: filtered out before its requirements are ever retrieved.
        if decision.applicability == DOES_NOT_APPLY:
            continue

        requirements = list(requirement_source(document))

        if not requirements:
            if document.is_citable():
                obtained_but_unmapped.append(document.document_id)
            if include_unassessable:
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
                    technical_findings,
                    model_id=model_id,
                    assurance_run_id=assurance_run_id,
                )
            )

    coverage = manifest.coverage_report()

    # TWO DIFFERENT EMPTY STATES, REPORTED DIFFERENTLY
    #
    # "we do not have the document" and "we have the document but have not
    # mapped its requirements" both produce zero assessed requirements, and
    # both are honest. They are not the same statement, and a reader who
    # cannot tell them apart cannot tell whether the remedy is to obtain a
    # source or to build extraction.
    #
    # Before this, only the first was reported, gated on the corpus having no
    # citable document at all. Obtaining the first PDFs silently switched the
    # note off while requirement extraction still did not exist -- so the
    # assessment stopped explaining itself at exactly the point the
    # explanation became least obvious. Neither branch changes a status:
    # every affected finding is NOT_ASSESSED either way.
    if not coverage.get("citable_documents"):
        notes.append(
            "No source document in the RBI corpus has been obtained, so no "
            "regulatory requirement could be extracted or cited. Every finding "
            "below records why it could not be assessed. This is not a "
            "statement of compliance."
        )
    elif obtained_but_unmapped:
        notes.append(
            f"{len(obtained_but_unmapped)} applicable source document(s) have "
            "been obtained, but no requirement has been extracted or mapped "
            "from them yet, so none could be assessed: "
            f"{', '.join(sorted(obtained_but_unmapped))}. Holding a document "
            "is not the same as having mapped its requirements. This is not a "
            "statement of compliance."
        )

    return RBIAssessment(
        findings=tuple(findings),
        applicability=tuple(decisions),
        context=context,
        corpus_coverage=coverage,
        model_id=model_id,
        assurance_run_id=assurance_run_id,
        notes=tuple(notes),
    )


def applicable_documents(
    context: AssessmentContext, manifest: Optional[RBIManifest] = None
) -> list[RBIDocument]:
    """Documents not excluded for this context, in authority order.

    Includes APPLICABILITY_UNCLEAR documents: an unresolved obligation must
    stay visible for a human to settle, not silently vanish.
    """
    manifest = manifest if manifest is not None else load_manifest()
    keep = [
        document
        for document in manifest.all()
        if evaluate_applicability(document, context).applicability != DOES_NOT_APPLY
    ]
    return rank_by_authority(keep)


__all__ = [
    "EVIDENCE_TYPE_RBI_REQUIREMENT",
    "OUTCOME_NO_APPLICABLE_REQUIREMENT",
    "OUTCOME_APPLICABILITY_UNCLEAR",
    "OUTCOME_EVIDENCE_MISSING",
    "OUTCOME_ASSESSED",
    "RBIAssessment",
    "RequirementSource",
    "build_assessment_context",
    "assess_rbi_requirements",
    "applicable_documents",
]
