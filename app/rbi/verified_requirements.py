"""A small, hand-verified RBI requirement register for the demonstration corpus.

WHY THIS MODULE EXISTS SEPARATELY FROM ``requirements_for_document()``
    ``app/rbi/requirements.py::requirements_for_document()`` extracts
    requirements from an INGESTED document and returns ``[]`` for anything not
    downloaded. That contract is correct and is not being weakened here: no
    document in ``corpus_manifest.json`` has been downloaded, so that function
    still returns nothing, and ``RBIDocument.is_citable()`` still gates quoting
    on ``source_status == "downloaded"``.

    The requirements below come from a DIFFERENT and weaker-but-honest
    provenance class: each clause was read from the official RBI web page for
    that instrument and transcribed with its printed clause number. That is
    verifiable — a reviewer can open ``source_url`` and find the clause — but it
    is not machine ingestion, and it is deliberately not represented as such.

    Every record therefore carries ``verification_method`` and
    ``verified_on`` so the provenance of the citation is visible rather than
    implied.

WHAT IS AND IS NOT HERE
    Three instruments produced requirements. A fourth (the Scale Based
    Regulation Directions, 2023) is identity-verified but produced NONE,
    because its clause text is served only as a PDF behind a bot check and
    could not be read. Recording zero requirements for it is the correct
    outcome: inventing NBFC layer definitions from general knowledge would be
    exactly the failure this project exists to prevent.

    The corpus is explicitly partial. It is a demonstration that verified
    requirements can be applied, bound to evidence and traced — not a claim of
    RBI coverage.

ASSESSMENT MODES, AND WHY MOST ARE ATTESTATION
    A model-assurance platform can evidence what a model does. It cannot
    evidence that a board constituted a committee, that a contract contains an
    audit clause, or that a lender obtained a borrower's income details before
    lending. Those are organisational facts about the regulated entity.

    Marking them ``attestation`` keeps them visible and unassessed rather than
    letting model analytics stand in for organisational evidence. A requirement
    in this state is NEVER ``PASS`` and never ``FAIL`` — absence of evidence is
    not a finding either way.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from app.rbi.requirements import (
    EVIDENCE_MISSING,
    NOT_ASSESSED,
    EvidenceRequirement,
    RegulatoryRequirement,
    SourceLocation,
)

__all__ = [
    "ASSESSMENT_MODES",
    "ATTESTATION",
    "AUTOMATED",
    "INFORMATIONAL",
    "INSTRUMENT_BINDING",
    "NOT_APPLICABLE",
    "EntityProfile",
    "VERIFIED_REQUIREMENTS",
    "VERIFIED_SOURCES",
    "VerifiedRequirement",
    "VerifiedSource",
    "applicable_requirements",
    "assess_verified_requirements",
    "requirement_by_id",
    "source_by_id",
]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Instrument character. A binding Direction and an advisory report may never be
# presented as the same kind of obligation.
INSTRUMENT_BINDING = "binding"
INSTRUMENT_ADVISORY = "advisory"

# How a requirement could be assessed AT ALL, independent of any result.
AUTOMATED = "automated"          # platform evidence can answer it
ATTESTATION = "attestation"      # only the regulated entity can evidence it
INFORMATIONAL = "informational"  # context; no pass/fail concept

ASSESSMENT_MODES = (AUTOMATED, ATTESTATION, INFORMATIONAL)

# Applicability outcome used when a requirement's conditions are not met.
# Distinct from "not assessed": we know it does not apply.
NOT_APPLICABLE = "NOT_APPLICABLE"

# Every clause below was read from the official RBI page on this date.
_VERIFIED_ON = "2026-09-19"
_VERIFICATION_METHOD = (
    "Clause read from the official RBI web page for this instrument and "
    "transcribed with its printed clause number. Not machine-ingested; the "
    "source document is not stored in this repository."
)


# ---------------------------------------------------------------------------
# Entity profile -- declared, never inferred
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityProfile:
    """Who is being assessed.

    Every field is optional and defaults to "not declared". An undeclared
    dimension makes any requirement that depends on it UNCLEAR rather than
    assumed either way -- the same rule
    ``app/rbi/applicability.py::evaluate_applicability`` already applies.

    These values are DECLARED BY THE CALLER. None of them is inferred from the
    model, its features, or the customer data: a dataset cannot tell you which
    NBFC layer an entity occupies, and guessing would manufacture the
    applicability decision the whole layer exists to make honestly.
    """

    entity_type: Optional[str] = None            # e.g. "NBFC"
    nbfc_layer: Optional[str] = None             # e.g. "Middle", "Upper"
    digital_lending: Optional[bool] = None
    microfinance: Optional[bool] = None
    uses_external_model_vendor: Optional[bool] = None

    def get(self, dimension: str) -> Any:
        return getattr(self, dimension, None)


# ---------------------------------------------------------------------------
# Verified sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedSource:
    """One RBI instrument whose identity was verified against rbi.org.in."""

    document_id: str
    title: str
    rbi_reference: str
    issued_date: str
    source_url: str
    instrument_type: str
    regulatory_status: str
    verified_on: str = _VERIFIED_ON
    verification_method: str = _VERIFICATION_METHOD
    supersedes: tuple[str, ...] = ()
    clause_text_available: bool = True
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "rbi_reference": self.rbi_reference,
            "issued_date": self.issued_date,
            "source_url": self.source_url,
            "instrument_type": self.instrument_type,
            "regulatory_status": self.regulatory_status,
            "verified_on": self.verified_on,
            "verification_method": self.verification_method,
            "supersedes": list(self.supersedes),
            "clause_text_available": self.clause_text_available,
            "notes": self.notes,
        }


VERIFIED_SOURCES: tuple[VerifiedSource, ...] = (
    VerifiedSource(
        document_id="RBI-DIGITAL-LENDING-2025",
        title="Reserve Bank of India (Digital Lending) Directions, 2025",
        rbi_reference="RBI/2025-26/36 DOR.STR.REC.19/21.07.001/2025-26",
        issued_date="2025-05-08",
        source_url="https://rbi.org.in/Scripts/NotificationUser.aspx?Id=12848",
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        notes=(
            "The current consolidated digital-lending source. The 2022 "
            "Guidelines on Digital Lending are a different, earlier instrument "
            "and are not used as the current source."
        ),
    ),
    VerifiedSource(
        document_id="RBI-IT-OUTSOURCING-2023",
        title=(
            "Reserve Bank of India (Outsourcing of Information Technology "
            "Services) Directions, 2023"
        ),
        rbi_reference="RBI/2023-24/102",
        issued_date="2023-10-01",  # effective date stated on the RBI page
        source_url="https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=12486",
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
    ),
    VerifiedSource(
        document_id="RBI-FRAUD-NBFC-2024",
        title="Reserve Bank of India (Fraud Risk Management in NBFCs) Directions, 2024",
        rbi_reference="RBI/DOS/2024-25/120 DOS.CO.FMG.SEC.No.7/23.04.001/2024-25",
        issued_date="2024-07-15",
        source_url="https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12704",
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        supersedes=("RBI-MONITORING-FRAUDS-NBFC-2016",),
        notes=(
            "Supersedes the Master Direction - Monitoring of Frauds in NBFCs "
            "(Reserve Bank) Directions, 2016 dated September 29, 2016, as "
            "stated on the RBI page for this instrument."
        ),
    ),
    # Identity verified; clause text NOT obtainable. Produces no requirements.
    VerifiedSource(
        document_id="RBI-SBR-NBFC-2023",
        title=(
            "Master Direction - Reserve Bank of India (Non-Banking Financial "
            "Company - Scale Based Regulation) Directions, 2023"
        ),
        rbi_reference="RBI/DoR/2023-24/106",
        issued_date="2023-10-19",
        source_url="https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12550",
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        clause_text_available=False,
        notes=(
            "Identity verified against the official RBI page. The clause text "
            "is served only as a PDF behind a bot check and could not be read, "
            "so NO requirement is encoded from this instrument. The NBFC layer "
            "vocabulary used for applicability below is taken from the Fraud "
            "Risk Management Directions, 2024, which state it in their own "
            "clause text."
        ),
    ),
    # Superseded source, retained ONLY to demonstrate that stale material is
    # distinguishable. It carries no requirements and must never be cited as
    # current regulation.
    VerifiedSource(
        document_id="RBI-MONITORING-FRAUDS-NBFC-2016",
        title=(
            "Master Direction - Monitoring of Frauds in NBFCs (Reserve Bank) "
            "Directions, 2016"
        ),
        rbi_reference="Master Direction dated September 29, 2016",
        issued_date="2016-09-29",
        source_url="https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12704",
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="superseded",
        clause_text_available=False,
        notes=(
            "Superseded by RBI-FRAUD-NBFC-2024. Retained as a stale-source "
            "demonstration only; it carries no requirements and must never "
            "outrank the current instrument. The supersession is stated on the "
            "RBI page for the 2024 Directions."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Verified requirements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedRequirement:
    """One verified clause, its applicability conditions and how it can be assessed.

    Composes the existing ``RegulatoryRequirement`` rather than replacing it,
    so the project keeps one requirement model. The added fields are the ones
    a verified-but-not-ingested citation needs: the clause's own source URL,
    the instrument's character, how it can be assessed at all, and the explicit
    conditions under which it applies.
    """

    requirement: RegulatoryRequirement
    source_url: str
    instrument_type: str
    regulatory_status: str
    assessment_mode: str
    clause: str
    quote: Optional[str] = None
    applies_when: dict[str, Any] = field(default_factory=dict)
    limitation: Optional[str] = None
    verified_on: str = _VERIFIED_ON

    @property
    def requirement_id(self) -> str:
        return self.requirement.requirement_id

    @property
    def document_id(self) -> str:
        return self.requirement.document_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.requirement.to_dict(),
            "source_url": self.source_url,
            "instrument_type": self.instrument_type,
            "regulatory_status": self.regulatory_status,
            "assessment_mode": self.assessment_mode,
            "clause": self.clause,
            "quote": self.quote,
            "applies_when": dict(self.applies_when),
            "limitation": self.limitation,
            "verified_on": self.verified_on,
        }


def _requirement(
    requirement_id: str,
    document_id: str,
    document_title: str,
    clause: str,
    summary: str,
    *,
    domain: tuple[str, ...],
    evidence: tuple[EvidenceRequirement, ...] = (),
) -> RegulatoryRequirement:
    return RegulatoryRequirement(
        requirement_id=requirement_id,
        document_id=document_id,
        requirement=summary,
        source=SourceLocation(
            document_id=document_id,
            document_title=document_title,
            section_id=clause,
        ),
        evidence_required=evidence,
        domain=domain,
    )


_DL_TITLE = "Reserve Bank of India (Digital Lending) Directions, 2025"
_ITO_TITLE = (
    "Reserve Bank of India (Outsourcing of Information Technology Services) "
    "Directions, 2023"
)
_FRAUD_TITLE = (
    "Reserve Bank of India (Fraud Risk Management in NBFCs) Directions, 2024"
)

_DL_URL = "https://rbi.org.in/Scripts/NotificationUser.aspx?Id=12848"
_ITO_URL = "https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=12486"
_FRAUD_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12704"


VERIFIED_REQUIREMENTS: tuple[VerifiedRequirement, ...] = (
    # -- Digital Lending Directions, 2025 --------------------------------
    VerifiedRequirement(
        requirement=_requirement(
            "RBI-DL-2025-5.ii",
            "RBI-DIGITAL-LENDING-2025",
            _DL_TITLE,
            "5.ii",
            "Enhanced due diligence must be conducted before entering into an "
            "agreement with a Lending Service Provider for digital lending.",
            domain=("digital_lending", "vendor_governance"),
        ),
        source_url=_DL_URL,
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        assessment_mode=ATTESTATION,
        clause="5.ii",
        quote=(
            "RE shall conduct enhanced due diligence before they enter into an "
            "agreement with a LSP for digital lending, taking into account "
            "LSP's technical capabilities, robustness of data privacy policies "
            "and storage systems, fairness in conduct with borrowers, past "
            "records of conduct and ability to comply with all applicable "
            "regulations and statutes."
        ),
        applies_when={"digital_lending": True},
        limitation=(
            "Due-diligence records are organisational evidence held by the "
            "regulated entity. The platform cannot observe them, and model "
            "analytics do not substitute for them."
        ),
    ),
    VerifiedRequirement(
        requirement=_requirement(
            "RBI-DL-2025-5.iii",
            "RBI-DIGITAL-LENDING-2025",
            _DL_TITLE,
            "5.iii",
            "The conduct of the Lending Service Provider must be reviewed "
            "periodically against the contractual agreement.",
            domain=("digital_lending", "vendor_governance"),
        ),
        source_url=_DL_URL,
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        assessment_mode=ATTESTATION,
        clause="5.iii",
        quote=(
            "RE shall carry out periodic review of the conduct of the LSP "
            "vis-a-vis the terms of the contractual agreement and shall take "
            "appropriate action in the event of any deviation therefrom."
        ),
        applies_when={"digital_lending": True},
        limitation=(
            "Periodic review is a governance activity evidenced by the entity's "
            "own review records."
        ),
    ),
    VerifiedRequirement(
        requirement=_requirement(
            "RBI-DL-2025-7.i",
            "RBI-DIGITAL-LENDING-2025",
            _DL_TITLE,
            "7.i",
            "Information on the borrower's economic profile must be obtained to "
            "assess creditworthiness before extending a loan, including at a "
            "minimum age, occupation and income details.",
            domain=("digital_lending", "credit_assessment"),
        ),
        source_url=_DL_URL,
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        assessment_mode=ATTESTATION,
        clause="7.i",
        quote=(
            "RE shall obtain the necessary information relating to economic "
            "profile of the borrower with a view to assessing the borrower's "
            "creditworthiness before extending any loan, including, at a "
            "minimum, age, occupation and income details."
        ),
        applies_when={"digital_lending": True},
        limitation=(
            "DELIBERATELY NOT AUTOMATED. The platform can list a model's input "
            "features, and a feature inventory is useful context here -- but a "
            "feature list is not proof that the lender obtained the borrower's "
            "economic profile before lending. Treating the two as equivalent "
            "would convert a model artefact into a compliance claim."
        ),
    ),
    # -- Outsourcing of IT Services Directions, 2023 ----------------------
    VerifiedRequirement(
        requirement=_requirement(
            "RBI-ITO-2023-4.1",
            "RBI-IT-OUTSOURCING-2023",
            _ITO_TITLE,
            "4.1",
            "Outsourcing does not diminish the regulated entity's obligations; "
            "its Board and Senior Management remain ultimately responsible for "
            "the outsourced activity.",
            domain=("it_outsourcing", "vendor_governance", "accountability"),
        ),
        source_url=_ITO_URL,
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        assessment_mode=ATTESTATION,
        clause="4.1",
        quote=(
            "Outsourcing of any activity shall not diminish RE's obligations as "
            "also of its Board and Senior Management, who shall be ultimately "
            "responsible for the outsourced activity."
        ),
        applies_when={"uses_external_model_vendor": True},
        limitation=(
            "This is the clause that makes the external-model story a "
            "regulatory one: hosting a model elsewhere does not move "
            "responsibility. The platform can evidence that an externally "
            "served model was assured; it cannot evidence board accountability."
        ),
    ),
    VerifiedRequirement(
        requirement=_requirement(
            "RBI-ITO-2023-13.1",
            "RBI-IT-OUTSOURCING-2023",
            _ITO_TITLE,
            "13.1",
            "Appropriate due diligence must be performed to assess the "
            "capability of the service provider when considering or renewing an "
            "IT outsourcing arrangement.",
            domain=("it_outsourcing", "vendor_governance"),
        ),
        source_url=_ITO_URL,
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        assessment_mode=ATTESTATION,
        clause="13.1",
        quote=(
            "In considering or renewing an Outsourcing of IT Services "
            "arrangement, appropriate due diligence shall be performed to "
            "assess the capability of the service provider..."
        ),
        applies_when={"uses_external_model_vendor": True},
        limitation="Due-diligence records are held by the regulated entity.",
    ),
    VerifiedRequirement(
        requirement=_requirement(
            "RBI-ITO-2023-16.13",
            "RBI-IT-OUTSOURCING-2023",
            _ITO_TITLE,
            "16.13",
            "The outsourcing agreement must grant the regulated entity the right "
            "to audit the service provider, including its sub-contractors.",
            domain=("it_outsourcing", "audit"),
        ),
        source_url=_ITO_URL,
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        assessment_mode=ATTESTATION,
        clause="16.13",
        quote=(
            "right to conduct audit of the service provider (including its "
            "sub-contractors) by the RE, whether by its internal or external "
            "auditors, or by agents appointed to act on its behalf..."
        ),
        applies_when={"uses_external_model_vendor": True},
        limitation=(
            "A contractual term. The platform cannot read contracts, and an "
            "assurance run against a vendor-hosted model does not evidence one."
        ),
    ),
    # -- Fraud Risk Management in NBFCs Directions, 2024 ------------------
    VerifiedRequirement(
        requirement=_requirement(
            "RBI-FRAUD-2024-2.3",
            "RBI-FRAUD-NBFC-2024",
            _FRAUD_TITLE,
            "Chapter II, 2.3",
            "Applicable NBFCs must constitute a Special Committee of the Board "
            "for Monitoring and Follow-up of cases of Frauds (SCBMF) with a "
            "minimum of three members.",
            domain=("fraud_risk", "governance"),
        ),
        source_url=_FRAUD_URL,
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        assessment_mode=ATTESTATION,
        clause="Chapter II, 2.3",
        quote=(
            "Applicable NBFCs shall constitute a Committee of the Board to be "
            "known as 'Special Committee of the Board for Monitoring and "
            "Follow-up of cases of Frauds' (SCBMF) with a minimum of three "
            "members"
        ),
        applies_when={"entity_type": "NBFC"},
        limitation=(
            "Board composition is organisational evidence. No model analytic "
            "bears on it."
        ),
    ),
    VerifiedRequirement(
        requirement=_requirement(
            "RBI-FRAUD-2024-3.1.3",
            "RBI-FRAUD-NBFC-2024",
            _FRAUD_TITLE,
            "Chapter III, 3.1.3",
            "Upper Layer and Middle Layer NBFCs must identify appropriate early "
            "warning indicators for monitoring credit facilities, loan accounts "
            "and other financial transactions.",
            domain=("fraud_risk", "monitoring"),
        ),
        source_url=_FRAUD_URL,
        instrument_type=INSTRUMENT_BINDING,
        regulatory_status="current",
        assessment_mode=ATTESTATION,
        clause="Chapter III, 3.1.3",
        quote=(
            "NBFCs - UL & ML shall identify appropriate early warning indicators "
            "for monitoring credit facilities / loan accounts and other "
            "financial transactions"
        ),
        # The applicability demonstration: this clause names the layers it
        # binds, so a Base Layer NBFC is genuinely outside its scope.
        applies_when={"entity_type": "NBFC", "nbfc_layer": ["Upper", "Middle"]},
        limitation=(
            "The platform's monitoring channels measure feature drift, "
            "prediction drift and fairness. Those are model-behaviour "
            "indicators, NOT the fraud early-warning indicators this clause "
            "requires, and must not be presented as satisfying it. Fraud "
            "analytics are an extension, not an implemented capability."
        ),
    ),
)


def source_by_id(document_id: str) -> Optional[VerifiedSource]:
    for source in VERIFIED_SOURCES:
        if source.document_id == document_id:
            return source
    return None


def requirement_by_id(requirement_id: str) -> Optional[VerifiedRequirement]:
    for requirement in VERIFIED_REQUIREMENTS:
        if requirement.requirement_id == requirement_id:
            return requirement
    return None


# ---------------------------------------------------------------------------
# Applicability -- deterministic, declared-only
# ---------------------------------------------------------------------------


def _condition_met(declared: Any, expected: Any) -> Optional[bool]:
    """Whether ``declared`` satisfies ``expected``. None means "undeclared".

    Returning None rather than False for an undeclared dimension is the whole
    point: "we were not told" and "it does not apply" are different answers,
    and collapsing them would silently drop requirements that might well apply.
    """
    if declared is None:
        return None
    if isinstance(expected, list):
        return declared in expected
    return declared == expected


def applicable_requirements(
    profile: EntityProfile,
) -> list[tuple[VerifiedRequirement, str, list[str]]]:
    """Every verified requirement paired with its applicability decision.

    Returns ``(requirement, applicability, reasons)`` for ALL requirements --
    never a filtered list. A requirement that does not apply is reported as
    ``NOT_APPLICABLE`` with the reason, because a silently dropped regulation
    is indistinguishable from one that passed.

    Applicability values reuse the project's existing vocabulary: ``APPLIES``,
    ``NOT_APPLICABLE`` and ``APPLICABILITY_UNCLEAR``.
    """
    from app.rbi.applicability import APPLICABILITY_UNCLEAR, APPLIES

    results: list[tuple[VerifiedRequirement, str, list[str]]] = []
    for requirement in VERIFIED_REQUIREMENTS:
        reasons: list[str] = []
        undeclared: list[str] = []
        failed: list[str] = []

        for dimension, expected in requirement.applies_when.items():
            met = _condition_met(profile.get(dimension), expected)
            if met is None:
                undeclared.append(dimension)
            elif not met:
                failed.append(
                    f"{dimension}={profile.get(dimension)!r} does not match "
                    f"{expected!r}"
                )

        if failed:
            applicability = NOT_APPLICABLE
            reasons = failed
        elif undeclared:
            applicability = APPLICABILITY_UNCLEAR
            reasons = [
                f"{dimension} was not declared, so applicability cannot be "
                "determined and is not assumed."
                for dimension in undeclared
            ]
        else:
            applicability = APPLIES
            reasons = [
                f"{dimension}={profile.get(dimension)!r} matches {expected!r}"
                for dimension, expected in requirement.applies_when.items()
            ]

        results.append((requirement, applicability, reasons))
    return results


# ---------------------------------------------------------------------------
# Assessment -- deterministic; never PASS on absent evidence
# ---------------------------------------------------------------------------


def assess_verified_requirements(
    profile: EntityProfile,
    *,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Assess every verified requirement against an entity profile.

    Produces flat, self-describing finding records in the project's existing
    evidence style: each carries its own type, the requirement's full source
    identity, the applicability decision and the reason for its status.

    **No requirement in this register can return PASS.** Every one is
    ``attestation`` mode, meaning only the regulated entity can evidence it —
    so an applicable requirement is reported ``EVIDENCE_MISSING`` and a
    non-applicable one ``NOT_ASSESSED`` with its applicability reason. Model
    analytics never stand in for organisational evidence.

    Args:
        profile: The declared entity profile. Nothing is inferred from it.
        model_id / model_version / assurance_run_id: Optional identity of the
            assurance run these findings belong to. Included only when
            supplied, matching the convention the other evidence producers use.

    Returns:
        One record per verified requirement, in register order.
    """
    from app.rbi.applicability import APPLIES

    identity = {
        k: v
        for k, v in (
            ("model_id", model_id),
            ("model_version", model_version),
            ("assurance_run_id", assurance_run_id),
        )
        if v is not None
    }

    findings: list[dict[str, Any]] = []
    for requirement, applicability, reasons in applicable_requirements(profile):
        if applicability == APPLIES:
            status = EVIDENCE_MISSING
            reason = (
                "Applicable, but this requirement can only be evidenced by the "
                "regulated entity (assessment_mode=attestation). The platform "
                "holds no evidence bearing on it, so no compliance conclusion "
                "is drawn."
            )
        else:
            status = NOT_ASSESSED
            reason = "; ".join(reasons) or "Applicability could not be established."

        findings.append(
            {
                "evidence_type": "rbi_verified_requirement",
                "requirement_id": requirement.requirement_id,
                "document_id": requirement.document_id,
                "document_title": requirement.requirement.source.document_title,
                "clause": requirement.clause,
                "requirement": requirement.requirement.requirement,
                "quote": requirement.quote,
                "source_url": requirement.source_url,
                "instrument_type": requirement.instrument_type,
                "regulatory_status": requirement.regulatory_status,
                "assessment_mode": requirement.assessment_mode,
                "applicability": applicability,
                "applicability_reasons": list(reasons),
                "status": status,
                "reason": reason,
                "limitation": requirement.limitation,
                "verified_on": requirement.verified_on,
                "is_mock": False,
                **identity,
            }
        )
    return findings
