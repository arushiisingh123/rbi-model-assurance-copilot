"""Deterministic applicability engine (owner: Nidhi / Manas, corpus phase).

Decides whether one RBI source document applies to one assessment context.

THE LLM NEVER RUNS THIS. Applicability is the decision that determines
whether a regulation binds an institution at all, and it is decided here by
explicit metadata comparison -- ``applies_to``, ``excluded``,
``requires_conditions``, effective dates and supersession. A language model
asked "does this circular apply to a base-layer NBFC?" will produce a fluent
answer whether or not it knows, and that answer would silently become a
regulatory conclusion.

THREE OUTCOMES, AND THE MIDDLE ONE IS LOAD-BEARING

    APPLIES               the document's own metadata says so
    DOES_NOT_APPLY        the document's own metadata excludes this context
    APPLICABILITY_UNCLEAR the metadata does not settle it

UNCLEAR IS NEVER SILENTLY RESOLVED. It is not rounded down to
DOES_NOT_APPLY (which would drop a real obligation) and never up to APPLIES
(which would assert an obligation that may not exist). Every document in the
current manifest carries ``applicability_verified: false``, so unverified
metadata yields UNCLEAR even when ``applies_to`` matches -- "we transcribed
this" is not "we confirmed this against the source".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.rbi.manifest import RBIDocument

APPLIES = "APPLIES"
DOES_NOT_APPLY = "DOES_NOT_APPLY"
APPLICABILITY_UNCLEAR = "APPLICABILITY_UNCLEAR"

VALID_APPLICABILITY = (APPLIES, DOES_NOT_APPLY, APPLICABILITY_UNCLEAR)

# The dimensions an assessment context may declare. A dimension the caller
# omits is simply not tested -- absence of a declaration is not a licence to
# assume a value.
CONTEXT_DIMENSIONS = (
    "regulated_entity_type",
    "product",
    "model_use_case",
    "customer_population",
    "loan_type",
    "digital_vs_physical",
    "third_party_dependency",
    "data_type",
    "geography",
    "as_of_date",
)


@dataclass(frozen=True)
class AssessmentContext:
    """Who is being assessed, and on what.

    Every field is optional. An undeclared dimension makes any document whose
    applicability depends on it UNCLEAR rather than assumed either way.
    """

    regulated_entity_type: Optional[str] = None
    product: Optional[str] = None
    model_use_case: Optional[str] = None
    customer_population: Optional[str] = None
    loan_type: Optional[str] = None
    digital_vs_physical: Optional[str] = None
    third_party_dependency: Optional[bool] = None
    data_type: Optional[str] = None
    geography: Optional[str] = None
    as_of_date: Optional[str] = None

    def get(self, dimension: str) -> Any:
        return getattr(self, dimension, None)

    def to_dict(self) -> dict[str, Any]:
        return {d: self.get(d) for d in CONTEXT_DIMENSIONS}


@dataclass(frozen=True)
class ApplicabilityDecision:
    """One document's applicability to one context, with its reasoning.

    ``reasons`` is a plain audit trail: a reviewer must be able to see WHY a
    regulation was applied or skipped without re-running anything.
    """

    document_id: str
    applicability: str
    reasons: tuple[str, ...]
    verified: bool
    regulatory_status: str
    is_operative: bool
    is_citable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "applicability": self.applicability,
            "reasons": list(self.reasons),
            "applicability_verified": self.verified,
            "regulatory_status": self.regulatory_status,
            "is_operative": self.is_operative,
            "is_citable": self.is_citable,
        }


def _normalise(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def _matches(candidate: Any, allowed: Any) -> bool:
    return _normalise(candidate) == _normalise(allowed)


def evaluate_applicability(
    document: RBIDocument, context: AssessmentContext
) -> ApplicabilityDecision:
    """Decide whether ``document`` applies to ``context``. Pure metadata logic.

    Order matters. Exclusions are checked FIRST and are absolute: a document
    that explicitly excludes an entity type does not apply to it, whatever
    else matches. Only then is inclusion considered, and only then the
    document's extra conditions.
    """
    reasons: list[str] = []
    entity = context.regulated_entity_type

    def decide(outcome: str) -> ApplicabilityDecision:
        return ApplicabilityDecision(
            document_id=document.document_id,
            applicability=outcome,
            reasons=tuple(reasons),
            verified=document.applicability_verified,
            regulatory_status=document.regulatory_status,
            is_operative=document.is_operative(),
            is_citable=document.is_citable(),
        )

    # 1. EXPLICIT EXCLUSION -- absolute, and checked before anything else.
    if entity is not None:
        for excluded in document.excluded:
            if _matches(entity, excluded):
                reasons.append(
                    f"'{entity}' is in the document's explicit exclusion list "
                    f"({list(document.excluded)}), so the document does not apply."
                )
                return decide(DOES_NOT_APPLY)

    # 2. ENTITY TYPE.
    if entity is None:
        reasons.append(
            "No regulated_entity_type was declared, so entity applicability "
            "cannot be determined."
        )
        return decide(APPLICABILITY_UNCLEAR)

    if not document.applies_to:
        reasons.append("The document declares no applies_to list.")
        return decide(APPLICABILITY_UNCLEAR)

    if not any(_matches(entity, allowed) for allowed in document.applies_to):
        reasons.append(
            f"'{entity}' is not in the document's applies_to list "
            f"({list(document.applies_to)})."
        )
        return decide(DOES_NOT_APPLY)

    reasons.append(f"'{entity}' is in the document's applies_to list.")

    # 3. DOCUMENT-SPECIFIC CONDITIONS (e.g. digital lending requires
    #    digital_vs_physical == 'digital'; DLG additionally requires a
    #    third-party dependency). A condition the context does not declare
    #    makes the decision UNCLEAR -- it is not assumed satisfied.
    for dimension, allowed_values in document.requires_conditions.items():
        actual = context.get(dimension)
        if actual is None:
            reasons.append(
                f"The document is conditional on '{dimension}' (one of "
                f"{allowed_values}), which this context does not declare."
            )
            return decide(APPLICABILITY_UNCLEAR)
        if not any(_matches(actual, candidate) for candidate in allowed_values):
            reasons.append(
                f"The document requires {dimension} in {allowed_values}; this "
                f"context declares {actual!r}."
            )
            return decide(DOES_NOT_APPLY)
        reasons.append(f"Condition satisfied: {dimension} == {actual!r}.")

    # 4. EFFECTIVE DATE. A document not yet in force does not bind.
    if document.effective_date and context.as_of_date:
        if str(context.as_of_date) < str(document.effective_date):
            reasons.append(
                f"The document takes effect on {document.effective_date}, after "
                f"the assessment date {context.as_of_date}."
            )
            return decide(DOES_NOT_APPLY)
        reasons.append(
            f"In force as of {context.as_of_date} (effective {document.effective_date})."
        )

    # 5. VERIFICATION. Everything above matched -- but matching TRANSCRIBED
    #    metadata is not the same as confirming it against the source. An
    #    unverified document stops at UNCLEAR.
    if not document.applicability_verified:
        reasons.append(
            "Applicability metadata for this document is not verified against "
            "the source (applicability_verified is false), so applicability "
            "cannot be asserted."
        )
        return decide(APPLICABILITY_UNCLEAR)

    return decide(APPLIES)


def filter_applicable(
    documents: list[RBIDocument],
    context: AssessmentContext,
    *,
    include_unclear: bool = True,
) -> list[ApplicabilityDecision]:
    """Applicability decisions for many documents.

    ``include_unclear`` defaults to True: an unclear document is surfaced for
    human resolution, because dropping it would make an unresolved obligation
    invisible. Callers that genuinely want only settled applications pass
    False.
    """
    decisions = [evaluate_applicability(document, context) for document in documents]
    if include_unclear:
        return [d for d in decisions if d.applicability != DOES_NOT_APPLY]
    return [d for d in decisions if d.applicability == APPLIES]


def rank_by_authority(documents: list[RBIDocument]) -> list[RBIDocument]:
    """Order documents so a CURRENT source always outranks a stale one.

    Guards the corpus rule that historical or draft text must never appear to
    be the operative requirement. Ordering: operative-and-current first, then
    tier, then documents that something else supersedes last.
    """

    def key(document: RBIDocument) -> tuple:
        superseded = bool(document.superseded_by)
        tier_rank = {"tier_0": 0, "tier_1": 1, "tier_2": 2}.get(document.priority, 3)
        return (
            0 if document.is_operative() else 1,
            1 if superseded else 0,
            1 if document.is_provisional() else 0,
            tier_rank,
            document.document_id,
        )

    return sorted(documents, key=key)


__all__ = [
    "APPLIES",
    "DOES_NOT_APPLY",
    "APPLICABILITY_UNCLEAR",
    "VALID_APPLICABILITY",
    "CONTEXT_DIMENSIONS",
    "AssessmentContext",
    "ApplicabilityDecision",
    "evaluate_applicability",
    "filter_applicable",
    "rank_by_authority",
]
