"""Pydantic schemas for the RBI Model Assurance Copilot API.

These models strictly mirror the module interface specifications defined in
docs/module-interfaces.md field-for-field.
"""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

# Shared status type used across all assurance checks
Status = Literal["PASS", "WARNING", "FAIL", "PENDING"]


class LabelSemantics(BaseModel):
    """Semantic mapping of model target labels and probabilities."""
    zero: str = Field(default="GOOD - low credit risk", alias="0")
    one: str = Field(default="BAD - high credit risk / likely default", alias="1")
    positive_class: int = 1
    probabilities_represent: str = "P(class == 1) = P(BAD / high credit risk)"
    favorable_outcome_label: int = 0

    model_config = {"populate_by_name": True}


class ModelMetadata(BaseModel):
    """Metadata describing the trained credit model."""
    model_type: str
    version: str
    trained_on: str
    feature_names: list[str]
    label_semantics: Optional[LabelSemantics] = None


class ModelResult(BaseModel):
    """Output payload from the Model/Data module."""
    predictions: list[int]
    probabilities: list[float]
    instance_ids: list[str]
    feature_matrix: list[dict[str, Any]]
    model_metadata: ModelMetadata
    is_mock: bool


class PerInstanceContribution(BaseModel):
    """Feature contributions for an individual data record."""
    row_index: int
    contributions: dict[str, float]


class ExplainabilityResult(BaseModel):
    """Output payload from the Explainability module."""
    method: Literal["shap", "lime"]
    per_instance: list[PerInstanceContribution]
    global_importance: dict[str, float]
    is_mock: bool


class FairnessResult(BaseModel):
    """Output payload for fairness evaluation."""
    protected_attribute: str
    demographic_parity_diff: float
    disparate_impact_ratio: float
    status: Status
    is_mock: bool


class DriftResult(BaseModel):
    """Output payload for data/concept drift evaluation."""
    features_evaluated: list[str]
    psi: float
    ks_statistic: float
    status: Status
    is_mock: bool
    note: Optional[str] = None


class FairnessDriftResult(BaseModel):
    """Combined output payload for fairness and drift analysis."""
    fairness: FairnessResult
    drift: DriftResult
    note: Optional[str] = None


class ComplianceFinding(BaseModel):
    """Individual compliance evaluation finding against an RBI requirement."""
    rule_id: str
    rule_description: str
    technical_finding_ref: str
    status: Status
    evidence_chunks: list[str] = Field(default_factory=list)


class ComplianceResult(BaseModel):
    """Output payload from the Compliance/Rule Engine module."""
    findings: list[ComplianceFinding]
    is_mock: bool


class AssuranceResult(BaseModel):
    """Full aggregated assurance result combining all four evaluation domains."""
    model: ModelResult
    explainability: ExplainabilityResult
    fairness_drift: FairnessDriftResult
    compliance: ComplianceResult
    note: str


# =============================================================================
# Phase 3 LLM Reporting Schemas (PROVISIONAL — pending Nidhi + team sign-off)
# =============================================================================

class TechnicalFinding(BaseModel):
    """Layer 1: Deterministic analytical result from Python evaluation modules."""
    ref: str
    value: Any
    status: str
    source_module: str
    provenance: Literal["observed", "mock", "synthetic_fixture"]


class Citation(BaseModel):
    """Source reference for retrieved regulatory evidence."""
    source: str
    locator: str
    quote: str
    provenance: Literal["verified", "illustrative"]


class RetrievedEvidence(BaseModel):
    """Layer 2: Grounded regulatory evidence from RAG vector retrieval."""
    evidence_status: Literal["RETRIEVED", "NOT_FOUND", "NOT_ATTEMPTED"]
    citations: list[Citation] = Field(default_factory=list)


class LLMInterpretation(BaseModel):
    """Layer 3: Synthesized natural language interpretation strictly grounded in evidence."""
    text: str
    grounded_in: list[str]
    regulatory_basis: Literal["cited_evidence", "illustrative_rule_only", "none"]
    is_mock: bool


class ReportSection(BaseModel):
    """Three-layer report section preserving distinct evidence boundaries (never collapsed)."""
    heading: str
    technical_finding: TechnicalFinding
    retrieved_evidence: RetrievedEvidence
    llm_interpretation: LLMInterpretation


class EvidenceCoverage(BaseModel):
    """Summary of evidence retrieval coverage across report sections."""
    retrieved: int
    not_found: int
    total: int


class ReportResult(BaseModel):
    """PROVISIONAL: Output payload for LLM-assisted model assurance report."""
    report_id: str
    generated_at: str
    model_version: str
    sections: list[ReportSection]
    disclaimers: list[str]
    evidence_coverage: EvidenceCoverage
    is_mock: bool
