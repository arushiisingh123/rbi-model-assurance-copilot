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
