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


class ModelMetrics(BaseModel):
    """Held-out test set evaluation metrics for the trained credit model."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    roc_auc_status: Literal["computed", "unavailable_no_probabilities"] = "computed"
    n_test_samples: int
    is_mock: bool


class ModelResult(BaseModel):
    """Output payload from the Model/Data module."""
    predictions: list[int]
    probabilities: list[float]
    instance_ids: list[str]
    feature_matrix: list[dict[str, Any]]
    model_metadata: ModelMetadata
    is_mock: bool
    model_metrics: Optional[ModelMetrics] = None


class PerInstanceContribution(BaseModel):
    """Feature contributions for an individual data record."""
    row_index: int
    contributions: dict[str, float]


class ExplainabilityResult(BaseModel):
    """Output payload from the Explainability module.

    The four original fields are unchanged and always present. The rest are
    additive and populated only when the request resolved a model adapter
    (i.e. ``?model_id=...``); they default to None so a no-adapter response
    serializes exactly as it did before.

    WHY THEY MUST BE DECLARED HERE: a response_model silently DROPS keys it
    does not declare. Without ``scale``, a probability-scale TreeSHAP result
    would reach a consumer with no unit attached, and any consumer inferring
    the unit from ``method == "shap"`` would label it log-odds -- numbers that
    look entirely normal while describing the wrong quantity.
    """
    method: Literal["shap", "lime"]
    per_instance: list[PerInstanceContribution]
    global_importance: dict[str, float]
    is_mock: bool

    # --- Identity: derived from the adapter actually explained -------------
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    integration_type: Optional[str] = None

    # --- How to read the numbers ------------------------------------------
    # available=False is a first-class result (empty per_instance /
    # global_importance plus limitations), NOT an error and NOT is_mock.
    available: Optional[bool] = None
    explainer: Optional[str] = None
    # "log_odds" for linear SHAP; "probability" for tree/kernel SHAP and LIME.
    # Read this rather than inferring a scale from `method`.
    scale: Optional[str] = None
    # "exact" | "approximate" | "surrogate" -- three different claims.
    fidelity: Optional[str] = None
    feature_space: Optional[list[str]] = None
    limitations: Optional[list[str]] = None


class FairnessGroup(BaseModel):
    """Per-group selection rate and sample size breakdown for protected attributes."""
    group: str
    count: int
    favorable_count: int
    selection_rate: float


class FairnessResult(BaseModel):
    """Output payload for fairness evaluation."""
    protected_attribute: str
    demographic_parity_diff: float
    disparate_impact_ratio: float
    status: Status
    is_mock: bool
    groups: list[FairnessGroup] = Field(default_factory=list)


class DriftPerFeature(BaseModel):
    """Per-feature PSI and KS drift metric entry."""
    feature: str
    psi: float
    ks_statistic: float


class DriftResult(BaseModel):
    """Output payload for data/concept drift evaluation."""
    features_evaluated: list[str]
    psi: float
    ks_statistic: float
    status: Status
    is_mock: bool
    note: Optional[str] = None
    per_feature: list[DriftPerFeature] = Field(default_factory=list)


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
    model_id: Optional[str] = None
    assurance_run_id: Optional[str] = None


class VerifiedRequirementFinding(BaseModel):
    """One verified RBI requirement, its applicability and its evidence status.

    A DIFFERENT layer from ``ComplianceFinding`` above, and deliberately kept
    separate rather than folded into it:

    - ``ComplianceFinding`` reports the six-rule engine against technical
      findings, using the four analytical statuses (PASS/WARNING/FAIL/PENDING).
    - This reports a clause read from an official RBI instrument, using the
      requirement vocabulary (``app/rbi/requirements.py``), which additionally
      distinguishes EVIDENCE_MISSING, NOT_ASSESSED and APPLICABILITY_UNCLEAR.

    Collapsing the two would force those three distinctions into PENDING and
    lose exactly the information a reviewer needs: whether a requirement did
    not apply, or applied and could not be evidenced.

    ``status`` is intentionally a free string rather than the four-value
    ``Status`` literal, because this layer's vocabulary is the requirement one.
    Every value is produced deterministically by
    ``app.rbi.verified_requirements``; none is ever set by an LLM.
    """

    evidence_type: str
    requirement_id: str
    document_id: str
    document_title: Optional[str] = None
    clause: str
    requirement: str
    quote: Optional[str] = None
    source_url: str
    instrument_type: str
    regulatory_status: str
    assessment_mode: str
    applicability: str
    applicability_reasons: list[str] = Field(default_factory=list)
    status: str
    reason: Optional[str] = None
    limitation: Optional[str] = None
    verified_on: Optional[str] = None
    is_mock: bool = False
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    assurance_run_id: Optional[str] = None


class ComplianceResult(BaseModel):
    """Output payload from the Compliance/Rule Engine module."""
    findings: list[ComplianceFinding]
    is_mock: bool
    model_id: Optional[str] = None
    assurance_run_id: Optional[str] = None
    # Additive. Empty unless the caller declares an entity profile, because an
    # undeclared profile makes every requirement's applicability UNCLEAR and
    # there is nothing useful to report. Never merged into ``findings``: these
    # are verified regulatory clauses, not rule-engine outputs.
    verified_requirements: list[VerifiedRequirementFinding] = Field(
        default_factory=list
    )


class AssuranceResult(BaseModel):
    """Full aggregated assurance result across every evaluation domain.

    The five original fields are unchanged. The rest are additive:

    - ``model_id`` / ``assurance_run_id`` make the run self-identifying. Every
      nested result and every evidence record produced by the same run carries
      the SAME ``assurance_run_id``, so a reviewer can gather one run's output
      without relying on request timing.
    - ``monitoring`` is the monitoring lane's own contract, unchanged.
    - ``monitoring_unavailable_reason`` is set when, and only when,
      ``monitoring`` is null. A null monitoring block with a stated reason is a
      different claim from a monitoring run that measured no drift, and the two
      must never be confused by a consumer.
    """
    model: ModelResult
    explainability: ExplainabilityResult
    fairness_drift: FairnessDriftResult
    compliance: ComplianceResult
    note: str

    model_id: Optional[str] = None
    assurance_run_id: Optional[str] = None
    # Forward reference: the monitoring schemas are defined further down (they
    # depend on DriftResult/FairnessResult above). Resolved by the
    # model_rebuild() call at the end of this module.
    monitoring: Optional["MonitoringAssuranceResult"] = None
    monitoring_unavailable_reason: Optional[str] = None


# =============================================================================
# Phase 5A/5B Model Assurance Contract & Identity Envelopes (New, additive)
# =============================================================================

class AssuranceRunContext(BaseModel):
    """Identity envelope threaded through one assurance run. New (5A)."""
    model_id: str
    model_version: str
    assurance_run_id: str
    adapter_id: Optional[str] = None


class FairnessAssuranceEnvelope(BaseModel):
    context: AssuranceRunContext
    result: FairnessResult          # existing, unmodified


class DriftAssuranceEnvelope(BaseModel):
    context: AssuranceRunContext
    dataset_id: str
    dataset_version: Optional[str] = None
    feature_space: str
    result: DriftResult             # existing, unmodified


class DriftComparisonResult(BaseModel):
    comparability: Literal["COMPARABLE", "NOT_COMPARABLE"]
    reason: Optional[str] = None
    drift_a: DriftAssuranceEnvelope
    drift_b: DriftAssuranceEnvelope


# =============================================================================
# Monitoring Schemas (Owner: Arushi)
#
# These are the HTTP projection of the monitoring lane's EXISTING internal
# contract (app/monitoring/monitor.py, evidence.py) -- not a parallel model of
# it. FeatureDrift reuses DriftResult and the fairness channel reuses
# FairnessResult unchanged, so a monitored drift/fairness result is the same
# shape as the one /fairness-drift already returns. Only the genuinely new
# concepts (prediction drift, window metadata, channel status, alerts) get a
# new model.
# =============================================================================

class MonitoringWindowInfo(BaseModel):
    """One window's own declared metadata.

    provenance is Optional and None means "the caller did not state where this
    data came from". It is NEVER defaulted to "observed" -- that would let
    generated scenario data be read as real observed drift.

    window_start/window_end are monitoring-window BOUNDARIES (the period the
    records describe), not collection or scoring timestamps, carried as
    ISO-8601 strings.
    """
    window_id: str
    provenance: Optional[Literal["observed", "mock", "synthetic_fixture"]] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    record_count: Optional[int] = None


class MonitoringWindows(BaseModel):
    reference: MonitoringWindowInfo
    current: MonitoringWindowInfo


class PredictionDriftResult(BaseModel):
    """Drift in the MODEL'S OWN OUTPUT -- distinct from feature drift.

    Two channels measured with two different PSI formulations (categorical over
    labels, quantile over scores); see docs/thresholds.md 3.6. score_* is None
    with score_availability="unavailable_no_scores" when the model has no
    probability capability -- never a fabricated zero.
    """
    label_psi: float
    label_status: Status
    classes_evaluated: list[Any] = Field(default_factory=list)
    # Left as plain dicts rather than a model: each entry's key is literally
    # "class", which is a Python keyword and would need an alias whose
    # serialization behaviour could silently rename the field on the wire. The
    # producing module already defines the shape (class/reference_rate/
    # current_rate) and re-declaring it here would be a second source of truth.
    per_class: list[dict[str, Any]] = Field(default_factory=list)
    score_psi: Optional[float] = None
    score_ks_statistic: Optional[float] = None
    score_status: Status
    score_availability: Literal["computed", "unavailable_no_scores"]
    status: Status
    is_mock: bool


class MonitoringAlert(BaseModel):
    """Detection only -- no delivery, routing, or suppression state.

    An alert carries no severity of its own: it echoes its channel's existing
    status, so it can never disagree with the result it came from.
    """
    channel: str
    status: Status
    detail: dict[str, Any] = Field(default_factory=dict)


class MonitoringResult(BaseModel):
    """One monitoring run over a reference/current window pair."""
    context: AssuranceRunContext
    reference_window_id: str
    current_window_id: str
    windows: MonitoringWindows
    feature_drift: Optional[DriftResult] = None
    prediction_drift: Optional[PredictionDriftResult] = None
    fairness: Optional[FairnessResult] = None
    channel_status: dict[str, Status]
    monitoring_status: Status
    alerts: list[MonitoringAlert] = Field(default_factory=list)
    is_mock: bool


class MonitoringAssuranceResult(BaseModel):
    """The monitoring lane's outward contract: result plus its evidence.

    ``evidence`` records are flat dicts carrying their own ``evidence_type``
    (one of app.monitoring.evidence.MONITORING_EVIDENCE_TYPES) and identity.
    They are typed as plain dicts here for the same reason the report layer
    treats evidence records as dicts: the evidence contract is defined by its
    producing module, and re-declaring it here would create a second source of
    truth for it.

    ``protected_attribute`` is the attribute fairness was actually evaluated
    over, or None when the adapter declared none and the caller supplied none.
    None means the fairness channel is PENDING -- never that it passed.
    """
    result: MonitoringResult
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    protected_attribute: Optional[str] = None


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
    model_id: Optional[str] = None


class Citation(BaseModel):
    """Source reference for retrieved regulatory evidence.

    The five optional fields below are additive (Phase 3 C3) and carry the
    canonical source attribution that ``app.rag.evidence.RBIEvidence`` already
    holds, so a citation is not reduced to quote + filename. They default to
    ``None`` so every existing caller stays valid.

    ``is_excerpt`` / ``is_current`` matter most: the one approved source is a
    limited 2014 excerpt (``is_excerpt=True``, ``is_current=False``). Retrieving
    it must never present it as current or binding regulation
    (docs/decisions.md, "Historical / excerpt status").
    """
    source: str
    locator: str
    quote: str
    provenance: Literal["verified", "illustrative", "interim_single_document"]
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    document_type: Optional[str] = None
    is_excerpt: Optional[bool] = None
    is_current: Optional[bool] = None


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
    """Three-layer report section preserving distinct evidence boundaries (never collapsed).

    ``supporting_evidence`` is additive (Phase 3 C3): the structured records the
    Phase 3 evidence builders produced for THIS section, carried through
    verbatim. It is a fourth, clearly-separate channel -- it never merges into
    ``technical_finding`` and never becomes ``llm_interpretation`` text.

    Records are routed by their own ``evidence_type``, so population-level
    fairness evidence and instance-level explanation evidence cannot end up
    attached to the same section by accident. Defaults to empty, so existing
    callers and fixtures stay valid.
    """
    heading: str
    technical_finding: TechnicalFinding
    retrieved_evidence: RetrievedEvidence
    llm_interpretation: LLMInterpretation
    supporting_evidence: list[dict[str, Any]] = Field(default_factory=list)


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
    # Run identity (additive). A report is evidence about ONE model produced by
    # ONE assurance run; without these a reader cannot tell which. Optional so
    # the pre-existing default-model report and the mock fallback still
    # validate unchanged.
    model_id: Optional[str] = None
    assurance_run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Resolve forward references declared before their target model.
#
# AssuranceResult.monitoring points at MonitoringAssuranceResult, which is
# defined below it because it depends on DriftResult/FairnessResult. Without
# this rebuild the annotation stays an unresolved string and FastAPI cannot
# build the response schema.
# ---------------------------------------------------------------------------
AssuranceResult.model_rebuild()
