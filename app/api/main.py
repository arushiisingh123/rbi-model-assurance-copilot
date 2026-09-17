"""FastAPI application for AI Model Risk & Assurance Copilot (Owner: Khushi).

Exposes schema-validated endpoints for credit model evaluation, explainability,
fairness & drift, and RBI compliance assurance.
"""
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger(__name__)

from app.api.mock_data import MOCK_ASSURANCE_RESULT, MOCK_REPORT_RESULT
from app.api.orchestration import (
    build_assurance_result,
    build_drift_comparison,
    build_evidence_by_rule,
    build_evidence_records,
    build_drift_assurance_envelope,
    build_fairness_assurance_envelope,
    compute_real_compliance,
    compute_real_drift,
    compute_real_explainability,
    compute_real_fairness,
    compute_real_model,
    compute_real_model_metrics,
    format_model_for_api,
)
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    DriftAssuranceEnvelope,
    DriftComparisonResult,
    ExplainabilityResult,
    FairnessAssuranceEnvelope,
    FairnessDriftResult,
    ModelResult,
    ReportResult,
)
from app.models import (
    ModelAdapter,
    ModelNotFoundError,
    get_default_registry,
)

app = FastAPI(
    title="AI Model Risk & Assurance Copilot",
    version="0.2.0-phase2",
    description="API for credit-scoring model assurance and RBI compliance evidence.",
)


def _resolve_adapter(model_id: Optional[str]) -> Optional[ModelAdapter]:
    """Resolve a model_id to its registered ModelAdapter.

    None -> None (preserves today's default-LR behavior exactly).
    Otherwise -> get_default_registry().get(model_id), raising HTTPException(404)
    on ModelNotFoundError.
    """
    if model_id is None:
        return None
    try:
        return get_default_registry().get(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.get("/models")
def list_models() -> list[dict]:
    """List all registered credit-scoring models and their metadata."""
    return get_default_registry().list_models()


@app.get("/models/{model_id}")
def get_model_metadata(model_id: str) -> dict:
    """Retrieve metadata for a specific registered model."""
    try:
        return get_default_registry().get(model_id).metadata()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/models/{model_id}/health")
def get_model_health(model_id: str) -> dict:
    """Check health/readiness status for a specific registered model."""
    try:
        return get_default_registry().get(model_id).health()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/model", response_model=ModelResult)
def get_model(model_id: Optional[str] = Query(default=None)) -> dict:
    """Retrieve real model predictions, feature matrix (list[dict]), metadata, and metrics.

    model_metrics is real held-out performance for the default model and
    for any adapter sharing German Credit's schema (e.g. RandomForestAdapter).
    For an adapter with a different schema (e.g. the synthetic bank's
    RESTAdapter), evaluate_current_model() raises ValueError -- German
    Credit's held-out split has no meaning for it -- and model_metrics is
    null rather than a crash or another model's numbers under this one's
    label.
    """
    adapter = _resolve_adapter(model_id)
    raw_model = compute_real_model(adapter=adapter)
    model_dict = format_model_for_api(raw_model)
    if adapter is None:
        model_dict["model_metrics"] = compute_real_model_metrics()
    else:
        try:
            model_dict["model_metrics"] = compute_real_model_metrics(adapter=adapter)
        except ValueError:
            model_dict["model_metrics"] = None
    return model_dict


@app.get("/explainability", response_model=ExplainabilityResult)
def get_explainability(
    method: str = Query(
        default="shap",
        description="Explainability method ('shap' or 'lime')",
    ),
    model_id: Optional[str] = Query(default=None),
) -> dict:
    """Retrieve real explainability results for the credit scoring model (SHAP or LIME)."""
    adapter = _resolve_adapter(model_id)
    raw_model = compute_real_model(adapter=adapter)
    try:
        return compute_real_explainability(raw_model, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/fairness-drift", response_model=FairnessDriftResult)
def get_fairness_drift(model_id: Optional[str] = Query(default=None)) -> dict:
    """Retrieve real fairness metrics and train/test drift detection analysis."""
    adapter = _resolve_adapter(model_id)
    raw_model = compute_real_model(adapter=adapter)
    fairness_res = compute_real_fairness(raw_model, adapter=adapter)
    drift_res = compute_real_drift(raw_model, adapter=adapter)
    return {
        "fairness": fairness_res,
        "drift": drift_res,
    }


@app.get("/fairness-assurance", response_model=FairnessAssuranceEnvelope)
def get_fairness_assurance() -> dict:
    """Retrieve real fairness metrics wrapped in a Phase 5A identity envelope."""
    raw_model = compute_real_model()
    fairness_res = compute_real_fairness(raw_model)
    model_version = raw_model.get("model_metadata", {}).get("version", "0.1.0")
    return build_fairness_assurance_envelope(
        fairness_res, model_version=model_version
    )


@app.get("/drift-assurance", response_model=DriftAssuranceEnvelope)
def get_drift_assurance() -> dict:
    """Retrieve real drift metrics wrapped in a Phase 5A identity envelope."""
    raw_model = compute_real_model()
    drift_res = compute_real_drift(raw_model)
    model_version = raw_model.get("model_metadata", {}).get("version", "0.1.0")
    try:
        return build_drift_assurance_envelope(drift_res, model_version=model_version)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/drift-comparison", response_model=DriftComparisonResult)
def get_drift_comparison() -> dict:
    """Compare drift between the default Logistic Regression and
    Random Forest models.

    Always returns HTTP 200. COMPARABLE vs NOT_COMPARABLE is the
    response payload, not an HTTP status code -- this mirrors
    app/drift/comparability.py's own design rationale ("not an
    HTTP error code -- gives the dashboard nothing to render").
    """
    try:
        return build_drift_comparison()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/compliance", response_model=ComplianceResult)
def get_compliance(model_id: Optional[str] = Query(default=None)) -> dict:
    """Retrieve RBI compliance findings mapped against real technical checks.

    Each finding's evidence_chunks is populated from a real, live RAG
    retrieval per rule category (build_evidence_by_rule(), Phase 5D) rather
    than left as [] -- the same evidence-wiring build_assurance_result()
    already uses, now also reachable through this standalone route.
    """
    adapter = _resolve_adapter(model_id)
    raw_model = compute_real_model(adapter=adapter)
    explain_res = compute_real_explainability(raw_model, method="shap")
    fairness_res = compute_real_fairness(raw_model, adapter=adapter)
    drift_res = compute_real_drift(raw_model, adapter=adapter)
    evidence_by_rule = build_evidence_by_rule()
    return compute_real_compliance(
        raw_model, explain_res, fairness_res, drift_res, evidence_by_rule=evidence_by_rule
    )


@app.get("/assurance-result", response_model=AssuranceResult)
def get_assurance_result() -> dict:
    """Retrieve aggregated assurance results across all four evaluation domains."""
    return build_assurance_result()


@app.get("/mock-assurance-result", response_model=AssuranceResult, deprecated=True)
def mock_assurance_result() -> dict:
    """[DEPRECATED] Phase 0 mock result endpoint.

    Use /assurance-result instead. Retained for backwards compatibility with
    Phase 0 dashboard callers.
    """
    return MOCK_ASSURANCE_RESULT


@app.get("/report", response_model=ReportResult)
def get_report() -> dict:
    """Retrieve Phase 3 LLM model assurance report.

    Attempts live report generation using Groq (openai/gpt-oss-120b) and
    interim RAG retrieval. If live generation is unavailable (e.g. missing
    GROQ_API_KEY, API error, rate limit, timeout), falls back gracefully
    to MOCK_REPORT_RESULT with a disclaimer. Never returns HTTP 500.
    """
    if not os.getenv("GROQ_API_KEY", "").strip():
        logger.info("GROQ_API_KEY not configured; returning mock report fallback immediately.")
        fallback = dict(MOCK_REPORT_RESULT)
        fallback_disclaimers = list(fallback.get("disclaimers", []))
        disclaimer = (
            "FALLBACK MOCK REPORT: Live report generation unavailable (ReportGenerationUnavailable). "
            "Displaying static mock fixture."
        )
        if disclaimer not in fallback_disclaimers:
            fallback_disclaimers.insert(0, disclaimer)
        fallback["disclaimers"] = fallback_disclaimers
        return fallback

    try:
        raw_model = compute_real_model()
        explain_res = compute_real_explainability(raw_model, method="shap")
        fairness_res = compute_real_fairness(raw_model)
        drift_res = compute_real_drift(raw_model)
        compliance_res = compute_real_compliance(raw_model, explain_res, fairness_res, drift_res)
        evidence_records = build_evidence_records(
            raw_model, explain_res, method="shap"
        )
        from app.report import generate_report
        return generate_report(
            model=raw_model,
            explainability=explain_res,
            fairness=fairness_res,
            drift=drift_res,
            compliance=compliance_res,
            evidence_records=evidence_records,
        )
    except Exception as exc:
        logger.warning("Live report generation failed, falling back to mock: %s", exc)
        fallback = dict(MOCK_REPORT_RESULT)
        fallback_disclaimers = list(fallback.get("disclaimers", []))
        disclaimer = (
            f"FALLBACK MOCK REPORT: Live report generation unavailable ({exc.__class__.__name__}). "
            "Displaying static mock fixture."
        )
        if disclaimer not in fallback_disclaimers:
            fallback_disclaimers.insert(0, disclaimer)
        fallback["disclaimers"] = fallback_disclaimers
        return fallback
