"""FastAPI application for AI Model Risk & Assurance Copilot (Owner: Khushi).

Exposes schema-validated endpoints for credit model evaluation, explainability,
fairness & drift, and RBI compliance assurance.
"""
from fastapi import FastAPI, HTTPException, Query

from app.api.mock_data import MOCK_ASSURANCE_RESULT, MOCK_REPORT_RESULT
from app.api.orchestration import (
    build_assurance_result,
    compute_real_compliance,
    compute_real_drift,
    compute_real_explainability,
    compute_real_fairness,
    compute_real_model,
    format_model_for_api,
)
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    ExplainabilityResult,
    FairnessDriftResult,
    ModelResult,
    ReportResult,
)

app = FastAPI(
    title="AI Model Risk & Assurance Copilot",
    version="0.2.0-phase2",
    description="API for credit-scoring model assurance and RBI compliance evidence.",
)


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.get("/model", response_model=ModelResult)
def get_model() -> dict:
    """Retrieve real model predictions, feature matrix (list[dict]), and metadata."""
    raw_model = compute_real_model()
    return format_model_for_api(raw_model)


@app.get("/explainability", response_model=ExplainabilityResult)
def get_explainability(
    method: str = Query(
        default="shap",
        description="Explainability method ('shap' or 'lime')",
    ),
) -> dict:
    """Retrieve real explainability results for the credit scoring model (SHAP or LIME)."""
    raw_model = compute_real_model()
    try:
        return compute_real_explainability(raw_model, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/fairness-drift", response_model=FairnessDriftResult)
def get_fairness_drift() -> dict:
    """Retrieve real fairness metrics and train/test drift detection analysis."""
    raw_model = compute_real_model()
    fairness_res = compute_real_fairness(raw_model)
    drift_res = compute_real_drift(raw_model)
    return {
        "fairness": fairness_res,
        "drift": drift_res,
    }


@app.get("/compliance", response_model=ComplianceResult)
def get_compliance() -> dict:
    """Retrieve RBI compliance findings mapped against real technical checks."""
    raw_model = compute_real_model()
    explain_res = compute_real_explainability(raw_model, method="shap")
    fairness_res = compute_real_fairness(raw_model)
    drift_res = compute_real_drift(raw_model)
    return compute_real_compliance(raw_model, explain_res, fairness_res, drift_res)


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
    """Mock-backed; generate_report() pending team decision on module home for the real report generator."""
    return MOCK_REPORT_RESULT
