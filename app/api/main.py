"""FastAPI application for AI Model Risk & Assurance Copilot (Owner: Khushi).

Exposes schema-validated endpoints for credit model evaluation, explainability,
fairness & drift, and RBI compliance assurance.
"""
from fastapi import FastAPI, HTTPException, Query

from app.api.mock_data import (
    MOCK_ASSURANCE_RESULT,
    MOCK_COMPLIANCE_RESULT,
    MOCK_DRIFT_RESULT,
    MOCK_EXPLAINABILITY_RESULT_LIME,
    MOCK_EXPLAINABILITY_RESULT_SHAP,
    MOCK_FAIRNESS_RESULT,
    MOCK_MODEL_RESULT,
)
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    ExplainabilityResult,
    FairnessDriftResult,
    ModelResult,
)

app = FastAPI(
    title="AI Model Risk & Assurance Copilot",
    version="0.1.0-phase1",
    description="API for credit-scoring model assurance and RBI compliance evidence.",
)


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.get("/model", response_model=ModelResult)
def get_model() -> dict:
    """Retrieve model predictions, feature matrix, and metadata."""
    return MOCK_MODEL_RESULT


@app.get("/explainability", response_model=ExplainabilityResult)
def get_explainability(
    method: str = Query(
        default="shap",
        description="Explainability method ('shap' or 'lime')",
    ),
) -> dict:
    """Retrieve explainability results for the credit scoring model (SHAP or LIME)."""
    method_lower = method.lower()
    if method_lower == "shap":
        return MOCK_EXPLAINABILITY_RESULT_SHAP
    elif method_lower == "lime":
        return MOCK_EXPLAINABILITY_RESULT_LIME
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid explainability method '{method}'. Supported methods are 'shap' and 'lime'.",
        )


@app.get("/fairness-drift", response_model=FairnessDriftResult)
def get_fairness_drift() -> dict:
    """Retrieve fairness metrics and drift detection analysis."""
    return {
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": MOCK_DRIFT_RESULT,
    }


@app.get("/compliance", response_model=ComplianceResult)
def get_compliance() -> dict:
    """Retrieve RBI compliance findings mapped against technical checks."""
    return MOCK_COMPLIANCE_RESULT


@app.get("/assurance-result", response_model=AssuranceResult)
def get_assurance_result() -> dict:
    """Retrieve aggregated assurance results across all four evaluation domains."""
    return MOCK_ASSURANCE_RESULT


@app.get("/mock-assurance-result", response_model=AssuranceResult, deprecated=True)
def mock_assurance_result() -> dict:
    """[DEPRECATED] Phase 0 mock result endpoint.

    Use /assurance-result instead. Retained for backwards compatibility with
    Phase 0 dashboard callers.
    """
    return MOCK_ASSURANCE_RESULT
