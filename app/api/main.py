"""Phase 0 FastAPI skeleton (owner: Khushi).

Only a health check and one hardcoded mock endpoint exist so far.
Real endpoints that wire in the other modules' real outputs are
Phase 2 work (see docs/development-phases.md).

Run locally:
    uvicorn app.api.main:app --reload
"""
from fastapi import FastAPI

app = FastAPI(title="AI Model Risk & Assurance Copilot", version="0.0.1-phase0")

MOCK_ASSURANCE_RESULT = {
    "model": {"status": "PASS", "is_mock": True},
    "explainability": {"status": "PASS", "is_mock": True},
    "fairness_drift": {"status": "WARNING", "is_mock": True},
    "compliance": {"status": "PENDING", "is_mock": True},
    "note": "SYNTHETIC / MOCK DATA. Not real results. Phase 0 skeleton only.",
}


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.get("/mock-assurance-result")
def mock_assurance_result() -> dict:
    """Hardcoded mock result so the dashboard has something to render
    before real modules are wired together (Phase 2)."""
    return MOCK_ASSURANCE_RESULT
