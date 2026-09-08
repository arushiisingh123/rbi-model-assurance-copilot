"""Shared fixtures for cross-module integration tests.

OUT OF SCOPE for this suite (documented, not tested):
1. "Multiple relevant documents / evidence ranking" — only one interim document
   exists; multi-document retrieval is Nidhi's future RAG work.
2. "Invalid source rejected/flagged" — there is no source-validation layer;
   IsolatedRAGRetriever uses a fixed DOCUMENT_PATH.
3. "Synthetic drift labelled synthetic" — per docs/decisions.md 2026-09-07,
   the integrated drift path is the real train/test split (is_mock=False,
   provenance "observed"). The synthetic path now only exists in the mock
   fixtures, covered by test_provenance_labels_preserved.py.
"""
import os
from typing import Any, Dict
import pytest

from app.api.orchestration import (
    build_assurance_result,
    compute_real_compliance,
    compute_real_drift,
    compute_real_explainability,
    compute_real_fairness,
    compute_real_model,
)
from app.models.model import DEFAULT_MODEL_ARTIFACT_PATH, train
from app.models.preprocessing import DEFAULT_DATASET_PATH


@pytest.fixture(scope="session", autouse=True)
def _ensure_model_artifact():
    """Provision credit_model.joblib if missing by calling app.models.model.train()."""
    if not os.path.exists(DEFAULT_MODEL_ARTIFACT_PATH):
        train(
            dataset_path=DEFAULT_DATASET_PATH,
            save_path=DEFAULT_MODEL_ARTIFACT_PATH,
            random_state=42,
        )
    return DEFAULT_MODEL_ARTIFACT_PATH


@pytest.fixture(scope="session")
def real_pipeline() -> Dict[str, Any]:
    """Execute the real analytical pipeline once for the session and return artifacts."""
    raw_model = compute_real_model()
    explain_res = compute_real_explainability(raw_model, method="shap")
    fairness_res = compute_real_fairness(raw_model)
    drift_res = compute_real_drift(raw_model)
    compliance_res = compute_real_compliance(raw_model, explain_res, fairness_res, drift_res)
    assurance = build_assurance_result()
    return {
        "model": raw_model,
        "explainability": explain_res,
        "fairness": fairness_res,
        "drift": drift_res,
        "compliance": compliance_res,
        "assurance_result": assurance,
    }


@pytest.fixture
def no_groq_key(monkeypatch):
    """Ensure GROQ_API_KEY is unset for testing graceful degradation."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
