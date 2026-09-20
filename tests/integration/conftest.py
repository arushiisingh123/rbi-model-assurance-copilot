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


# ---------------------------------------------------------------------------
# "The model's service is unavailable" -- made TRUE, not assumed.
#
# Several tests assert the unreachable-upstream contract (health
# "unreachable", /explainability 502, monitoring degrading to None + reason).
# They used to rely on nothing listening on the registry's DEFAULT address,
# 127.0.0.1:8100 -- an ambient property of the developer's machine that the
# tests neither established nor checked.
#
# That made them pass or fail on unrelated state. Running the synthetic bank
# for a demo (the documented way to run this project) leaves a real service on
# 8100, the adapter reaches it, and three tests fail while the application is
# behaving perfectly correctly.
#
# This fixture removes the assumption instead of the assertion: it points the
# registry at a port confirmed to have nothing on it, so the REAL RESTAdapter
# makes a REAL connection attempt that REALLY fails. No mock, no stub, no
# patched transport, and the assertions are untouched -- the failure being
# asserted is a genuine one, just one the test now causes rather than hopes
# for.
# ---------------------------------------------------------------------------


def _closed_local_port(attempts: int = 20) -> int:
    """A loopback port with nothing listening on it.

    Binds port 0 so the OS picks a free one, releases it, then CONFIRMS a
    connection is refused before handing it back. The confirmation matters:
    without it this would be the same "probably nothing is there" assumption
    the fixture exists to eliminate.
    """
    import socket

    for _ in range(attempts):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        finally:
            probe.close()

        check = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        check.settimeout(0.25)
        try:
            if check.connect_ex(("127.0.0.1", port)) != 0:
                return port  # refused -- genuinely closed
        finally:
            check.close()

    raise RuntimeError(
        "could not obtain a confirmed-closed loopback port after "
        f"{attempts} attempts"
    )


@pytest.fixture
def unreachable_synthetic_bank(monkeypatch):
    """Point the registry's synthetic bank at a confirmed-closed port.

    Yields the endpoint URL. Restores the previous environment and rebuilds
    the registry afterwards, so a test that needs the real service is
    unaffected -- reset order matters, because the cached registry singleton
    would otherwise keep holding an adapter built from this URL.
    """
    from app.models.registry import reset_default_registry

    endpoint = f"http://127.0.0.1:{_closed_local_port()}"
    monkeypatch.setenv("SYNTHETIC_BANK_URL", endpoint)
    reset_default_registry()
    try:
        yield endpoint
    finally:
        monkeypatch.delenv("SYNTHETIC_BANK_URL", raising=False)
        reset_default_registry()
