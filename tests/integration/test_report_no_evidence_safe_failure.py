"""Exercises the NOT_FOUND safe-failure path through the real pipeline + real IsolatedRAGRetriever, not hand-crafted fixtures.

Scope note: this file covers only the NOT_FOUND branch -- that an unsupported
regulatory claim is stripped when no evidence backs it. That the real retriever
can reach RETRIEVED at all is asserted in ``tests/report/test_generate_report.py``
(the C1 regression tests). Both halves are needed: before the C1 fix, real
retrieval raised TypeError into the broad fallback, so all five sections became
NOT_FOUND and this file passed for the wrong reason.
"""
from app.api.schemas import ReportResult
from app.report.generate import (
    SAFE_FALLBACK_TEXT,
    generate_report,
)
from tests.report.test_generate_report import FakeGroqClient


def test_not_found_safe_failure_end_to_end(real_pipeline, monkeypatch):
    """Test full real retrieval path triggering NOT_FOUND safe-failure and regulatory claim stripping."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    hallucinated = {
        k: "This finding means the model must comply with RBI circular clause 2.1 and the governing regulation mandates action."
        for k in ["model", "explainability", "fairness", "drift", "compliance"]
    }

    # retrieval_fn=None uses the real IsolatedRAGRetriever against the actual IRAC document
    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=FakeGroqClient(canned_response_dict=hallucinated),
        retrieval_fn=None,
    )

    validated = ReportResult(**out)
    assert validated is not None

    not_found_sections = [
        s for s in out["sections"]
        if s["retrieved_evidence"]["evidence_status"] == "NOT_FOUND"
    ]
    # In the real IRAC document, ML governance keywords (model, explainability,
    # fairness, drift) do not exist, so exactly those four are NOT_FOUND while
    # the compliance/NPA section does retrieve. Asserted as a set rather than
    # ">= 1": all five being NOT_FOUND was the C1 symptom (retrieval raising
    # TypeError into the broad fallback), and a loose count could not tell the
    # two situations apart.
    assert {s["heading"] for s in not_found_sections} == {
        "Credit Scoring Model Evaluation",
        "Feature Explainability (SHAP)",
        "Fairness Evaluation",
        "Data & Prediction Drift Detection",
    }

    for s in not_found_sections:
        ev = s["retrieved_evidence"]
        interp = s["llm_interpretation"]
        assert ev["evidence_status"] == "NOT_FOUND"
        assert ev["citations"] == []
        assert interp["regulatory_basis"] == "none"
        # The hallucinated regulatory claim must have been replaced with SAFE_FALLBACK_TEXT
        assert interp["text"] == SAFE_FALLBACK_TEXT
        assert "RBI" not in interp["text"]
        assert "§" not in interp["text"]

    coverage = out["evidence_coverage"]
    assert coverage["not_found"] >= 1
    assert coverage["retrieved"] + coverage["not_found"] == coverage["total"] == 5

    # No section may have "cited_evidence" (Phase 3 regulatory corpus is interim only)
    for s in out["sections"]:
        assert s["llm_interpretation"]["regulatory_basis"] != "cited_evidence"


def test_retrieval_exception_fails_safe(real_pipeline):
    """Ensure that if the retrieval backend raises an exception, generate_report fails safely."""
    def boom(query: str):
        raise RuntimeError("retrieval backend down")

    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=FakeGroqClient(),
        retrieval_fn=boom,
    )

    validated = ReportResult(**out)
    assert validated is not None
    assert len(out["sections"]) == 5

    for s in out["sections"]:
        assert s["retrieved_evidence"]["evidence_status"] == "NOT_FOUND"
        assert s["llm_interpretation"]["regulatory_basis"] == "none"
