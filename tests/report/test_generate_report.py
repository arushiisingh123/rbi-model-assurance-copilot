"""Unit and integration tests for Phase 3 LLM Report Generation.

Verifies:
- Schema validation against ReportResult
- Python-enforced regulatory_basis constraints
- Critical safety scan-and-strip for NOT_FOUND sections
- Zero real network calls in test suite (mock/fake injection)
- Graceful fallback on missing API key or Groq errors
"""
from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.mock_data import (
    MOCK_COMPLIANCE_RESULT,
    MOCK_DRIFT_RESULT,
    MOCK_EXPLAINABILITY_RESULT,
    MOCK_FAIRNESS_RESULT,
    MOCK_MODEL_RESULT,
)
from app.api.schemas import ReportResult
from app.report.generate import (
    DEFAULT_PROVIDER_LABEL,
    IsolatedRAGRetriever,
    ReportGenerationError,
    ReportGenerationUnavailable,
    SAFE_FALLBACK_TEXT,
    SAFE_RETRIEVED_FALLBACK_TEMPLATE,
    generate_report,
)


@dataclass
class FakeChoice:
    message: Any


@dataclass
class FakeMessage:
    content: str


class FakeGroqClient:
    """Fake Groq client providing .chat.completions.create() with canned responses."""

    def __init__(self, canned_response_dict: Optional[Dict[str, str]] = None, raise_exc: Optional[Exception] = None):
        self.call_count = 0
        self.last_prompt = None
        self.raise_exc = raise_exc
        if canned_response_dict is None:
            canned_response_dict = {
                "model": "Model metadata records baseline probabilities and feature dimensions.",
                "explainability": "SHAP feature attributions identify top input contributors.",
                "fairness": "Disparate impact ratio is computed against the internal 0.80 benchmark.",
                "drift": "Population stability index and KS tests evaluate distribution shift.",
                "compliance": "Technical checks are mapped against sample regulatory rules.",
            }
        self.canned_response_dict = canned_response_dict

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs) -> Any:
        self.call_count += 1
        self.last_prompt = kwargs
        if self.raise_exc:
            raise self.raise_exc

        payload_json = json.dumps(self.canned_response_dict)
        msg = FakeMessage(content=f"```json\n{payload_json}\n```")
        return type("FakeChatCompletion", (), {"choices": [FakeChoice(message=msg)]})()


def fake_retrieval_all_not_found(query: str) -> Dict[str, Any]:
    """Retrieval test double returning text that clears no domain relevance keywords."""
    return {
        "query": query,
        "retrieved_text": "General bank administrative procedure regarding branch working hours.",
        "source": "RBI_CIRCULAR_SAMPLE.txt",
        "chunk_index": 0,
        "num_chunks_indexed": 1,
    }


def fake_retrieval_mixed(query: str) -> Dict[str, Any]:
    """Retrieval test double returning relevant text for compliance and non-relevant text for others."""
    if "non performing" in query.lower() or "prudential" in query.lower():
        return {
            "query": query,
            "retrieved_text": "A non-performing asset (NPA) is an advance where interest remains overdue.",
            "source": "RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt",
            "chunk_index": 2,
            "num_chunks_indexed": 17,
        }
    return {
        "query": query,
        "retrieved_text": "General bank administrative guidelines without technical keywords.",
        "source": "RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt",
        "chunk_index": 0,
        "num_chunks_indexed": 17,
    }


def fake_retrieval_all_retrieved(query: str) -> Dict[str, Any]:
    """Retrieval test double returning relevant keywords for any section."""
    return {
        "query": query,
        "retrieved_text": (
            "model risk validation explainability shap disparate impact "
            "population stability index non-performing advances prudential norms"
        ),
        "source": "RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt",
        "chunk_index": 1,
        "num_chunks_indexed": 17,
    }


@pytest.fixture
def sample_inputs():
    """Standard analytical module output inputs for report testing."""
    return {
        "model": MOCK_MODEL_RESULT,
        "explainability": MOCK_EXPLAINABILITY_RESULT,
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": MOCK_DRIFT_RESULT,
        "compliance": MOCK_COMPLIANCE_RESULT,
    }


def test_generate_report_validates_against_report_result_schema(sample_inputs):
    """Verify generate_report returns a valid ReportResult payload with 5 three-layer sections."""
    fake_client = FakeGroqClient()
    result = generate_report(
        **sample_inputs,
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_mixed,
    )

    # Validate against schema
    parsed = ReportResult(**result)
    assert parsed.is_mock is False
    assert len(parsed.sections) == 5
    assert parsed.evidence_coverage.total == 5
    assert parsed.evidence_coverage.retrieved == 1  # only compliance cleared relevance
    assert parsed.evidence_coverage.not_found == 4
    assert len(parsed.disclaimers) >= 4
    assert fake_client.call_count == 1  # Exactly ONE Groq call for whole report


def test_generate_report_not_found_sets_regulatory_basis_none(sample_inputs):
    """Verify NOT_FOUND sections force regulatory_basis = 'none' and empty citations."""
    fake_client = FakeGroqClient()
    result = generate_report(
        **sample_inputs,
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_all_not_found,
    )

    parsed = ReportResult(**result)
    assert parsed.evidence_coverage.not_found == 5
    for section in parsed.sections:
        assert section.retrieved_evidence.evidence_status == "NOT_FOUND"
        assert section.retrieved_evidence.citations == []
        assert section.llm_interpretation.regulatory_basis == "none"
        assert section.llm_interpretation.is_mock is False


def test_generate_report_retrieved_sets_regulatory_basis_illustrative(sample_inputs):
    """Verify RETRIEVED sections from interim doc force regulatory_basis = 'illustrative_rule_only'."""
    fake_client = FakeGroqClient()
    result = generate_report(
        **sample_inputs,
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_all_retrieved,
    )

    parsed = ReportResult(**result)
    assert parsed.evidence_coverage.retrieved == 5
    for section in parsed.sections:
        assert section.retrieved_evidence.evidence_status == "RETRIEVED"
        assert len(section.retrieved_evidence.citations) == 1
        assert section.retrieved_evidence.citations[0].provenance == "interim_single_document"
        assert section.llm_interpretation.regulatory_basis == "illustrative_rule_only"
        assert section.llm_interpretation.regulatory_basis != "cited_evidence"


def test_safety_scan_strips_unsupported_regulatory_claims(sample_inputs):
    """CRITICAL SAFETY TEST: Verify that when the LLM generates regulatory claims in a

    NOT_FOUND section, Python scans and replaces the text with the safe fallback string.
    """
    # Configure fake LLM to inject an ungrounded regulatory claim into the drift section
    hallucinated_responses = {
        "model": "Model metadata and baseline probability distributions are documented.",
        "explainability": "SHAP feature importances highlight duration_months.",
        "fairness": "Disparate impact ratio evaluates against the internal threshold.",
        "drift": "The model complies with RBI circular §2.1 requirement for population stability.",
        "compliance": "Technical checks align with prudential asset norms.",
    }
    fake_client = FakeGroqClient(canned_response_dict=hallucinated_responses)

    # Use retrieval where drift is NOT_FOUND
    result = generate_report(
        **sample_inputs,
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_all_not_found,
    )

    parsed = ReportResult(**result)
    drift_section = next(s for s in parsed.sections if s.technical_finding.ref == "drift.psi")

    # Safety enforcement: drift text MUST be replaced with safe fallback
    assert drift_section.retrieved_evidence.evidence_status == "NOT_FOUND"
    assert drift_section.llm_interpretation.regulatory_basis == "none"
    assert drift_section.llm_interpretation.text == SAFE_FALLBACK_TEXT
    assert "RBI" not in drift_section.llm_interpretation.text
    assert "§" not in drift_section.llm_interpretation.text


def test_generate_report_missing_api_key_raises_unavailable(sample_inputs, monkeypatch):
    """Verify generate_report() raises ReportGenerationUnavailable when GROQ_API_KEY is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ReportGenerationUnavailable) as exc_info:
        generate_report(
            **sample_inputs,
            llm_client=None,
            retrieval_fn=fake_retrieval_mixed,
        )
    assert "GROQ_API_KEY" in str(exc_info.value)


def test_report_endpoint_fallback_on_missing_key(monkeypatch):
    """Verify /report endpoint returns HTTP 200 with mock fallback when GROQ_API_KEY is unset."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(app)
    response = client.get("/report")

    assert response.status_code == 200
    body = response.json()
    parsed = ReportResult(**body)
    assert parsed.is_mock is True
    assert any("FALLBACK MOCK REPORT" in d for d in parsed.disclaimers)


def test_report_endpoint_fallback_on_groq_exception(monkeypatch, sample_inputs):
    """Verify /report endpoint returns HTTP 200 with mock fallback when Groq client fails."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_test_key")

    def failing_generate_report(**kwargs):
        raise ReportGenerationError("Groq 429 rate limit exceeded")

    monkeypatch.setattr("app.report.generate_report", failing_generate_report)

    client = TestClient(app)
    response = client.get("/report")

    assert response.status_code == 200
    body = response.json()
    parsed = ReportResult(**body)
    assert parsed.is_mock is True
    assert any("FALLBACK MOCK REPORT" in d for d in parsed.disclaimers)


def test_zero_real_network_calls(sample_inputs):
    """Verify fake LLM client and mock retrieval execute with zero live HTTP calls."""
    fake_client = FakeGroqClient()
    result = generate_report(
        **sample_inputs,
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_mixed,
    )
    assert fake_client.call_count == 1
    assert result["report_id"].startswith("rep-")


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Manual live verification only")
def test_live_groq_call_optional(sample_inputs):
    """Optional live test against Groq API. Skipped in CI/standard test run."""
    result = generate_report(
        **sample_inputs,
        llm_client=None,
        retrieval_fn=fake_retrieval_mixed,
    )
    parsed = ReportResult(**result)
    assert parsed.is_mock is False
    assert len(parsed.sections) == 5


def test_safety_scan_strips_overreaching_regulatory_claims_in_retrieved_section(sample_inputs):
    """CRITICAL SAFETY TEST (Fix B): Verify that when the LLM generates overreaching

    regulatory claims (such as a fabricated circular number not present in the citation quote)
    in a RETRIEVED section, Python scans and strips it, replacing with safe fallback.
    """
    # Configure fake LLM to fabricate a specific circular number in the retrieved compliance section
    hallucinated_responses = {
        "model": "Model metadata and baseline probability distributions are documented.",
        "explainability": "SHAP feature importances highlight duration_months.",
        "fairness": "Disparate impact ratio evaluates against the internal threshold.",
        "drift": "Population stability index and KS statistic indicate stable distributions.",
        "compliance": (
            "Under RBI circular RBI/2023-24/99, banks must classify non-performing assets within 90 days."
        ),
    }
    fake_client = FakeGroqClient(canned_response_dict=hallucinated_responses)

    # Use retrieval where compliance is RETRIEVED (citation quote does NOT contain RBI/2023-24/99)
    result = generate_report(
        **sample_inputs,
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_mixed,
    )

    parsed = ReportResult(**result)
    comp_section = next(s for s in parsed.sections if s.technical_finding.ref == "compliance.findings")

    # Must be RETRIEVED with illustrative_rule_only
    assert comp_section.retrieved_evidence.evidence_status == "RETRIEVED"
    assert comp_section.llm_interpretation.regulatory_basis == "illustrative_rule_only"

    # Safety enforcement: fabricated circular number MUST be stripped and replaced with safe fallback
    assert "RBI/2023-24/99" not in comp_section.llm_interpretation.text
    assert "Detailed regulatory interpretation was withheld because generated text went beyond the retrieved evidence excerpt" in comp_section.llm_interpretation.text


def test_isolated_rag_retriever_unique_collection_and_no_interference():
    """Verify IsolatedRAGRetriever creates a uniquely-named collection and leaves rbi_smoke_test intact."""
    retriever1 = IsolatedRAGRetriever()
    retriever2 = IsolatedRAGRetriever()

    assert retriever1.collection_name != retriever2.collection_name
    assert retriever1.collection_name.startswith("rbi_rep_")
    assert retriever2.collection_name.startswith("rbi_rep_")

    res1 = retriever1.query("What is a non performing asset?")
    assert "non-performing" in res1["retrieved_text"].lower() or "asset" in res1["retrieved_text"].lower()

    retriever1.close()
    # retriever2 still works and was not disrupted by retriever1's cleanup
    res2 = retriever2.query("What is a non performing asset?")
    assert "non-performing" in res2["retrieved_text"].lower() or "asset" in res2["retrieved_text"].lower()

    retriever2.close()


def test_generate_report_skip_live_raises_immediately(sample_inputs):
    """Verify skip_live=True raises ReportGenerationUnavailable immediately."""
    with pytest.raises(ReportGenerationUnavailable) as exc_info:
        generate_report(
            **sample_inputs,
            skip_live=True,
        )
    assert "skip_live" in str(exc_info.value)


# ======================================================================
# C1 regression: the real retriever must be reachable through the
# production call path using the shared retrieval_fn(query=...) protocol.
#
# Original failure mode: IsolatedRAGRetriever.query was declared
# `query(self, query_text)` while _retrieve_section_evidence invokes
# `retrieval_fn(query=...)`. Every real retrieval therefore raised
# TypeError, which the broad `except Exception` swallowed into
# NOT_FOUND -- indistinguishable from a legitimate "no relevant evidence"
# result. Real coverage was 0/5 while the whole suite stayed green,
# because every other test injects a retrieval_fn double.
#
# These tests fail if the parameter is renamed again, and they fail on the
# real symptom (a swallowed error / NOT_FOUND) rather than on a signature
# detail alone.
# ======================================================================


def test_isolated_retriever_query_parameter_follows_the_shared_protocol():
    """The parameter must be named `query`, matching every other retrieval_fn.

    `app.rag.retrieval.RBIRetriever.__call__` declares `query` keyword-only,
    and the fake_retrieval_* doubles all take `query`. IsolatedRAGRetriever was
    the sole outlier. Pinned by name because a mismatch here is silent.
    """
    import inspect

    from app.report.generate import IsolatedRAGRetriever as _Retriever

    params = list(inspect.signature(_Retriever.query).parameters)
    assert params == ["self", "query"], (
        f"IsolatedRAGRetriever.query params are {params}; "
        "_retrieve_section_evidence calls retrieval_fn(query=...), so the "
        "parameter must be named 'query' or every real retrieval silently "
        "becomes NOT_FOUND."
    )


def test_isolated_retriever_accepts_the_query_keyword():
    """Calling with the protocol keyword must not raise TypeError."""
    from app.report.generate import SECTION_QUERIES

    retriever = IsolatedRAGRetriever()
    try:
        outcome = retriever.query(query=SECTION_QUERIES["compliance"])
    finally:
        retriever.close()

    assert isinstance(outcome, dict)
    # The keys _retrieve_section_evidence reads off a retrieval result.
    for field in ("retrieved_text", "source", "chunk_index"):
        assert field in outcome
    assert outcome["retrieved_text"].strip()


def test_real_retriever_reaches_retrieved_through_retrieve_section_evidence(caplog):
    """C1 regression: real retriever + production call path -> RETRIEVED.

    Uses the compliance/NPA query, which the interim IRAC corpus genuinely
    contains, so RETRIEVED is the correct expectation. Pre-fix this returned
    NOT_FOUND with a swallowed TypeError.
    """
    import logging

    from app.report.generate import SECTION_QUERIES, _retrieve_section_evidence

    retriever = IsolatedRAGRetriever()
    try:
        with caplog.at_level(logging.WARNING, logger="app.report.generate"):
            evidence = _retrieve_section_evidence(
                section_key="compliance",
                query=SECTION_QUERIES["compliance"],
                retrieval_fn=retriever.query,
            )
    finally:
        retriever.close()

    # 1. Nothing was swallowed by the broad retrieval fallback.
    assert "Retrieval failed" not in caplog.text
    assert "unexpected keyword argument" not in caplog.text
    assert "TypeError" not in caplog.text

    # 2. The real symptom: NOT_FOUND instead of RETRIEVED.
    assert evidence.evidence_status == "RETRIEVED", (
        "Real retrieval returned NOT_FOUND for a query the interim corpus "
        "contains -- the retrieval_fn call is broken again."
    )

    # 3. Retrieval actually carried usable, attributed content.
    assert evidence.citations
    citation = evidence.citations[0]
    assert "RBI_MASTER_CIRCULAR_IRAC_ADVANCES" in citation.source
    assert citation.locator.startswith("chunk #")
    assert citation.provenance == "interim_single_document"
    assert "non performing" in citation.quote.lower()


def test_generate_report_with_real_retrieval_has_nonzero_coverage(sample_inputs):
    """End-to-end: real retrieval must yield at least one RETRIEVED section.

    `retrieval_fn=None` exercises the production path that builds an
    IsolatedRAGRetriever internally. Pre-fix, coverage was retrieved=0 /
    not_found=5. Asserted as >= 1 rather than == 1 so that broadening the
    corpus later does not make this fail spuriously.
    """
    out = generate_report(
        **sample_inputs,
        llm_client=FakeGroqClient(),
        retrieval_fn=None,
    )

    coverage = out["evidence_coverage"]
    assert coverage["total"] == 5
    assert coverage["retrieved"] + coverage["not_found"] == 5
    assert coverage["retrieved"] >= 1, (
        "No section retrieved evidence through the real retriever; "
        "retrieved=0 was the exact C1 symptom."
    )

    retrieved = [
        s for s in out["sections"]
        if s["retrieved_evidence"]["evidence_status"] == "RETRIEVED"
    ]
    assert retrieved
    for section in retrieved:
        assert section["retrieved_evidence"]["citations"]
        # A RETRIEVED section may claim an illustrative basis; never cited_evidence
        # while the corpus is the interim single document.
        assert section["llm_interpretation"]["regulatory_basis"] == "illustrative_rule_only"


def test_generate_report_custom_provider_label(sample_inputs):
    """Verify passing a custom provider_label updates the disclaimer text accordingly."""
    fake_client = FakeGroqClient()
    custom_label = "Local HuggingFace (meta-llama/Llama-3-8B-Instruct)"
    result = generate_report(
        **sample_inputs,
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_mixed,
        provider_label=custom_label,
    )
    parsed = ReportResult(**result)
    assert any(
        f"LLM-generated report text produced via {custom_label}." in d
        for d in parsed.disclaimers
    )
    assert not any("produced via Groq" in d for d in parsed.disclaimers)


def test_generate_report_default_provider_label(sample_inputs):
    """Verify default provider_label produces the standard Groq disclaimer."""
    fake_client = FakeGroqClient()
    result = generate_report(
        **sample_inputs,
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_mixed,
    )
    parsed = ReportResult(**result)
    expected_disclaimer = (
        f"LLM-generated report text produced via {DEFAULT_PROVIDER_LABEL}. "
        "Technical findings are calculated by Python analytical modules and passed verbatim."
    )
    assert expected_disclaimer in parsed.disclaimers
    assert "produced via Groq (openai/gpt-oss-120b)" in expected_disclaimer

