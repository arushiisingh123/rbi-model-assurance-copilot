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
