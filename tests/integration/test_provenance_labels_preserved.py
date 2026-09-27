"""Integration tests verifying provenance and is_mock labels are preserved across all layers."""
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
from app.report.generate import generate_report
from tests.report.test_generate_report import (
    FakeGroqClient,
    fake_retrieval_all_retrieved,
    fake_retrieval_mixed,
)


def test_real_data_stays_is_mock_false(real_pipeline):
    """Verify that real pipeline outputs produce is_mock=False and observed provenance."""
    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=FakeGroqClient(),
        retrieval_fn=fake_retrieval_mixed,
    )

    assert out["is_mock"] is False
    for s in out["sections"]:
        assert s["llm_interpretation"]["is_mock"] is False
        if s["heading"] == "RBI Compliance Rules Mapping":
            assert s["technical_finding"]["provenance"] == "mock"
        else:
            assert s["technical_finding"]["provenance"] == "observed"

    # Also verify the real_pipeline assurance_result per-section is_mock values survive assembly
    assurance = real_pipeline["assurance_result"]
    assert assurance["model"]["is_mock"] is False
    assert assurance["explainability"]["is_mock"] is False
    assert assurance["fairness_drift"]["fairness"]["is_mock"] is False
    assert assurance["fairness_drift"]["drift"]["is_mock"] is False
    assert assurance["compliance"]["is_mock"] is True


def test_mock_fixtures_stay_is_mock_true():
    """Verify that passing mock data results in mock technical finding provenance."""
    out = generate_report(
        model=MOCK_MODEL_RESULT,
        explainability=MOCK_EXPLAINABILITY_RESULT,
        fairness=MOCK_FAIRNESS_RESULT,
        drift=MOCK_DRIFT_RESULT,
        compliance=MOCK_COMPLIANCE_RESULT,
        llm_client=FakeGroqClient(),
        retrieval_fn=fake_retrieval_mixed,
    )

    for s in out["sections"]:
        assert s["technical_finding"]["provenance"] == "mock"


def test_report_without_a_key_labels_its_provenance_honestly(monkeypatch):
    """Absent an LLM provider, provenance must still say what the data really is.

    This used to assert the endpoint returned the stored fixture with
    ``is_mock: True``. That fixture's disclaimers described a single 2014
    excerpt and its coverage read 1 of 5, contradicting the PDF generated from
    the same page. The report is now produced for real and omits only the
    narrative layer.

    The invariant this file exists to protect is unchanged and is what is
    checked here: a label never overstates or understates the data behind it.
    Real analytical output must be labelled ``observed``, and a report built
    from real output must not call itself mock.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(app)
    response = client.get("/report")
    assert response.status_code == 200

    body = response.json()
    validated = ReportResult(**body)
    assert validated is not None

    # Real analytical findings -> honest provenance, and not labelled mock.
    assert body["is_mock"] is False

    provenance = {
        section["heading"]: section["technical_finding"]["provenance"]
        for section in body["sections"]
    }

    # The four ANALYTICAL sections are computed from real model output.
    for heading, label in provenance.items():
        if "Compliance" in heading:
            continue
        assert label == "observed", f"{heading} should be observed, got {label}"

    # The COMPLIANCE section is legitimately "mock", and that is the point of
    # this file: the six rules it maps are illustrative sample rules, so the
    # section must not borrow the "observed" label from the real measurements
    # those rules read. A real input does not make an invented rule real.
    compliance_labels = [v for k, v in provenance.items() if "Compliance" in k]
    assert compliance_labels == ["mock"], compliance_labels

    # The narrative layer is absent rather than invented, in every section.
    for section in body["sections"]:
        assert section["llm_interpretation"] is None

    # And the absence is stated rather than left for the reader to infer.
    assert any("NO LLM NARRATIVE" in d for d in body["disclaimers"])


def test_citation_provenance_interim_multi_document(real_pipeline):
    """Verify that retrieved citations carry interim_multi_document provenance and illustrative_rule_only basis."""
    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=FakeGroqClient(),
        retrieval_fn=fake_retrieval_all_retrieved,
    )

    assert len(out["sections"]) == 5
    for s in out["sections"]:
        ev = s["retrieved_evidence"]
        interp = s["llm_interpretation"]
        assert ev["evidence_status"] == "RETRIEVED"
        assert len(ev["citations"]) > 0
        assert ev["citations"][0]["provenance"] == "interim_multi_document"
        assert interp["regulatory_basis"] == "illustrative_rule_only"
        assert interp["regulatory_basis"] != "cited_evidence"


def test_dashboard_fallback_preserves_labels(monkeypatch):
    """Verify Streamlit dashboard API client falls back to mock fixture on unreachable backend."""
    import dashboard.api_client
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(dashboard.api_client, "API_BASE_URL", "http://127.0.0.1:1")

    data, source = dashboard.api_client.get_report()
    assert source == "fallback"
    assert data["is_mock"] is True
