"""Unit tests for API mock data fixtures (Owner: Khushi)."""
from app.api.mock_data import (
    MOCK_ASSURANCE_RESULT,
    MOCK_COMPLIANCE_RESULT,
    MOCK_DRIFT_RESULT,
    MOCK_EXPLAINABILITY_RESULT,
    MOCK_EXPLAINABILITY_RESULT_LIME,
    MOCK_EXPLAINABILITY_RESULT_SHAP,
    MOCK_FAIRNESS_RESULT,
    MOCK_MODEL_RESULT,
    MOCK_REPORT_RESULT,
)
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    DriftResult,
    ExplainabilityResult,
    FairnessResult,
    ModelResult,
    ReportResult,
)


def test_mock_model_result_validity_and_shape():
    parsed = ModelResult(**MOCK_MODEL_RESULT)
    assert parsed.is_mock is True
    assert len(parsed.predictions) == 3
    assert len(parsed.probabilities) == 3
    assert len(MOCK_MODEL_RESULT["instance_ids"]) == len(MOCK_MODEL_RESULT["predictions"])
    assert len(parsed.instance_ids) == len(parsed.predictions)
    assert all(isinstance(iid, str) for iid in parsed.instance_ids)
    assert parsed.instance_ids == ["gc-0000", "gc-0001", "gc-0002"]
    assert len(parsed.feature_matrix) == 3
    assert len(parsed.model_metadata.feature_names) == 20
    assert "personal_status_and_sex" in parsed.model_metadata.feature_names
    for row in parsed.feature_matrix:
        assert len(row) == 20
        assert "personal_status_and_sex" in row


def test_mock_explainability_shap_validity_and_shape():
    parsed = ExplainabilityResult(**MOCK_EXPLAINABILITY_RESULT_SHAP)
    assert parsed.is_mock is True
    assert parsed.method == "shap"
    assert len(parsed.per_instance) == 3
    for inst in parsed.per_instance:
        assert len(inst.contributions) == 20
        assert "duration_months" in inst.contributions
    assert len(parsed.global_importance) == 20
    assert "duration_months" in parsed.global_importance


def test_mock_explainability_lime_validity_and_shape():
    parsed = ExplainabilityResult(**MOCK_EXPLAINABILITY_RESULT_LIME)
    assert parsed.is_mock is True
    assert parsed.method == "lime"
    assert len(parsed.per_instance) == 3
    for inst in parsed.per_instance:
        assert len(inst.contributions) == 20
        assert "duration_months" in inst.contributions
    assert len(parsed.global_importance) == 20
    assert "duration_months" in parsed.global_importance


def test_mock_fairness_result_validity():
    parsed = FairnessResult(**MOCK_FAIRNESS_RESULT)
    assert parsed.is_mock is True
    assert parsed.protected_attribute == "personal_status_and_sex"
    assert parsed.status in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_mock_drift_result_validity():
    parsed = DriftResult(**MOCK_DRIFT_RESULT)
    assert parsed.is_mock is True
    assert parsed.features_evaluated == [
        "duration_months",
        "credit_amount",
        "installment_rate",
        "present_residence",
        "age",
        "existing_credits",
        "num_dependents",
    ]
    assert parsed.status in {"PASS", "WARNING", "FAIL", "PENDING"}
    assert parsed.note is not None


def test_mock_compliance_result_validity():
    parsed = ComplianceResult(**MOCK_COMPLIANCE_RESULT)
    assert parsed.is_mock is True
    assert len(parsed.findings) >= 2
    for finding in parsed.findings:
        assert finding.rule_id.startswith("RBI-")
        assert finding.status in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_mock_assurance_result_validity():
    parsed = AssuranceResult(**MOCK_ASSURANCE_RESULT)
    assert parsed.model.is_mock is True
    assert parsed.model.instance_ids == ["gc-0000", "gc-0001", "gc-0002"]
    assert parsed.explainability.is_mock is True
    assert parsed.fairness_drift.fairness.is_mock is True
    assert parsed.fairness_drift.drift.is_mock is True
    assert parsed.compliance.is_mock is True
    assert "SYNTHETIC" in parsed.note or "MOCK" in parsed.note


def test_mock_report_result_validity_and_contract():
    parsed = ReportResult(**MOCK_REPORT_RESULT)
    assert parsed.is_mock is True
    assert len(parsed.sections) == 5
    assert parsed.evidence_coverage.total == 5
    assert parsed.evidence_coverage.retrieved == 1
    assert parsed.evidence_coverage.not_found == 4

    # Check section headings encompass Model, Explainability, Fairness, Drift, Compliance
    combined_headings = " ".join(s.heading.lower() for s in parsed.sections)
    for domain in ["model", "explain", "fairness", "drift", "compliance"]:
        assert domain in combined_headings

    not_found_sections = [
        s for s in parsed.sections if s.retrieved_evidence.evidence_status == "NOT_FOUND"
    ]
    assert len(not_found_sections) == 4
    for not_found_sec in not_found_sections:
        assert not_found_sec.llm_interpretation.regulatory_basis == "none"
        assert not_found_sec.retrieved_evidence.citations == []

    # The advertised coverage must match the fixture's own sections, so the mock
    # can never again claim more retrieved evidence than it actually carries.
    retrieved_sections = [
        s for s in parsed.sections if s.retrieved_evidence.evidence_status == "RETRIEVED"
    ]
    assert parsed.evidence_coverage.retrieved == len(retrieved_sections)
    assert parsed.evidence_coverage.not_found == len(not_found_sections)
    assert parsed.evidence_coverage.total == len(parsed.sections)

    # Mandatory invariant: NOT_FOUND section must never claim cited_evidence
    for s in parsed.sections:
        if s.retrieved_evidence.evidence_status == "NOT_FOUND":
            assert s.llm_interpretation.regulatory_basis != "cited_evidence"

    # Verify citations use sample / illustrative indicators
    for s in parsed.sections:
        for cit in s.retrieved_evidence.citations:
            assert "ILLUSTRATIVE" in cit.source or "sample" in cit.locator.lower()
            assert cit.provenance == "illustrative"

    # Verify all grounded_in items follow real dotted-path vocabulary (no colons)
    for s in parsed.sections:
        for item in s.llm_interpretation.grounded_in:
            assert ":" not in item, f"Malformed grounded_in item contains colon: {item}"
            assert "." in item, f"Expected dotted path in grounded_in item: {item}"

    # Verify fairness text mentions internal convention rather than RBI mandate
    fairness_sec = next(s for s in parsed.sections if "fairness" in s.heading.lower())
    assert "internal convention" in fairness_sec.llm_interpretation.text.lower()

