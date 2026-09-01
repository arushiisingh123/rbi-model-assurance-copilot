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
)
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    DriftResult,
    ExplainabilityResult,
    FairnessResult,
    ModelResult,
)


def test_mock_model_result_validity_and_shape():
    parsed = ModelResult(**MOCK_MODEL_RESULT)
    assert parsed.is_mock is True
    assert len(parsed.predictions) == 3
    assert len(parsed.probabilities) == 3
    assert len(parsed.feature_matrix) == 3
    assert parsed.model_metadata.feature_names == ["income", "age", "credit_history_len"]
    for row in parsed.feature_matrix:
        assert set(row.keys()) == {"income", "age", "credit_history_len"}


def test_mock_explainability_shap_validity_and_shape():
    parsed = ExplainabilityResult(**MOCK_EXPLAINABILITY_RESULT_SHAP)
    assert parsed.is_mock is True
    assert parsed.method == "shap"
    assert len(parsed.per_instance) == 3
    for inst in parsed.per_instance:
        assert set(inst.contributions.keys()) == {"income", "age", "credit_history_len"}
    assert set(parsed.global_importance.keys()) == {"income", "age", "credit_history_len"}


def test_mock_explainability_lime_validity_and_shape():
    parsed = ExplainabilityResult(**MOCK_EXPLAINABILITY_RESULT_LIME)
    assert parsed.is_mock is True
    assert parsed.method == "lime"
    assert len(parsed.per_instance) == 3
    for inst in parsed.per_instance:
        assert set(inst.contributions.keys()) == {"income", "age", "credit_history_len"}
    assert set(parsed.global_importance.keys()) == {"income", "age", "credit_history_len"}


def test_mock_fairness_result_validity():
    parsed = FairnessResult(**MOCK_FAIRNESS_RESULT)
    assert parsed.is_mock is True
    assert parsed.protected_attribute == "gender"
    assert parsed.status in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_mock_drift_result_validity():
    parsed = DriftResult(**MOCK_DRIFT_RESULT)
    assert parsed.is_mock is True
    assert parsed.features_evaluated == ["income", "age", "credit_history_len"]
    assert parsed.status in {"PASS", "WARNING", "FAIL", "PENDING"}


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
    assert parsed.explainability.is_mock is True
    assert parsed.fairness_drift.fairness.is_mock is True
    assert parsed.fairness_drift.drift.is_mock is True
    assert parsed.compliance.is_mock is True
    assert "SYNTHETIC" in parsed.note or "MOCK" in parsed.note
