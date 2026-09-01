"""Unit tests for API Pydantic schemas (Owner: Khushi)."""
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AssuranceResult,
    ComplianceFinding,
    ComplianceResult,
    DriftResult,
    ExplainabilityResult,
    FairnessDriftResult,
    FairnessResult,
    ModelMetadata,
    ModelResult,
    PerInstanceContribution,
)


def test_model_metadata_valid():
    meta = ModelMetadata(
        model_type="xgboost",
        version="0.1.0",
        trained_on="data/sample/credit_sample.csv",
        feature_names=["income", "age", "credit_history_len"],
    )
    assert meta.model_type == "xgboost"
    assert len(meta.feature_names) == 3


def test_model_metadata_missing_field():
    with pytest.raises(ValidationError):
        ModelMetadata(
            model_type="xgboost",
            version="0.1.0",
            # missing trained_on and feature_names
        )


def test_model_result_valid():
    res = ModelResult(
        predictions=[0, 1, 0],
        probabilities=[0.12, 0.81, 0.33],
        feature_matrix=[
            {"income": 45000, "age": 34, "credit_history_len": 5},
            {"income": 120000, "age": 45, "credit_history_len": 12},
            {"income": 28000, "age": 22, "credit_history_len": 2},
        ],
        model_metadata=ModelMetadata(
            model_type="xgboost",
            version="0.1.0",
            trained_on="data/sample/credit_sample.csv",
            feature_names=["income", "age", "credit_history_len"],
        ),
        is_mock=True,
    )
    assert res.is_mock is True
    assert len(res.predictions) == 3
    assert len(res.feature_matrix) == 3


def test_model_result_missing_field():
    with pytest.raises(ValidationError):
        ModelResult(
            predictions=[0, 1],
            probabilities=[0.12, 0.81],
            # missing feature_matrix, model_metadata, is_mock
        )


def test_per_instance_contribution_valid():
    contrib = PerInstanceContribution(
        row_index=0,
        contributions={"income": 0.31, "age": 0.12},
    )
    assert contrib.row_index == 0
    assert contrib.contributions["income"] == 0.31


def test_explainability_result_valid():
    res = ExplainabilityResult(
        method="shap",
        per_instance=[
            PerInstanceContribution(row_index=0, contributions={"income": 0.31}),
        ],
        global_importance={"income": 0.42},
        is_mock=True,
    )
    assert res.method == "shap"
    assert res.is_mock is True


def test_explainability_result_invalid_method():
    with pytest.raises(ValidationError):
        ExplainabilityResult(
            method="invalid_method",
            per_instance=[],
            global_importance={},
            is_mock=True,
        )


def test_fairness_result_valid():
    res = FairnessResult(
        protected_attribute="gender",
        demographic_parity_diff=0.14,
        disparate_impact_ratio=0.78,
        status="WARNING",
        is_mock=True,
    )
    assert res.status == "WARNING"
    assert res.is_mock is True


def test_fairness_result_invalid_status():
    with pytest.raises(ValidationError):
        FairnessResult(
            protected_attribute="gender",
            demographic_parity_diff=0.14,
            disparate_impact_ratio=0.78,
            status="INVALID_STATUS",
            is_mock=True,
        )


def test_drift_result_valid():
    res = DriftResult(
        features_evaluated=["income", "age", "credit_history_len"],
        psi=0.09,
        ks_statistic=0.11,
        status="PASS",
        is_mock=True,
    )
    assert res.status == "PASS"
    assert res.psi == 0.09


def test_drift_result_invalid_status():
    with pytest.raises(ValidationError):
        DriftResult(
            features_evaluated=["income"],
            psi=0.09,
            ks_statistic=0.11,
            status="UNKNOWN",
            is_mock=True,
        )


def test_fairness_drift_result_valid():
    fd = FairnessDriftResult(
        fairness=FairnessResult(
            protected_attribute="gender",
            demographic_parity_diff=0.14,
            disparate_impact_ratio=0.78,
            status="WARNING",
            is_mock=True,
        ),
        drift=DriftResult(
            features_evaluated=["income", "age", "credit_history_len"],
            psi=0.09,
            ks_statistic=0.11,
            status="PASS",
            is_mock=True,
        ),
    )
    assert fd.fairness.status == "WARNING"
    assert fd.drift.status == "PASS"


def test_compliance_finding_valid():
    finding = ComplianceFinding(
        rule_id="RBI-FAIR-01",
        rule_description="Test rule",
        technical_finding_ref="fairness.disparate_impact_ratio",
        status="FAIL",
        evidence_chunks=[],
    )
    assert finding.rule_id == "RBI-FAIR-01"
    assert finding.status == "FAIL"
    assert finding.evidence_chunks == []


def test_compliance_finding_invalid_status():
    with pytest.raises(ValidationError):
        ComplianceFinding(
            rule_id="RBI-FAIR-01",
            rule_description="Test rule",
            technical_finding_ref="fairness.disparate_impact_ratio",
            status="NOT_A_STATUS",
        )


def test_compliance_result_valid():
    res = ComplianceResult(
        findings=[
            ComplianceFinding(
                rule_id="RBI-FAIR-01",
                rule_description="Test rule",
                technical_finding_ref="fairness.disparate_impact_ratio",
                status="FAIL",
            )
        ],
        is_mock=True,
    )
    assert len(res.findings) == 1
    assert res.is_mock is True


def test_assurance_result_valid():
    assurance = AssuranceResult(
        model=ModelResult(
            predictions=[0, 1, 0],
            probabilities=[0.12, 0.81, 0.33],
            feature_matrix=[{"income": 45000}],
            model_metadata=ModelMetadata(
                model_type="xgboost",
                version="0.1.0",
                trained_on="data/sample/credit_sample.csv",
                feature_names=["income"],
            ),
            is_mock=True,
        ),
        explainability=ExplainabilityResult(
            method="shap",
            per_instance=[PerInstanceContribution(row_index=0, contributions={"income": 0.31})],
            global_importance={"income": 0.42},
            is_mock=True,
        ),
        fairness_drift=FairnessDriftResult(
            fairness=FairnessResult(
                protected_attribute="gender",
                demographic_parity_diff=0.14,
                disparate_impact_ratio=0.78,
                status="WARNING",
                is_mock=True,
            ),
            drift=DriftResult(
                features_evaluated=["income"],
                psi=0.09,
                ks_statistic=0.11,
                status="PASS",
                is_mock=True,
            ),
        ),
        compliance=ComplianceResult(
            findings=[
                ComplianceFinding(
                    rule_id="RBI-FAIR-01",
                    rule_description="Test rule",
                    technical_finding_ref="fairness.disparate_impact_ratio",
                    status="FAIL",
                )
            ],
            is_mock=True,
        ),
        note="SYNTHETIC / MOCK DATA.",
    )
    assert assurance.model.is_mock is True
    assert assurance.explainability.method == "shap"
    assert assurance.fairness_drift.fairness.status == "WARNING"
    assert assurance.compliance.findings[0].rule_id == "RBI-FAIR-01"
    assert "SYNTHETIC" in assurance.note
