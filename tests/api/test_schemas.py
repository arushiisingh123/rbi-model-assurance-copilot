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
    LabelSemantics,
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
        instance_ids=["gc-0000", "gc-0001", "gc-0002"],
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
    assert len(res.instance_ids) == 3
    assert res.instance_ids == ["gc-0000", "gc-0001", "gc-0002"]


def test_model_result_missing_field():
    with pytest.raises(ValidationError) as exc_info:
        ModelResult(
            predictions=[0, 1],
            probabilities=[0.12, 0.81],
            # missing instance_ids, feature_matrix, model_metadata, is_mock
        )
    missing_fields = {e["loc"][0] for e in exc_info.value.errors()}
    assert "instance_ids" in missing_fields


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
            instance_ids=["gc-0000", "gc-0001", "gc-0002"],
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


def test_model_metadata_with_label_semantics():
    sem = LabelSemantics(
        **{
            "0": "GOOD - low credit risk",
            "1": "BAD - high credit risk / likely default",
            "positive_class": 1,
            "probabilities_represent": "P(class == 1) = P(BAD / high credit risk)",
            "favorable_outcome_label": 0,
        }
    )
    meta = ModelMetadata(
        model_type="logistic_regression",
        version="0.1.0",
        trained_on="data/german_credit/german_credit.csv",
        feature_names=["duration_months", "credit_amount"],
        label_semantics=sem,
    )
    assert meta.label_semantics is not None
    assert meta.label_semantics.favorable_outcome_label == 0
    assert meta.label_semantics.positive_class == 1
    dumped = meta.model_dump(by_alias=True)
    assert dumped["label_semantics"]["0"] == "GOOD - low credit risk"
    assert dumped["label_semantics"]["1"] == "BAD - high credit risk / likely default"


def test_drift_result_with_note():
    res = DriftResult(
        features_evaluated=["duration_months", "credit_amount"],
        psi=0.09,
        ks_statistic=0.11,
        status="PASS",
        is_mock=False,
        note="SYNTHETIC DRIFT SCENARIO: Controlled shift.",
    )
    assert res.note == "SYNTHETIC DRIFT SCENARIO: Controlled shift."
    assert res.is_mock is False


def test_model_result_instance_ids_valid_dict():
    payload = {
        "predictions": [0, 1, 0],
        "probabilities": [0.12, 0.81, 0.33],
        "instance_ids": ["gc-0000", "gc-0001", "gc-0002"],
        "feature_matrix": [{"col": 1}, {"col": 2}, {"col": 3}],
        "model_metadata": {
            "model_type": "logistic_regression",
            "version": "0.1.0",
            "trained_on": "data/german_credit/german_credit.csv",
            "feature_names": ["col"],
        },
        "is_mock": False,
    }
    result = ModelResult(**payload)
    assert result.instance_ids == ["gc-0000", "gc-0001", "gc-0002"]


def test_model_result_missing_instance_ids_raises():
    payload = {
        "predictions": [0, 1, 0],
        "probabilities": [0.12, 0.81, 0.33],
        # instance_ids omitted
        "feature_matrix": [{"col": 1}],
        "model_metadata": {
            "model_type": "logistic_regression",
            "version": "0.1.0",
            "trained_on": "data/german_credit/german_credit.csv",
            "feature_names": ["col"],
        },
        "is_mock": False,
    }
    with pytest.raises(ValidationError) as exc_info:
        ModelResult(**payload)
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("instance_ids",) and e["type"] == "missing" for e in errors)


def test_model_result_instance_ids_non_list_raises():
    payload = {
        "predictions": [0, 1, 0],
        "probabilities": [0.12, 0.81, 0.33],
        "instance_ids": "gc-0000",  # string instead of list[str]
        "feature_matrix": [{"col": 1}],
        "model_metadata": {
            "model_type": "logistic_regression",
            "version": "0.1.0",
            "trained_on": "data/german_credit/german_credit.csv",
            "feature_names": ["col"],
        },
        "is_mock": False,
    }
    with pytest.raises(ValidationError):
        ModelResult(**payload)


def test_model_result_instance_ids_non_string_elements_raises():
    payload = {
        "predictions": [0, 1, 0],
        "probabilities": [0.12, 0.81, 0.33],
        "instance_ids": [1, 2, 3],  # integers instead of strings
        "feature_matrix": [{"col": 1}],
        "model_metadata": {
            "model_type": "logistic_regression",
            "version": "0.1.0",
            "trained_on": "data/german_credit/german_credit.csv",
            "feature_names": ["col"],
        },
        "is_mock": False,
    }
    with pytest.raises(ValidationError):
        ModelResult(**payload)

