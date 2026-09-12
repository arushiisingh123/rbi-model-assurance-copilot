"""Unit tests for API Pydantic schemas (Owner: Khushi)."""
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AssuranceResult,
    Citation,
    ComplianceFinding,
    ComplianceResult,
    DriftPerFeature,
    DriftResult,
    EvidenceCoverage,
    ExplainabilityResult,
    FairnessDriftResult,
    FairnessGroup,
    FairnessResult,
    LabelSemantics,
    LLMInterpretation,
    ModelMetadata,
    ModelMetrics,
    ModelResult,
    PerInstanceContribution,
    ReportResult,
    ReportSection,
    RetrievedEvidence,
    TechnicalFinding,
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


def test_model_metrics_valid():
    metrics = ModelMetrics(
        accuracy=0.85,
        precision=0.80,
        recall=0.75,
        f1=0.77,
        roc_auc=0.91,
        n_test_samples=200,
        is_mock=False,
    )
    assert metrics.accuracy == 0.85
    assert metrics.precision == 0.80
    assert metrics.recall == 0.75
    assert metrics.f1 == 0.77
    assert metrics.roc_auc == 0.91
    assert metrics.n_test_samples == 200
    assert metrics.is_mock is False


def test_model_metrics_missing_field():
    with pytest.raises(ValidationError):
        ModelMetrics(
            accuracy=0.85,
            precision=0.80,
            # missing recall, f1, roc_auc, n_test_samples, is_mock
        )


def test_model_metrics_invalid_type():
    with pytest.raises(ValidationError):
        ModelMetrics(
            accuracy="not-a-float",
            precision=0.80,
            recall=0.75,
            f1=0.77,
            roc_auc=0.91,
            n_test_samples=200,
            is_mock=False,
        )


def test_model_result_with_model_metrics():
    res = ModelResult(
        predictions=[0, 1],
        probabilities=[0.12, 0.81],
        instance_ids=["gc-0000", "gc-0001"],
        feature_matrix=[
            {"income": 45000, "age": 34},
            {"income": 120000, "age": 45},
        ],
        model_metadata=ModelMetadata(
            model_type="xgboost",
            version="0.1.0",
            trained_on="data/sample/credit_sample.csv",
            feature_names=["income", "age"],
        ),
        is_mock=False,
        model_metrics=ModelMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            roc_auc=0.91,
            n_test_samples=200,
            is_mock=False,
        ),
    )
    assert res.model_metrics is not None
    assert res.model_metrics.accuracy == 0.85
    assert res.model_metrics.n_test_samples == 200


def test_model_result_malformed_model_metrics():
    with pytest.raises(ValidationError):
        ModelResult(
            predictions=[0, 1],
            probabilities=[0.12, 0.81],
            instance_ids=["gc-0000", "gc-0001"],
            feature_matrix=[],
            model_metadata=ModelMetadata(
                model_type="xgboost",
                version="0.1.0",
                trained_on="data/sample/credit_sample.csv",
                feature_names=[],
            ),
            is_mock=False,
            model_metrics={"accuracy": "not-a-number"},
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


def test_fairness_group_valid():
    group = FairnessGroup(
        group="A92",
        count=310,
        favorable_count=201,
        selection_rate=0.6484,
    )
    assert group.group == "A92"
    assert group.count == 310
    assert group.favorable_count == 201
    assert group.selection_rate == 0.6484


def test_fairness_group_missing_key():
    with pytest.raises(ValidationError):
        FairnessGroup(
            group="A92",
            count=310,
            favorable_count=201,
            # missing selection_rate
        )


def test_fairness_group_invalid_type():
    with pytest.raises(ValidationError):
        FairnessGroup(
            group="A92",
            count="not_an_int",
            favorable_count=201,
            selection_rate=0.6484,
        )


def test_fairness_result_with_groups():
    res = FairnessResult(
        protected_attribute="personal_status_and_sex",
        demographic_parity_diff=0.14,
        disparate_impact_ratio=0.78,
        status="WARNING",
        is_mock=True,
        groups=[
            FairnessGroup(group="A92", count=310, favorable_count=201, selection_rate=0.65),
            {"group": "A93", "count": 548, "favorable_count": 432, "selection_rate": 0.79},
        ],
    )
    assert len(res.groups) == 2
    assert res.groups[0].group == "A92"
    assert res.groups[1].group == "A93"
    assert isinstance(res.groups[1], FairnessGroup)


def test_fairness_result_malformed_group_entry():
    with pytest.raises(ValidationError):
        FairnessResult(
            protected_attribute="personal_status_and_sex",
            demographic_parity_diff=0.14,
            disparate_impact_ratio=0.78,
            status="WARNING",
            is_mock=True,
            groups=[
                {"group": "A92"},  # missing required count, favorable_count, selection_rate
            ],
        )


def test_drift_per_feature_valid():
    dpf = DriftPerFeature(
        feature="duration_months",
        psi=0.08,
        ks_statistic=0.10,
    )
    assert dpf.feature == "duration_months"
    assert dpf.psi == 0.08
    assert dpf.ks_statistic == 0.10


def test_drift_per_feature_missing_key():
    with pytest.raises(ValidationError):
        DriftPerFeature(
            feature="duration_months",
            psi=0.08,
            # missing ks_statistic
        )


def test_drift_per_feature_invalid_type():
    with pytest.raises(ValidationError):
        DriftPerFeature(
            feature="duration_months",
            psi="not_a_float",
            ks_statistic=0.10,
        )


def test_drift_result_with_per_feature():
    res = DriftResult(
        features_evaluated=["duration_months", "credit_amount"],
        psi=0.09,
        ks_statistic=0.11,
        status="PASS",
        is_mock=True,
        per_feature=[
            DriftPerFeature(feature="duration_months", psi=0.08, ks_statistic=0.10),
            {"feature": "credit_amount", "psi": 0.09, "ks_statistic": 0.11},
        ],
    )
    assert len(res.per_feature) == 2
    assert res.per_feature[0].feature == "duration_months"
    assert res.per_feature[1].feature == "credit_amount"
    assert isinstance(res.per_feature[1], DriftPerFeature)


def test_drift_result_malformed_per_feature_entry():
    with pytest.raises(ValidationError):
        DriftResult(
            features_evaluated=["duration_months"],
            psi=0.09,
            ks_statistic=0.11,
            status="PASS",
            is_mock=True,
            per_feature=[
                {"feature": "duration_months"},  # missing psi and ks_statistic
            ],
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


def test_report_result_schema_valid():
    res = ReportResult(
        report_id="rep-001",
        generated_at="2026-09-08T12:00:00Z",
        model_version="0.1.0",
        sections=[
            ReportSection(
                heading="Fairness Section",
                technical_finding=TechnicalFinding(
                    ref="fairness.disparate_impact_ratio",
                    value=0.78,
                    status="WARNING",
                    source_module="app.fairness",
                    provenance="synthetic_fixture",
                ),
                retrieved_evidence=RetrievedEvidence(
                    evidence_status="RETRIEVED",
                    citations=[
                        Citation(
                            source="ILLUSTRATIVE — not a real RBI source",
                            locator="§2 (sample)",
                            quote="[sample placeholder] Disparate impact below 0.80 requires review.",
                            provenance="illustrative",
                        )
                    ],
                ),
                llm_interpretation=LLMInterpretation(
                    text="Disparate impact ratio 0.78 requires mitigation plan.",
                    grounded_in=["fairness.disparate_impact_ratio"],
                    regulatory_basis="illustrative_rule_only",
                    is_mock=True,
                ),
            )
        ],
        disclaimers=["Sample disclaimer."],
        evidence_coverage=EvidenceCoverage(retrieved=1, not_found=0, total=1),
        is_mock=True,
    )
    assert res.report_id == "rep-001"
    assert len(res.sections) == 1
    assert res.sections[0].retrieved_evidence.evidence_status == "RETRIEVED"
    assert res.evidence_coverage.total == 1


def test_report_result_invalid_evidence_status_raises():
    with pytest.raises(ValidationError):
        RetrievedEvidence(evidence_status="UNKNOWN_STATUS")


def test_report_result_invalid_regulatory_basis_raises():
    with pytest.raises(ValidationError):
        LLMInterpretation(
            text="Interpretation",
            grounded_in=["fairness.status"],
            regulatory_basis="hallucinated_basis",
            is_mock=True,
        )


def test_report_result_invalid_citation_provenance_raises():
    with pytest.raises(ValidationError):
        Citation(
            source="Source",
            locator="Locator",
            quote="Quote",
            provenance="fabricated",
        )


def test_report_result_invalid_technical_finding_provenance_raises():
    with pytest.raises(ValidationError):
        TechnicalFinding(
            ref="fairness.status",
            value="PASS",
            status="PASS",
            source_module="app.fairness",
            provenance="unrecognized_provenance",
        )



