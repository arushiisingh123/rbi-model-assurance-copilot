"""Synthetic Mock Data Fixtures for Phase 1 API (Owner: Khushi).

===============================================================================
WARNING: SYNTHETIC / MOCK DATA ONLY — NOT REAL OUTPUT — NOT FOR PRODUCTION USE
All data fixtures defined in this module are synthetic placeholders designed
to validate API endpoint shapes, schemas, and serialization contracts during
Phase 1 independent module development and serve as fallback responses.
===============================================================================
"""

MOCK_FEATURE_NAMES = [
    "status_checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "present_employment",
    "installment_rate",
    "personal_status_and_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
]

MOCK_MODEL_RESULT = {
    "predictions": [0, 1, 0],
    "probabilities": [0.12, 0.81, 0.33],
    "instance_ids": ["gc-0000", "gc-0001", "gc-0002"],
    "feature_matrix": [
        {
            "status_checking_account": "A11",
            "duration_months": 6,
            "credit_history": "A34",
            "purpose": "A43",
            "credit_amount": 1169,
            "savings_account": "A65",
            "present_employment": "A75",
            "installment_rate": 4,
            "personal_status_and_sex": "A93",
            "other_debtors": "A101",
            "present_residence": 4,
            "property": "A121",
            "age": 67,
            "other_installment_plans": "A143",
            "housing": "A152",
            "existing_credits": 2,
            "job": "A173",
            "num_dependents": 1,
            "telephone": "A192",
            "foreign_worker": "A201",
        },
        {
            "status_checking_account": "A12",
            "duration_months": 48,
            "credit_history": "A32",
            "purpose": "A43",
            "credit_amount": 5951,
            "savings_account": "A61",
            "present_employment": "A73",
            "installment_rate": 2,
            "personal_status_and_sex": "A92",
            "other_debtors": "A101",
            "present_residence": 2,
            "property": "A121",
            "age": 22,
            "other_installment_plans": "A143",
            "housing": "A152",
            "existing_credits": 1,
            "job": "A173",
            "num_dependents": 1,
            "telephone": "A191",
            "foreign_worker": "A201",
        },
        {
            "status_checking_account": "A14",
            "duration_months": 12,
            "credit_history": "A34",
            "purpose": "A46",
            "credit_amount": 2096,
            "savings_account": "A61",
            "present_employment": "A74",
            "installment_rate": 2,
            "personal_status_and_sex": "A93",
            "other_debtors": "A101",
            "present_residence": 3,
            "property": "A121",
            "age": 49,
            "other_installment_plans": "A143",
            "housing": "A152",
            "existing_credits": 1,
            "job": "A172",
            "num_dependents": 2,
            "telephone": "A191",
            "foreign_worker": "A201",
        },
    ],
    "model_metadata": {
        "model_type": "logistic_regression",
        "version": "0.1.0",
        "trained_on": "data/german_credit/german_credit.csv",
        "feature_names": MOCK_FEATURE_NAMES,
        "label_semantics": {
            "0": "GOOD - low credit risk",
            "1": "BAD - high credit risk / likely default",
            "positive_class": 1,
            "probabilities_represent": "P(class == 1) = P(BAD / high credit risk)",
            "favorable_outcome_label": 0,
        },
    },
    "is_mock": True,
}

MOCK_EXPLAINABILITY_RESULT_SHAP = {
    "method": "shap",
    "per_instance": [
        {
            "row_index": 0,
            "contributions": {
                "status_checking_account": 0.42,
                "duration_months": 0.31,
                "credit_history": 0.15,
                "purpose": 0.08,
                "credit_amount": -0.22,
                "savings_account": 0.19,
                "present_employment": 0.11,
                "installment_rate": -0.05,
                "personal_status_and_sex": -0.07,
                "other_debtors": 0.03,
                "present_residence": 0.01,
                "property": -0.04,
                "age": 0.12,
                "other_installment_plans": -0.06,
                "housing": 0.05,
                "existing_credits": 0.02,
                "job": -0.01,
                "num_dependents": 0.01,
                "telephone": 0.02,
                "foreign_worker": -0.08,
            },
        },
        {
            "row_index": 1,
            "contributions": {
                "status_checking_account": -0.35,
                "duration_months": -0.22,
                "credit_history": -0.10,
                "purpose": 0.05,
                "credit_amount": 0.18,
                "savings_account": -0.15,
                "present_employment": 0.07,
                "installment_rate": 0.04,
                "personal_status_and_sex": 0.06,
                "other_debtors": -0.02,
                "present_residence": 0.03,
                "property": 0.05,
                "age": 0.05,
                "other_installment_plans": 0.03,
                "housing": -0.04,
                "existing_credits": 0.01,
                "job": 0.02,
                "num_dependents": -0.01,
                "telephone": -0.03,
                "foreign_worker": 0.04,
            },
        },
        {
            "row_index": 2,
            "contributions": {
                "status_checking_account": 0.28,
                "duration_months": 0.40,
                "credit_history": 0.12,
                "purpose": -0.06,
                "credit_amount": -0.10,
                "savings_account": 0.14,
                "present_employment": -0.08,
                "installment_rate": -0.03,
                "personal_status_and_sex": -0.05,
                "other_debtors": 0.01,
                "present_residence": -0.02,
                "property": -0.03,
                "age": -0.15,
                "other_installment_plans": -0.04,
                "housing": 0.02,
                "existing_credits": -0.01,
                "job": -0.02,
                "num_dependents": 0.02,
                "telephone": 0.01,
                "foreign_worker": -0.05,
            },
        },
    ],
    "global_importance": {
        "status_checking_account": 0.45,
        "duration_months": 0.38,
        "credit_amount": 0.35,
        "savings_account": 0.28,
        "credit_history": 0.24,
        "age": 0.20,
        "present_employment": 0.18,
        "personal_status_and_sex": 0.15,
        "property": 0.12,
        "housing": 0.10,
        "installment_rate": 0.09,
        "purpose": 0.08,
        "other_installment_plans": 0.07,
        "foreign_worker": 0.06,
        "other_debtors": 0.05,
        "existing_credits": 0.04,
        "telephone": 0.03,
        "job": 0.03,
        "present_residence": 0.02,
        "num_dependents": 0.02,
    },
    "is_mock": True,
}

MOCK_EXPLAINABILITY_RESULT_LIME = {
    "method": "lime",
    "per_instance": [
        {
            "row_index": 0,
            "contributions": {
                "status_checking_account": 0.38,
                "duration_months": 0.29,
                "credit_amount": -0.18,
                "savings_account": 0.16,
                "credit_history": 0.14,
                "age": 0.10,
                "present_employment": 0.09,
                "personal_status_and_sex": -0.06,
                "installment_rate": -0.04,
                "property": -0.03,
                "housing": 0.04,
                "purpose": 0.07,
                "other_debtors": 0.02,
                "other_installment_plans": -0.05,
                "foreign_worker": -0.06,
                "existing_credits": 0.02,
                "job": -0.01,
                "num_dependents": 0.01,
                "telephone": 0.02,
                "present_residence": 0.01,
            },
        },
        {
            "row_index": 1,
            "contributions": {
                "status_checking_account": -0.30,
                "duration_months": -0.20,
                "credit_amount": 0.15,
                "savings_account": -0.12,
                "credit_history": -0.09,
                "age": 0.07,
                "present_employment": 0.05,
                "personal_status_and_sex": 0.05,
                "installment_rate": 0.03,
                "property": 0.04,
                "housing": -0.03,
                "purpose": 0.04,
                "other_debtors": -0.02,
                "other_installment_plans": 0.02,
                "foreign_worker": 0.03,
                "existing_credits": 0.01,
                "job": 0.02,
                "num_dependents": -0.01,
                "telephone": -0.02,
                "present_residence": 0.02,
            },
        },
        {
            "row_index": 2,
            "contributions": {
                "status_checking_account": 0.24,
                "duration_months": 0.38,
                "credit_amount": -0.08,
                "savings_account": 0.11,
                "credit_history": 0.10,
                "age": -0.12,
                "present_employment": -0.06,
                "personal_status_and_sex": -0.04,
                "installment_rate": -0.02,
                "property": -0.02,
                "housing": 0.02,
                "purpose": -0.05,
                "other_debtors": 0.01,
                "other_installment_plans": -0.03,
                "foreign_worker": -0.04,
                "existing_credits": -0.01,
                "job": -0.02,
                "num_dependents": 0.01,
                "telephone": 0.01,
                "present_residence": -0.01,
            },
        },
    ],
    "global_importance": {
        "status_checking_account": 0.42,
        "duration_months": 0.39,
        "credit_amount": 0.32,
        "savings_account": 0.25,
        "credit_history": 0.21,
        "age": 0.18,
        "present_employment": 0.15,
        "personal_status_and_sex": 0.13,
        "property": 0.11,
        "housing": 0.09,
        "installment_rate": 0.08,
        "purpose": 0.07,
        "other_installment_plans": 0.06,
        "foreign_worker": 0.05,
        "other_debtors": 0.04,
        "existing_credits": 0.03,
        "telephone": 0.03,
        "job": 0.02,
        "present_residence": 0.02,
        "num_dependents": 0.01,
    },
    "is_mock": True,
}

# Default explainability fixture is SHAP
MOCK_EXPLAINABILITY_RESULT = MOCK_EXPLAINABILITY_RESULT_SHAP

MOCK_FAIRNESS_RESULT = {
    "protected_attribute": "personal_status_and_sex",
    "demographic_parity_diff": 0.14,
    "disparate_impact_ratio": 0.78,
    "status": "WARNING",
    "is_mock": True,
    "groups": [
        {
            "group": "A92",
            "count": 310,
            "favorable_count": 201,
            "selection_rate": 0.65,
        },
        {
            "group": "A93",
            "count": 548,
            "favorable_count": 432,
            "selection_rate": 0.79,
        },
    ],
}

MOCK_DRIFT_RESULT = {
    "features_evaluated": [
        "duration_months",
        "credit_amount",
        "installment_rate",
        "present_residence",
        "age",
        "existing_credits",
        "num_dependents",
    ],
    "psi": 0.09,
    "ks_statistic": 0.11,
    "status": "PASS",
    "is_mock": True,
    "note": (
        "SYNTHETIC DRIFT SCENARIO: Fallback mock scenario for interface validation. "
        "Production assurance evaluates dev training split vs held-out test split."
    ),
    "per_feature": [
        {"feature": "duration_months", "psi": 0.08, "ks_statistic": 0.10},
        {"feature": "credit_amount", "psi": 0.09, "ks_statistic": 0.11},
        {"feature": "installment_rate", "psi": 0.02, "ks_statistic": 0.04},
        {"feature": "present_residence", "psi": 0.01, "ks_statistic": 0.03},
        {"feature": "age", "psi": 0.05, "ks_statistic": 0.07},
        {"feature": "existing_credits", "psi": 0.03, "ks_statistic": 0.05},
        {"feature": "num_dependents", "psi": 0.01, "ks_statistic": 0.02},
    ],
}

MOCK_COMPLIANCE_RESULT = {
    "findings": [
        {
            "rule_id": "RBI-FAIR-01",
            "rule_description": "Assess model decisions for disparate impact and unfair bias across protected groups.",
            "technical_finding_ref": "fairness.status",
            "status": "WARNING",
            "evidence_chunks": [],
        },
        {
            "rule_id": "RBI-FAIR-02",
            "rule_description": "The absolute difference in favourable-outcome rates between protected-attribute groups must be reported for review.",
            "technical_finding_ref": "fairness.demographic_parity_diff",
            "status": "PASS",
            "evidence_chunks": [],
        },
        {
            "rule_id": "RBI-DRIFT-01",
            "rule_description": "Monitor population stability index (PSI) to detect distribution shifts.",
            "technical_finding_ref": "drift.status",
            "status": "PASS",
            "evidence_chunks": [],
        },
        {
            "rule_id": "RBI-DRIFT-02",
            "rule_description": "The KS statistic comparing production inputs to the training distribution must be reported alongside PSI as corroborating drift evidence.",
            "technical_finding_ref": "drift.ks_statistic",
            "status": "PASS",
            "evidence_chunks": [],
        },
        {
            "rule_id": "RBI-EXPL-01",
            "rule_description": "The assurance run must produce a global feature-importance explanation for the model.",
            "technical_finding_ref": "explainability.global_importance",
            "status": "PASS",
            "evidence_chunks": [],
        },
        {
            "rule_id": "RBI-MODEL-01",
            "rule_description": "The assurance run must record model metadata (model type, version, training data reference, feature names).",
            "technical_finding_ref": "model.model_metadata",
            "status": "PASS",
            "evidence_chunks": [],
        },
    ],
    "is_mock": True,
}

MOCK_ASSURANCE_RESULT = {
    "model": MOCK_MODEL_RESULT,
    "explainability": MOCK_EXPLAINABILITY_RESULT,
    "fairness_drift": {
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": MOCK_DRIFT_RESULT,
        "note": (
            "SYNTHETIC DRIFT SCENARIO: Fallback mock scenario for interface validation. "
            "Production assurance compares dev training split (reference) with held-out "
            "test split (current); this is not production monitoring data."
        ),
    },
    "compliance": MOCK_COMPLIANCE_RESULT,
    "note": "SYNTHETIC / MOCK DATA. Not real results. Phase 1 mock fixtures only.",
}

# =============================================================================
# Phase 3 Mock Report Result (PROVISIONAL — pending Nidhi + team sign-off)
# =============================================================================

MOCK_REPORT_RESULT = {
    "report_id": "rep-mock-001",
    "generated_at": "2026-09-08T12:00:00Z",
    "model_version": "0.1.0",
    "sections": [
        {
            "heading": "Credit Scoring Model Evaluation",
            "technical_finding": {
                "ref": "model.model_metadata",
                "value": {
                    "model_type": "logistic_regression",
                    "version": "0.1.0",
                    "predictions": [0, 1, 0],
                    "probabilities": [0.12, 0.81, 0.33],
                },
                "status": "PASS",
                "source_module": "app.models",
                "provenance": "synthetic_fixture",
            },
            "retrieved_evidence": {
                "evidence_status": "NOT_FOUND",
                "citations": [],
            },
            "llm_interpretation": {
                "text": (
                    "Model metadata and baseline probability distributions are recorded by "
                    "the assurance run. No governing RBI regulatory text was retrieved from "
                    "the indexed corpus for model documentation, so no regulatory conclusion "
                    "is drawn for this section."
                ),
                "grounded_in": ["model.model_metadata", "model.version"],
                "regulatory_basis": "none",
                "is_mock": True,
            },
        },
        {
            "heading": "Feature Explainability (SHAP)",
            "technical_finding": {
                "ref": "explainability.global_importance",
                "value": {
                    "method": "shap",
                    "top_feature": "duration_months",
                    "top_importance": 0.35,
                },
                "status": "PASS",
                "source_module": "app.explainability",
                "provenance": "synthetic_fixture",
            },
            "retrieved_evidence": {
                "evidence_status": "NOT_FOUND",
                "citations": [],
            },
            "llm_interpretation": {
                "text": (
                    "SHAP global feature importance identifies duration_months as the primary "
                    "decision driver. No governing RBI regulatory text was retrieved from the "
                    "indexed corpus for model explainability, so no regulatory conclusion is "
                    "drawn for this section."
                ),
                "grounded_in": ["explainability.global_importance"],
                "regulatory_basis": "none",
                "is_mock": True,
            },
        },
        {
            "heading": "Fairness Evaluation",
            "technical_finding": {
                "ref": "fairness.disparate_impact_ratio",
                "value": {
                    "protected_attribute": "personal_status_and_sex",
                    "disparate_impact_ratio": 0.78,
                    "demographic_parity_diff": 0.14,
                },
                "status": "WARNING",
                "source_module": "app.fairness",
                "provenance": "synthetic_fixture",
            },
            "retrieved_evidence": {
                "evidence_status": "NOT_FOUND",
                "citations": [],
            },
            "llm_interpretation": {
                "text": (
                    "Disparate impact ratio of 0.78 for personal_status_and_sex evaluates "
                    "against the project's internal fairness threshold of 0.80 (an internal "
                    "convention, not an RBI-mandated figure), resulting in a WARNING status "
                    "that warrants internal review. No governing RBI regulatory text was "
                    "retrieved from the indexed corpus for fairness, so no regulatory "
                    "conclusion is drawn for this section."
                ),
                "grounded_in": ["fairness.disparate_impact_ratio", "fairness.status"],
                "regulatory_basis": "none",
                "is_mock": True,
            },
        },
        {
            "heading": "Data & Prediction Drift Detection",
            "technical_finding": {
                "ref": "drift.psi",
                "value": {
                    "psi": 0.09,
                    "ks_statistic": 0.11,
                    "features_evaluated": ["duration_months", "credit_amount"],
                },
                "status": "PASS",
                "source_module": "app.drift",
                "provenance": "synthetic_fixture",
            },
            "retrieved_evidence": {
                "evidence_status": "NOT_FOUND",
                "citations": [],
            },
            "llm_interpretation": {
                "text": "Population Stability Index (PSI 0.09) and KS statistic (0.11) show stable distributions. No governing RBI regulatory requirement was retrieved for population stability thresholds.",
                "grounded_in": ["drift.psi", "drift.ks_statistic", "drift.status"],
                "regulatory_basis": "none",
                "is_mock": True,
            },
        },
        {
            "heading": "RBI Compliance Rules Mapping",
            "technical_finding": {
                "ref": "compliance.findings",
                "value": {
                    "total_findings": 6,
                    "evaluated_statuses": ["PASS", "WARNING"],
                },
                "status": "WARNING",
                "source_module": "app.compliance",
                "provenance": "synthetic_fixture",
            },
            "retrieved_evidence": {
                "evidence_status": "RETRIEVED",
                "citations": [
                    {
                        "source": "ILLUSTRATIVE — not a real RBI source",
                        "locator": "§4 (sample)",
                        "quote": "[sample placeholder] Any unresolved warning across credit compliance evaluations requires mitigation tracking.",
                        "provenance": "illustrative",
                    }
                ],
            },
            "llm_interpretation": {
                "text": "Overall compliance status evaluates to WARNING due to disparate impact finding under RBI-FAIR-01. Action item remediation plan recommended.",
                "grounded_in": ["compliance.findings", "fairness.status"],
                "regulatory_basis": "illustrative_rule_only",
                "is_mock": True,
            },
        },
    ],
    "disclaimers": [
        "PROVISIONAL MOCK REPORT: Synthetic fixture for API and Dashboard Phase 3 integration testing.",
        "No real LLM generation or live vector database retrieval was executed (is_mock: True).",
        "Citations reference illustrative sample text, not verified RBI regulatory requirements.",
        "Drift detection in production compares dev train/test splits; synthetic scenarios are testing infrastructure.",
        (
            "Evidence coverage (1 of 5 sections) mirrors the real Phase 3 retrieval path, "
            "which indexes a single approved RBI excerpt. NOT_FOUND means no verified "
            "evidence was retrieved from the indexed corpus - not that no RBI rule exists."
        ),
    ],
    "evidence_coverage": {
        "retrieved": 1,
        "not_found": 4,
        "total": 5,
    },
    "is_mock": True,
}
