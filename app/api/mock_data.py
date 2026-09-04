"""Synthetic Mock Data Fixtures for Phase 1 API (Owner: Khushi).

===============================================================================
WARNING: SYNTHETIC / MOCK DATA ONLY — NOT REAL OUTPUT — NOT FOR PRODUCTION USE
All data fixtures defined in this module are synthetic placeholders designed
to validate API endpoint shapes, schemas, and serialization contracts during
Phase 1 independent module development.
===============================================================================
"""

MOCK_MODEL_RESULT = {
    "predictions": [0, 1, 0],
    "probabilities": [0.12, 0.81, 0.33],
    "feature_matrix": [
        {"income": 45000, "age": 34, "credit_history_len": 5},
        {"income": 120000, "age": 45, "credit_history_len": 12},
        {"income": 28000, "age": 22, "credit_history_len": 2},
    ],
    "model_metadata": {
        "model_type": "xgboost",
        "version": "0.1.0",
        "trained_on": "data/sample/credit_sample.csv",
        "feature_names": ["income", "age", "credit_history_len"],
    },
    "is_mock": True,
}

MOCK_EXPLAINABILITY_RESULT_SHAP = {
    "method": "shap",
    "per_instance": [
        {
            "row_index": 0,
            "contributions": {"income": 0.31, "age": 0.12, "credit_history_len": -0.05},
        },
        {
            "row_index": 1,
            "contributions": {"income": -0.22, "age": 0.05, "credit_history_len": 0.18},
        },
        {
            "row_index": 2,
            "contributions": {"income": 0.40, "age": -0.15, "credit_history_len": -0.10},
        },
    ],
    "global_importance": {
        "income": 0.42,
        "age": 0.28,
        "credit_history_len": 0.19,
    },
    "is_mock": True,
}

MOCK_EXPLAINABILITY_RESULT_LIME = {
    "method": "lime",
    "per_instance": [
        {
            "row_index": 0,
            "contributions": {"income": 0.29, "age": 0.10, "credit_history_len": -0.04},
        },
        {
            "row_index": 1,
            "contributions": {"income": -0.20, "age": 0.07, "credit_history_len": 0.15},
        },
        {
            "row_index": 2,
            "contributions": {"income": 0.38, "age": -0.12, "credit_history_len": -0.08},
        },
    ],
    "global_importance": {
        "income": 0.39,
        "age": 0.25,
        "credit_history_len": 0.18,
    },
    "is_mock": True,
}

# Default explainability fixture is SHAP
MOCK_EXPLAINABILITY_RESULT = MOCK_EXPLAINABILITY_RESULT_SHAP

MOCK_FAIRNESS_RESULT = {
    "protected_attribute": "gender",
    "demographic_parity_diff": 0.14,
    "disparate_impact_ratio": 0.78,
    "status": "WARNING",
    "is_mock": True,
}

MOCK_DRIFT_RESULT = {
    "features_evaluated": ["income", "age", "credit_history_len"],
    "psi": 0.09,
    "ks_statistic": 0.11,
    "status": "PASS",
    "is_mock": True,
}

MOCK_COMPLIANCE_RESULT = {
    "findings": [
        {
            "rule_id": "RBI-FAIR-01",
            "rule_description": "Assess model decisions for disparate impact and unfair bias across protected groups.",
            "technical_finding_ref": "fairness.disparate_impact_ratio",
            "status": "FAIL",
            "evidence_chunks": [],
        },
        {
            "rule_id": "RBI-DRIFT-01",
            "rule_description": "Monitor population stability index (PSI) to detect distribution shifts.",
            "technical_finding_ref": "drift.psi",
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
    },
    "compliance": MOCK_COMPLIANCE_RESULT,
    "note": "SYNTHETIC / MOCK DATA. Not real results. Phase 1 mock fixtures only.",
}
