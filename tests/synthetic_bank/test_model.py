"""Unit tests for the synthetic bank XGBoost model (owner: Namitha)."""
import os

import pytest

from app.synthetic_bank.data_generator import NUMERIC_FEATURES, generate_missing_data_customers
from app.synthetic_bank.model import MODEL_ID, TRAINED_ON, evaluate, load, predict_one, save, train


def test_train_is_deterministic():
    model_a, metrics_a = train(random_state=42)
    model_b, metrics_b = train(random_state=42)
    assert metrics_a == metrics_b

    sample = {
        "employment_type": "salaried", "region": "north", "loan_purpose": "auto",
        "age": 35, "annual_income": 60000, "employment_years": 8,
        "existing_loans": 1, "credit_utilization_ratio": 0.3,
        "late_payments_12m": 0, "loan_amount": 10000,
    }
    assert predict_one(model_a, sample) == predict_one(model_b, sample)


def test_train_returns_expected_metric_keys():
    _model, metrics = train(random_state=42)
    assert set(metrics.keys()) == {
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "n_test_samples", "is_mock",
    }
    assert metrics["is_mock"] is False
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    model, _metrics = train(random_state=42)
    path = str(tmp_path / "model.joblib")
    save(model, path=path)
    assert os.path.exists(path)

    reloaded = load(path=path)
    sample = {
        "employment_type": "unemployed", "region": "east", "loan_purpose": "personal",
        "age": 28, "annual_income": 22000, "employment_years": 1,
        "existing_loans": 2, "credit_utilization_ratio": 0.8,
        "late_payments_12m": 3, "loan_amount": 12000,
    }
    assert predict_one(model, sample) == predict_one(reloaded, sample)


def test_load_missing_artifact_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load(path="app/synthetic_bank/artifacts/does-not-exist.joblib")


def test_predict_one_higher_risk_applicant_scores_higher_probability():
    model, _metrics = train(random_state=42)

    risky = {
        "employment_type": "unemployed", "region": "north", "loan_purpose": "personal",
        "age": 30, "annual_income": 18000, "employment_years": 0.2,
        "existing_loans": 4, "credit_utilization_ratio": 0.95,
        "late_payments_12m": 5, "loan_amount": 15000,
    }
    safe = {
        "employment_type": "salaried", "region": "north", "loan_purpose": "auto",
        "age": 45, "annual_income": 95000, "employment_years": 18,
        "existing_loans": 0, "credit_utilization_ratio": 0.05,
        "late_payments_12m": 0, "loan_amount": 8000,
    }

    risky_result = predict_one(model, risky)
    safe_result = predict_one(model, safe)
    assert risky_result["probability"] > safe_result["probability"]


# =====================================================================
# TRAINED_ON provenance
# =====================================================================


def test_trained_on_exists_and_is_a_string():
    assert isinstance(TRAINED_ON, str)
    assert TRAINED_ON


def test_trained_on_truthfully_names_the_synthetic_generator_not_german_credit():
    assert "generate_customers" in TRAINED_ON
    assert "synthetic_bank" in TRAINED_ON
    assert "german_credit" not in TRAINED_ON.lower()


def test_model_id_is_unaffected_by_trained_on_addition():
    assert MODEL_ID == "synthetic-bank-credit-v1"


# =====================================================================
# Missing-data scenario input -- app/synthetic_bank/model.py's pipeline is
# NOT modified by this task. Only a non-pinning observation is kept here:
# whether the current pipeline accepts or rejects NaN in a CATEGORICAL
# column is deliberately NOT asserted, so that a future decision to make
# the pipeline tolerant of missing categoricals is an improvement, not a
# test failure. Generator-level missing-data behavior (rate, determinism,
# validation) is covered in tests/synthetic_bank/test_data_generator.py.
# =====================================================================


def test_missing_numeric_only_input_currently_scores_successfully():
    """Documents today's behavior: NaN in NUMERIC_FEATURES only currently
    scores without error, because XGBoost natively tolerates missing
    numeric values in the passthrough columns. This is an observation, not
    a guarantee -- it may change if the pipeline changes."""
    model, _metrics = train(random_state=42)
    missing_df = generate_missing_data_customers(
        n=10, random_state=11, missing_rate=0.5, columns=NUMERIC_FEATURES
    )
    row = missing_df.iloc[0].to_dict()
    result = predict_one(model, row)
    assert result["prediction"] in (0, 1)
