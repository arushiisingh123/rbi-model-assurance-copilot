"""Shared fixtures for the explainability test suite.

The explainability module explains the REAL trained model and deliberately
never trains one itself. The model artifact is gitignored, so on a fresh
clone (and in CI) it does not exist yet. These fixtures provision it once
per session using Namitha's own public ``train()`` -- the tests own that
setup step, the explainability module does not.
"""
import os

import pandas as pd
import pytest

from app.models.model import DEFAULT_MODEL_ARTIFACT_PATH, train
from app.models.preprocessing import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    load_dataset,
    preprocess,
    split_data,
)


@pytest.fixture(scope="session", autouse=True)
def real_model_artifact():
    """Ensure a trained model artifact exists before any explainability test.

    Autouse: every test in this package explains the real model, so none of
    them can run without it.
    """
    if not os.path.exists(DEFAULT_MODEL_ARTIFACT_PATH):
        train(
            dataset_path=DEFAULT_DATASET_PATH,
            save_path=DEFAULT_MODEL_ARTIFACT_PATH,
            random_state=42,
        )
    return DEFAULT_MODEL_ARTIFACT_PATH


@pytest.fixture(scope="session")
def raw_rows() -> pd.DataFrame:
    """A few real held-out rows in the model's raw 20-feature schema.

    Kept small on purpose: LIME costs roughly a quarter-second per row
    against the real pipeline, so tests that exercise it stay brisk.
    """
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    return X_test.reset_index(drop=True).head(3)


@pytest.fixture(scope="session")
def model_output(raw_rows: pd.DataFrame) -> dict:
    """A predict_batch()-shaped dict wrapping those rows."""
    return {
        "feature_matrix": raw_rows,
        "model_metadata": {
            "model_type": "logistic_regression",
            "version": "0.1.0",
            "trained_on": DEFAULT_DATASET_PATH,
            "feature_names": list(FEATURE_COLUMNS),
        },
        "is_mock": False,
    }
