"""Fixtures for the drift integration tests (owner: Arushi).

Drift is evaluated over real German Credit data. The train/test split fixture
needs no trained model at all; only the tests that additionally check the model
output path request the artifact fixtures, which provision it once per session
using Namitha's public ``train()`` (the artifact is gitignored, so a fresh
clone and CI start without one). This mirrors the pattern already established
in ``tests/explainability/conftest.py``.

Deliberately NOT autouse: the drift unit tests in ``test_drift.py`` and
``test_scenario.py`` run on inline fixtures and must not pay the cost of
loading real data or training a model they never use.
"""
import os

import pandas as pd
import pytest

from app.models.model import DEFAULT_MODEL_ARTIFACT_PATH, predict_batch, train
from app.models.preprocessing import (
    DEFAULT_DATASET_PATH,
    load_dataset,
    preprocess,
    split_data,
)


@pytest.fixture(scope="session")
def german_credit_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The real German Credit train/test feature splits, as DataFrames.

    ``random_state=42`` and ``test_size=0.2`` match the project convention and
    the values ``predict_batch()`` uses internally, so the test split returned
    here is the same one the model scores.

    These are development splits, NOT production monitoring data -- see
    ``docs/decisions.md``, "Phase 2 drift integration reference/current data".
    """
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    return X_train, X_test


@pytest.fixture(scope="session")
def real_model_artifact() -> str:
    """Ensure a trained model artifact exists, training it once if missing."""
    if not os.path.exists(DEFAULT_MODEL_ARTIFACT_PATH):
        train(
            dataset_path=DEFAULT_DATASET_PATH,
            save_path=DEFAULT_MODEL_ARTIFACT_PATH,
            random_state=42,
        )
    return DEFAULT_MODEL_ARTIFACT_PATH


@pytest.fixture(scope="session")
def real_model_output(real_model_artifact: str) -> dict:
    """Real ``predict_batch()`` output over the held-out German Credit test split."""
    return predict_batch()
