"""Fixtures for the monitoring tests (owner: Arushi).

The identity tests monitor the REAL trained models over the project's canonical
train/test split. Those artifacts are gitignored, so on a fresh clone (and in
CI) they do not exist yet; these fixtures provision them once per session using
Namitha's own public ``train()``. The tests own that setup step, the monitoring
modules do not. Same pattern, and the same fixture names, as
``tests/drift/conftest.py`` and ``tests/fairness/conftest.py``.

Deliberately NOT autouse: ``test_monitoring.py`` runs entirely on inline
fixtures and must not pay the cost of loading real data or training a model it
never uses. Only ``test_monitoring_identity.py`` requests these.
"""
import os

import pandas as pd
import pytest

from app.models.model import DEFAULT_MODEL_ARTIFACT_PATH, train
from app.models.preprocessing import (
    DEFAULT_DATASET_PATH,
    load_dataset,
    preprocess,
    split_data,
)


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
def german_credit_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The real German Credit train/test feature splits, used as two windows.

    ``random_state=42`` and ``test_size=0.2`` match the project convention and
    the values ``predict_batch()`` uses internally.

    These are development splits standing in for monitoring windows so the
    monitoring flow can be demonstrated end to end. They are NOT observed
    production monitoring data -- see ``docs/decisions.md``, "Phase 2 drift
    integration reference/current data".
    """
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    return X_train, X_test
