"""Fixtures for the fairness integration tests (owner: Arushi).

The Phase 2 integration tests evaluate fairness over the output of the REAL
trained credit model. That artifact is gitignored, so on a fresh clone (and in
CI) it does not exist yet. These fixtures provision it once per session using
Namitha's own public ``train()`` -- the tests own that setup step, the fairness
module does not. This mirrors the pattern already established in
``tests/explainability/conftest.py``.

Deliberately NOT autouse: the fairness unit tests in ``test_fairness.py`` and
``test_grouping.py`` run on inline fixtures and must not pay the cost of
training a model they never use. Only the integration tests request these.
"""
import os

import pytest

from app.models.model import DEFAULT_MODEL_ARTIFACT_PATH, predict_batch, train
from app.models.preprocessing import DEFAULT_DATASET_PATH


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
    """Real ``predict_batch()`` output over the held-out German Credit test split.

    Deterministic: the split uses the project's ``random_state=42`` convention.
    """
    return predict_batch()
