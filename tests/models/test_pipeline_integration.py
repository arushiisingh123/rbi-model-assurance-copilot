"""Phase 2 integration tests for the model pipeline (owner: Namitha).

Phase 1 tests exercise the model's functions in isolation. These tests prove
the whole model-side path runs unattended and produces the stable output
contract downstream modules (explainability, fairness, drift, API) consume:

    German Credit CSV
     -> load_dataset()
     -> preprocess()
     -> split_data()
     -> build_pipeline() + fit
     -> save() / load()
     -> predict_batch()

Scope is strictly the model module. Downstream modules are NOT imported here:
coupling model tests to another owner's module would make this suite fail when
that module changes. Cross-module consumption was verified separately and is
recorded in the Phase 2 model-integration report.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.models import model as model_module
from app.models.model import (
    DEFAULT_DATASET_PATH,
    MODEL_TYPE,
    MODEL_VERSION,
    build_pipeline,
    evaluate,
    load,
    load_dataset,
    predict_batch,
    preprocess,
    save,
    split_data,
    train,
)
from app.models.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

CONTRACT_KEYS = {
    "predictions",
    "probabilities",
    "feature_matrix",
    "model_metadata",
    "is_mock",
}


# =====================================================================
# 1. Full pipeline: real dataset -> preprocessing -> model -> predictions
# =====================================================================


def test_full_pipeline_runs_end_to_end_without_manual_intervention(tmp_path):
    """The whole chain executes with no manual steps and yields the contract."""
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, cat_cols, num_cols = preprocess(df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert cat_cols == CATEGORICAL_FEATURES
    assert num_cols == NUMERIC_FEATURES
    assert set(y.unique()) == {0, 1}

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(
        categorical_cols=cat_cols, numeric_cols=num_cols, random_state=42
    )
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    assert metrics["is_mock"] is False
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    artifact = str(tmp_path / "credit_model.joblib")
    save(pipeline, path=artifact)
    reloaded = load(artifact)
    assert isinstance(reloaded, Pipeline)

    result = predict_batch(feature_matrix=X_test)

    assert set(result.keys()) == CONTRACT_KEYS
    assert result["is_mock"] is False
    assert len(result["predictions"]) == len(X_test)
    assert len(result["probabilities"]) == len(X_test)
    assert len(result["feature_matrix"]) == len(X_test)


def test_predict_batch_default_path_is_self_contained():
    """predict_batch() with no arguments runs the real dataset path itself."""
    result = predict_batch()

    assert set(result.keys()) == CONTRACT_KEYS
    assert result["is_mock"] is False
    assert len(result["feature_matrix"]) == 200  # held-out test split


# =====================================================================
# 2. Output schema the downstream modules depend on
# =====================================================================


def test_output_schema_prediction_labels_and_probabilities():
    """predictions are ints in {0, 1}; probabilities are floats in [0, 1]."""
    result = predict_batch()

    assert all(isinstance(p, int) and p in {0, 1} for p in result["predictions"])
    assert all(
        isinstance(p, float) and 0.0 <= p <= 1.0 for p in result["probabilities"]
    )

    # probabilities[i] is P(class == 1) = P(BAD): label 1 iff probability >= 0.5
    for pred, prob in zip(result["predictions"], result["probabilities"]):
        assert (pred == 1) == (prob >= 0.5)


def test_output_schema_feature_matrix_structure():
    """feature_matrix is a DataFrame of the 20 raw features in canonical order."""
    fm = predict_batch()["feature_matrix"]

    assert isinstance(fm, pd.DataFrame)
    assert list(fm.columns) == FEATURE_COLUMNS
    assert len(fm.columns) == 20
    # One-hot expanded columns are internal to the pipeline; they never leak.
    for col in fm.columns:
        assert "_A1" not in col and "_A2" not in col and "=" not in col
    assert set(fm["personal_status_and_sex"].unique()).issubset(
        {"A91", "A92", "A93", "A94", "A95"}
    )


def test_output_schema_metadata_structure():
    """model_metadata carries a consistent, self-describing contract."""
    result = predict_batch()
    meta = result["model_metadata"]

    assert meta["model_type"] == MODEL_TYPE == "logistic_regression"
    assert meta["version"] == MODEL_VERSION
    assert meta["trained_on"] == DEFAULT_DATASET_PATH
    assert meta["feature_names"] == FEATURE_COLUMNS
    assert meta["feature_names"] == list(result["feature_matrix"].columns)

    ls = meta["label_semantics"]
    assert ls["positive_class"] == 1
    assert ls["favorable_outcome_label"] == 0
    assert "GOOD" in ls["0"] and "BAD" in ls["1"]


def test_output_lengths_are_mutually_consistent():
    """predictions / probabilities / feature_matrix all describe the same rows."""
    result = predict_batch()
    n = len(result["feature_matrix"])
    assert len(result["predictions"]) == n
    assert len(result["probabilities"]) == n


# =====================================================================
# 3. Stability / determinism of the contract
# =====================================================================


def test_predict_batch_default_output_is_stable_across_calls():
    """Two default calls return identical predictions, probabilities, features."""
    a = predict_batch()
    b = predict_batch()

    assert a["predictions"] == b["predictions"]
    assert a["probabilities"] == b["probabilities"]
    pd.testing.assert_frame_equal(a["feature_matrix"], b["feature_matrix"])
    assert a["model_metadata"] == b["model_metadata"]


# =====================================================================
# 4. API-boundary serialization (feature_matrix DataFrame -> list[dict])
# =====================================================================


def test_feature_matrix_round_trips_through_list_of_records():
    """docs/module-interfaces.md: pd.DataFrame(payload['feature_matrix']) round-trips."""
    fm = predict_batch()["feature_matrix"]
    records = fm.to_dict(orient="records")

    assert isinstance(records, list) and isinstance(records[0], dict)
    assert set(records[0].keys()) == set(FEATURE_COLUMNS)

    restored = pd.DataFrame(records)
    pd.testing.assert_frame_equal(restored, fm.reset_index(drop=True))


def test_predictions_and_probabilities_are_json_native():
    """predictions / probabilities serialize directly (no numpy scalars)."""
    result = predict_batch()
    dumped = json.dumps(
        {
            "predictions": result["predictions"],
            "probabilities": result["probabilities"],
            "model_metadata": result["model_metadata"],
            "is_mock": result["is_mock"],
        }
    )
    assert isinstance(dumped, str)


# =====================================================================
# 5. Stale artifact handling (Phase 1 sign-off follow-up)
# =====================================================================


def _subset_schema_pipeline(tmp_path):
    """A pipeline fitted on a reduced feature set -> mismatches FEATURE_COLUMNS."""
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    cat = CATEGORICAL_FEATURES[:3]
    num = NUMERIC_FEATURES[:2]
    stale = build_pipeline(categorical_cols=cat, numeric_cols=num, random_state=42)
    stale.fit(X[cat + num], y)
    path = str(tmp_path / "stale_model.joblib")
    save(stale, path=path)
    return path


def test_load_stale_schema_artifact_raises_clear_error(tmp_path):
    """load() names the differing columns instead of failing opaquely later."""
    path = _subset_schema_pipeline(tmp_path)
    with pytest.raises(ValueError, match="does not match the current feature schema"):
        load(path)


def test_predict_batch_rebuilds_when_default_artifact_is_stale(tmp_path, monkeypatch):
    """predict_batch() self-heals from a stale default artifact, never serves it."""
    stale_path = _subset_schema_pipeline(tmp_path)
    default_path = str(tmp_path / "artifacts" / "credit_model.joblib")
    os.makedirs(os.path.dirname(default_path), exist_ok=True)
    # Put the stale model where the default artifact would live.
    import shutil

    shutil.copyfile(stale_path, default_path)
    monkeypatch.setattr(model_module, "DEFAULT_MODEL_ARTIFACT_PATH", default_path)

    result = predict_batch()

    assert result["is_mock"] is False
    assert result["model_metadata"]["feature_names"] == FEATURE_COLUMNS
    # The rebuilt artifact now on disk matches the current schema.
    assert load(default_path).feature_names_in_.tolist() == FEATURE_COLUMNS
