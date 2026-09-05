"""Phase 1 unit and integration tests for app.models (owner: Namitha).

Verifies real dataset ingestion, preprocessing, training, evaluation, persistence,
and the frozen batch prediction interface on the UCI German Credit dataset.
"""

import os
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.models import (
    evaluate,
    load,
    load_dataset,
    predict_batch,
    preprocess,
    save,
    split_data,
    train,
)
from app.models.model import (
    DEFAULT_DATASET_PATH,
    MODEL_TYPE,
    MODEL_VERSION,
    build_pipeline,
)
from app.models.preprocessing import (
    CATEGORICAL_FEATURES,
    EXPECTED_ROW_COUNT,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def raw_dataset():
    """Load the raw dataset once for the test module."""
    return load_dataset(DEFAULT_DATASET_PATH)


@pytest.fixture(scope="module")
def preprocessed_data(raw_dataset):
    """Preprocess the dataset once for the test module."""
    X, y, cat_cols, num_cols = preprocess(raw_dataset)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42
    )
    return {
        "X": X,
        "y": y,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture(scope="module")
def trained_model(preprocessed_data):
    """Train the model pipeline once for testing predictions and evaluation."""
    model = build_pipeline(
        categorical_cols=preprocessed_data["cat_cols"],
        numeric_cols=preprocessed_data["num_cols"],
        random_state=42,
    )
    model.fit(preprocessed_data["X_train"], preprocessed_data["y_train"])
    return model


# =====================================================================
# 1. Dataset Tests
# =====================================================================


def test_dataset_file_exists():
    """Verify the approved German Credit dataset file exists on disk."""
    assert os.path.exists(DEFAULT_DATASET_PATH), (
        f"Approved dataset file missing at {DEFAULT_DATASET_PATH}"
    )


def test_dataset_row_and_column_counts(raw_dataset):
    """Verify dataset contains 1,000 instances and 21 columns (20 features + target)."""
    assert len(raw_dataset) == EXPECTED_ROW_COUNT
    assert raw_dataset.shape == (1000, 21)


def test_dataset_required_columns_exist(raw_dataset):
    """Verify all 20 feature columns and the target column exist in the CSV."""
    for col in FEATURE_COLUMNS:
        assert col in raw_dataset.columns, f"Missing feature column: {col}"
    assert TARGET_COLUMN in raw_dataset.columns, f"Missing target column: {TARGET_COLUMN}"


def test_dataset_no_unexpected_missing_values(raw_dataset):
    """Verify there are zero null/missing values in the dataset."""
    assert raw_dataset.isnull().sum().sum() == 0


def test_dataset_original_target_values(raw_dataset):
    """Verify the raw target contains only original UCI values {1, 2}."""
    unique_targets = set(raw_dataset[TARGET_COLUMN].unique())
    assert unique_targets == {1, 2}, f"Unexpected target values: {unique_targets}"


# =====================================================================
# 2. Preprocessing Tests
# =====================================================================


def test_preprocess_shapes_and_target_polarity(preprocessed_data):
    """Verify preprocessing separates X and y with matching lengths and polarity {0, 1}."""
    X = preprocessed_data["X"]
    y = preprocessed_data["y"]

    assert len(X) == EXPECTED_ROW_COUNT
    assert len(y) == EXPECTED_ROW_COUNT
    assert set(y.unique()) == {0, 1}
    # Original: 700 Good (1 -> 0) and 300 Bad (2 -> 1)
    assert (y == 0).sum() == 700
    assert (y == 1).sum() == 300


def test_preprocess_preserves_personal_status_sex(preprocessed_data):
    """Verify Attribute 9 is preserved without converting to fairness groups.

    The column name must match CLAUDE.md and app/fairness/
    (`personal_status_and_sex`), and it must stay a raw combined
    marital-status + sex field, not a derived gender grouping.
    """
    X = preprocessed_data["X"]
    assert "personal_status_and_sex" in X.columns
    assert "personal_status_sex" not in X.columns
    assert "gender" not in X.columns and "sex" not in X.columns

    # Check that raw categories (A91..A95) are preserved unchanged.
    unique_vals = set(X["personal_status_and_sex"].unique())
    assert unique_vals.issubset({"A91", "A92", "A93", "A94", "A95"})


def test_raw_feature_schema_names_and_types(preprocessed_data):
    """Verify the authoritative RAW feature schema: 20 features, 13 categorical + 7 numeric."""
    assert len(FEATURE_COLUMNS) == 20
    assert len(CATEGORICAL_FEATURES) == 13
    assert len(NUMERIC_FEATURES) == 7
    # Categorical + numeric partition the full schema exactly
    assert set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) == set(FEATURE_COLUMNS)
    assert set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES) == set()
    assert list(preprocessed_data["X"].columns) == FEATURE_COLUMNS


def test_preprocess_raises_on_missing_feature_column():
    """Verify preprocess() raises ValueError (not KeyError) when a feature column is absent."""
    bad_df = pd.DataFrame({"age": [30, 40], TARGET_COLUMN: [1, 2]})
    with pytest.raises(ValueError, match="missing required feature columns"):
        preprocess(bad_df)


def test_preprocess_no_nan_in_features(preprocessed_data):
    """Verify feature matrix X contains no NaNs."""
    assert preprocessed_data["X"].isnull().sum().sum() == 0


def test_split_data_deterministic_and_stratified(preprocessed_data):
    """Verify train_test_split is stratified, deterministic, and 80/20 proportion."""
    X_train = preprocessed_data["X_train"]
    X_test = preprocessed_data["X_test"]
    y_train = preprocessed_data["y_train"]
    y_test = preprocessed_data["y_test"]

    assert len(X_train) == 800
    assert len(X_test) == 200
    assert len(y_train) == 800
    assert len(y_test) == 200

    # Stratified proportion check (70% good, 30% bad)
    train_bad_ratio = (y_train == 1).mean()
    test_bad_ratio = (y_test == 1).mean()
    assert abs(train_bad_ratio - 0.30) < 0.01
    assert abs(test_bad_ratio - 0.30) < 0.01

    # Determinism check with fixed seed
    X_train2, _, _, _ = split_data(
        preprocessed_data["X"], preprocessed_data["y"], test_size=0.2, random_state=42
    )
    pd.testing.assert_frame_equal(X_train, X_train2)


# =====================================================================
# 3. Model Training & Pipeline Tests
# =====================================================================


def test_pipeline_structure(trained_model):
    """Verify the trained model is an sklearn Pipeline with preprocessor and classifier."""
    assert isinstance(trained_model, Pipeline)
    assert "preprocessor" in trained_model.named_steps
    assert "classifier" in trained_model.named_steps


def test_train_function_execution(tmp_path):
    """Verify train() runs end-to-end and saves artifact to tmp_path."""
    save_file = str(tmp_path / "test_model.joblib")
    result = train(
        dataset_path=DEFAULT_DATASET_PATH,
        save_path=save_file,
        random_state=42,
    )

    assert result["status"] == "trained"
    assert isinstance(result["model"], Pipeline)
    assert os.path.exists(save_file)
    assert result["n_train_samples"] == 800
    assert result["n_test_samples"] == 200


def test_model_training_is_reproducible(preprocessed_data):
    """Verify training with fixed random_state produces identical predictions."""
    m1 = build_pipeline(random_state=42)
    m1.fit(preprocessed_data["X_train"], preprocessed_data["y_train"])

    m2 = build_pipeline(random_state=42)
    m2.fit(preprocessed_data["X_train"], preprocessed_data["y_train"])

    preds1 = m1.predict_proba(preprocessed_data["X_test"])
    preds2 = m2.predict_proba(preprocessed_data["X_test"])
    assert (preds1 == preds2).all()


# =====================================================================
# 4. Evaluation Tests
# =====================================================================


def test_evaluate_keys_and_ranges(trained_model, preprocessed_data):
    """Verify evaluation returns all required keys, metrics in [0, 1], and is_mock=False."""
    metrics = evaluate(
        trained_model, preprocessed_data["X_test"], preprocessed_data["y_test"]
    )

    required_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "n_test_samples",
        "is_mock",
    }
    assert set(metrics.keys()) == required_keys
    assert metrics["is_mock"] is False
    assert metrics["n_test_samples"] == 200

    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        val = metrics[metric_name]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"Metric {metric_name} out of [0, 1]: {val}"


def test_evaluate_performance_above_baseline(trained_model, preprocessed_data):
    """Verify model achieves meaningful performance (accuracy > baseline, ROC-AUC > 0.65)."""
    metrics = evaluate(
        trained_model, preprocessed_data["X_test"], preprocessed_data["y_test"]
    )

    # Majority class baseline is 70% (0.70)
    assert metrics["accuracy"] >= 0.70, (
        f"Accuracy ({metrics['accuracy']:.4f}) should be at or above majority baseline"
    )
    # ROC-AUC should be substantially above random 0.50
    assert metrics["roc_auc"] > 0.65, (
        f"ROC-AUC ({metrics['roc_auc']:.4f}) should be meaningfully above random"
    )


# =====================================================================
# 5. Model Save & Load Tests
# =====================================================================


def test_save_and_load_roundtrip(trained_model, preprocessed_data, tmp_path):
    """Verify saved and loaded model produces identical predictions."""
    model_path = str(tmp_path / "artifacts" / "roundtrip_model.joblib")
    save_result = save(trained_model, path=model_path)
    assert save_result["status"] == "saved"
    assert os.path.exists(model_path)

    loaded_model = load(path=model_path)
    assert isinstance(loaded_model, Pipeline)

    original_preds = trained_model.predict(preprocessed_data["X_test"])
    loaded_preds = loaded_model.predict(preprocessed_data["X_test"])
    assert (original_preds == loaded_preds).all()

    original_probs = trained_model.predict_proba(preprocessed_data["X_test"])
    loaded_probs = loaded_model.predict_proba(preprocessed_data["X_test"])
    assert (original_probs == loaded_probs).all()


def test_load_missing_file_raises_error(tmp_path):
    """Verify loading from non-existent path raises FileNotFoundError with explanation."""
    non_existent = str(tmp_path / "does_not_exist.joblib")
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load(non_existent)


# =====================================================================
# 6. Predict Batch Interface Tests (Frozen Contract)
# =====================================================================


def test_predict_batch_default_arguments():
    """Verify predict_batch() with no arguments uses test split and returns frozen shape."""
    result = predict_batch()

    assert "predictions" in result
    assert "probabilities" in result
    assert "feature_matrix" in result
    assert "model_metadata" in result
    assert "is_mock" in result

    assert result["is_mock"] is False
    assert isinstance(result["feature_matrix"], pd.DataFrame)
    assert len(result["feature_matrix"]) == 200  # Held-out test set size

    n_samples = len(result["feature_matrix"])
    assert len(result["predictions"]) == n_samples
    assert len(result["probabilities"]) == n_samples

    # Check predictions are integers in {0, 1}
    for p in result["predictions"]:
        assert isinstance(p, int)
        assert p in {0, 1}

    # Check probabilities are floats in [0, 1]
    for prob in result["probabilities"]:
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    # Check metadata
    metadata = result["model_metadata"]
    assert metadata["model_type"] == MODEL_TYPE
    assert metadata["version"] == MODEL_VERSION
    assert metadata["trained_on"] == DEFAULT_DATASET_PATH
    assert metadata["feature_names"] == list(result["feature_matrix"].columns)


def test_predict_batch_output_exposes_raw_feature_schema():
    """feature_names must be the 20 RAW features, never the one-hot expanded columns."""
    result = predict_batch()
    names = result["model_metadata"]["feature_names"]

    assert names == FEATURE_COLUMNS
    assert len(names) == 20
    assert "personal_status_and_sex" in names
    # One-hot expansion happens inside the pipeline and must not leak out.
    for n in names:
        assert "_A1" not in n and "_A2" not in n and "=" not in n
    # The raw combined attribute keeps its raw category values in the matrix.
    assert set(result["feature_matrix"]["personal_status_and_sex"].unique()).issubset(
        {"A91", "A92", "A93", "A94", "A95"}
    )


def test_predict_batch_label_semantics_metadata():
    """model_metadata must state 0=GOOD, 1=BAD, positive class 1, favorable label 0."""
    meta = predict_batch()["model_metadata"]
    ls = meta["label_semantics"]

    assert ls["positive_class"] == 1
    assert ls["favorable_outcome_label"] == 0
    assert "GOOD" in ls["0"] and "BAD" in ls["1"]
    assert "P(class == 1)" in ls["probabilities_represent"]


def test_predict_batch_probability_is_p_of_bad():
    """probabilities[i] must be P(class 1) = P(BAD): prediction 1 <=> probability >= 0.5."""
    result = predict_batch()
    for pred, prob in zip(result["predictions"], result["probabilities"]):
        if pred == 1:
            assert prob >= 0.5
        else:
            assert prob <= 0.5


def test_predict_batch_matches_direct_positive_class_column(trained_model, preprocessed_data):
    """The returned probabilities equal predict_proba's column for class label 1."""
    X = preprocessed_data["X_test"].head(15).copy()
    result = predict_batch(feature_matrix=X)

    classes = list(trained_model.classes_)
    pos_col = trained_model.predict_proba(X.reset_index(drop=True))[:, classes.index(1)]
    assert result["probabilities"] == [float(p) for p in pos_col]


def test_predict_batch_preserves_feature_order_when_columns_shuffled(preprocessed_data):
    """Columns supplied in a different order are reordered to FEATURE_COLUMNS."""
    shuffled = preprocessed_data["X_test"].head(5)[list(reversed(FEATURE_COLUMNS))].copy()
    result = predict_batch(feature_matrix=shuffled)
    assert list(result["feature_matrix"].columns) == FEATURE_COLUMNS
    assert result["model_metadata"]["feature_names"] == FEATURE_COLUMNS


def test_predict_batch_feature_matrix_is_dataframe_and_round_trips():
    """feature_matrix stays a DataFrame in-process; list[dict] round-trips it for the API."""
    result = predict_batch()
    fm = result["feature_matrix"]
    assert isinstance(fm, pd.DataFrame)

    records = fm.to_dict(orient="records")
    assert isinstance(records, list) and isinstance(records[0], dict)
    restored = pd.DataFrame(records)
    pd.testing.assert_frame_equal(restored, fm.reset_index(drop=True))


def test_predict_batch_with_custom_feature_matrix(preprocessed_data):
    """Verify predict_batch() handles a custom DataFrame properly."""
    custom_X = preprocessed_data["X_test"].head(10).copy()
    result = predict_batch(feature_matrix=custom_X)

    assert result["is_mock"] is False
    assert len(result["predictions"]) == 10
    assert len(result["probabilities"]) == 10
    assert len(result["feature_matrix"]) == 10
    assert list(result["model_metadata"]["feature_names"]) == FEATURE_COLUMNS


def test_predict_batch_invalid_input_types():
    """Verify predict_batch() raises TypeError when input is not a DataFrame."""
    with pytest.raises(TypeError, match="Expected feature_matrix to be a pandas DataFrame"):
        predict_batch(feature_matrix={"a": [1, 2, 3]})


def test_predict_batch_missing_required_columns():
    """Verify predict_batch() raises ValueError when required feature columns are missing."""
    incomplete_df = pd.DataFrame({"duration_months": [12, 24], "age": [30, 45]})
    with pytest.raises(ValueError, match="missing required columns"):
        predict_batch(feature_matrix=incomplete_df)


# =====================================================================
# 7. Public API Import Tests
# =====================================================================


def test_public_api_exports():
    """Verify public exports from app.models work as expected."""
    import app.models as models

    expected_exports = [
        "train",
        "predict_batch",
        "evaluate",
        "save",
        "load",
        "load_dataset",
        "preprocess",
        "split_data",
        # RAW feature schema + label semantics, exposed for downstream modules
        "FEATURE_COLUMNS",
        "CATEGORICAL_FEATURES",
        "NUMERIC_FEATURES",
        "LABEL_GOOD",
        "LABEL_BAD",
        "POSITIVE_CLASS",
        "FAVORABLE_OUTCOME_LABEL",
        "LABEL_SEMANTICS",
        "MODEL_TYPE",
        "MODEL_VERSION",
    ]
    for export in expected_exports:
        assert hasattr(models, export), f"app.models is missing export: {export}"

    assert models.LABEL_GOOD == 0
    assert models.LABEL_BAD == 1
    assert models.POSITIVE_CLASS == 1
    assert models.FAVORABLE_OUTCOME_LABEL == 0
    assert models.FEATURE_COLUMNS[8] == "personal_status_and_sex"


# =====================================================================
# 8. Reproducibility + Serialization Fixture Consistency
# =====================================================================


def test_train_end_to_end_is_reproducible(tmp_path):
    """Two independent train() runs with the same seed produce identical predictions."""
    p1 = str(tmp_path / "m1.joblib")
    p2 = str(tmp_path / "m2.joblib")
    train(dataset_path=DEFAULT_DATASET_PATH, save_path=p1, random_state=42)
    train(dataset_path=DEFAULT_DATASET_PATH, save_path=p2, random_state=42)

    m1, m2 = load(p1), load(p2)
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, _, _, _ = preprocess(df)

    assert (m1.predict(X) == m2.predict(X)).all()
    assert (m1.predict_proba(X) == m2.predict_proba(X)).all()


def test_serialized_output_fixture_matches_current_contract():
    """The tests/models fixture mirrors the serialized predict_batch() contract."""
    import json

    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "sample_model_output.json"
    )
    with open(fixture_path, encoding="utf-8") as fh:
        fixture = json.load(fh)

    assert set(fixture.keys()) == {
        "predictions",
        "probabilities",
        "feature_matrix",
        "model_metadata",
        "is_mock",
    }
    # feature_matrix is the serialized list[dict] form (API boundary shape)
    assert isinstance(fixture["feature_matrix"], list)
    assert set(fixture["feature_matrix"][0].keys()) == set(FEATURE_COLUMNS)

    meta = fixture["model_metadata"]
    assert meta["model_type"] == MODEL_TYPE
    assert meta["feature_names"] == FEATURE_COLUMNS
    assert meta["label_semantics"]["positive_class"] == 1
    assert meta["label_semantics"]["favorable_outcome_label"] == 0
