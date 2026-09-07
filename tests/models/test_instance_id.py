"""Phase 3 tests for stable per-record identity (`instance_id`) — owner: Namitha.

Team decision (2026-09-07): every input record gets a stable `instance_id` at
the data/model boundary. Explainability and later reporting/LLM components must
key record identity on this value, never on a DataFrame row position or a
reset index.

These tests prove:
  * `load_dataset()` attaches deterministic, unique, non-null IDs.
  * IDs survive preprocessing, the stratified train/test split, and shuffling.
  * `instance_id` is never a model feature and never enters the pipeline.
  * `predict_batch()` returns IDs aligned 1:1 with predictions / probabilities
    / feature_matrix rows, deterministically, with no random generation.
  * Custom-input IDs are preserved when supplied and never fed to the model.
  * Prediction / probability semantics are unchanged by adding identity.
"""

import numpy as np
import pandas as pd
import pytest

from app.models import (
    INSTANCE_ID_COLUMN,
    load_dataset,
    make_dataset_instance_ids,
    predict_batch,
    preprocess,
    split_data,
)
from app.models.model import DEFAULT_DATASET_PATH
from app.models.preprocessing import FEATURE_COLUMNS


# =====================================================================
# 1. Dataset-level identity
# =====================================================================


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DEFAULT_DATASET_PATH)


def test_dataset_rows_receive_instance_ids(dataset):
    assert INSTANCE_ID_COLUMN in dataset.columns
    assert len(dataset[INSTANCE_ID_COLUMN]) == len(dataset)


def test_instance_ids_are_non_null(dataset):
    assert dataset[INSTANCE_ID_COLUMN].notna().all()


def test_instance_ids_are_unique(dataset):
    ids = dataset[INSTANCE_ID_COLUMN]
    assert ids.is_unique
    assert len(set(ids)) == len(dataset)


def test_instance_ids_are_strings(dataset):
    assert all(isinstance(i, str) for i in dataset[INSTANCE_ID_COLUMN])


def test_repeated_dataset_loads_produce_identical_ids():
    a = load_dataset(DEFAULT_DATASET_PATH)[INSTANCE_ID_COLUMN].tolist()
    b = load_dataset(DEFAULT_DATASET_PATH)[INSTANCE_ID_COLUMN].tolist()
    assert a == b
    # And identical to the pure deterministic scheme.
    assert a == make_dataset_instance_ids(len(a))


def test_dataset_scheme_is_deterministic_positional():
    assert make_dataset_instance_ids(3) == ["gc-0000", "gc-0001", "gc-0002"]


def test_load_dataset_rejects_supplied_null_or_duplicate_ids(tmp_path):
    df = load_dataset(DEFAULT_DATASET_PATH)
    dup = df.copy()
    dup[INSTANCE_ID_COLUMN] = "gc-0000"  # all identical -> duplicates
    path = tmp_path / "dup_ids.csv"
    dup.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique and non-null"):
        load_dataset(str(path))


# =====================================================================
# 2. Identity survives preprocessing / splitting / shuffling
# =====================================================================


def test_instance_id_is_not_a_model_feature(dataset):
    """preprocess() keeps X to the 20 features; instance_id is dropped."""
    X, y, cat_cols, num_cols = preprocess(dataset)
    assert list(X.columns) == FEATURE_COLUMNS
    assert INSTANCE_ID_COLUMN not in X.columns
    assert INSTANCE_ID_COLUMN not in cat_cols
    assert INSTANCE_ID_COLUMN not in num_cols


def test_predictive_feature_columns_remain_the_approved_twenty():
    assert len(FEATURE_COLUMNS) == 20
    assert INSTANCE_ID_COLUMN not in FEATURE_COLUMNS
    assert FEATURE_COLUMNS[8] == "personal_status_and_sex"


def test_split_preserves_record_identity(dataset):
    """Each split row's identity still maps to its original raw feature row."""
    X, y, _, _ = preprocess(dataset)
    id_series = dataset[INSTANCE_ID_COLUMN]
    X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)

    # The split shuffles: test rows are not the last 200 by position.
    assert list(X_test.index) != list(range(800, 1000))

    for idx in list(X_test.index)[:25]:
        rec_id = id_series.loc[idx]
        # The identity resolves back to the exact same raw feature values.
        pd.testing.assert_series_equal(
            X_test.loc[idx], dataset.loc[idx, FEATURE_COLUMNS], check_names=False
        )
        assert rec_id == f"gc-{idx:04d}"

    # Train and test identities partition the dataset with no overlap.
    train_ids = set(id_series.loc[X_train.index])
    test_ids = set(id_series.loc[X_test.index])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == len(dataset)


def test_row_reset_does_not_redefine_identity(dataset):
    """After reset_index the positional index changes but instance_id does not."""
    X, y, _, _ = preprocess(dataset)
    id_series = dataset[INSTANCE_ID_COLUMN]
    _, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)

    ids_before = id_series.loc[X_test.index].tolist()
    original_positions = list(X_test.index)
    reset = X_test.reset_index(drop=True)

    # Positional index is now 0..N-1 ...
    assert reset.index.tolist() == list(range(len(reset)))
    # ... but that tells us nothing about identity: row 0's real identity is
    # still the original record's id, which is NOT "gc-0000" here because the
    # split shuffled.
    assert original_positions[0] != 0
    assert ids_before[0] == f"gc-{original_positions[0]:04d}"
    # The identity list itself is unchanged by the reset.
    assert ids_before == id_series.loc[X_test.index].tolist()


# =====================================================================
# 3. predict_batch() — default path
# =====================================================================


@pytest.fixture(scope="module")
def default_output():
    return predict_batch()


def test_predict_batch_output_contains_instance_ids(default_output):
    assert "instance_ids" in default_output
    assert isinstance(default_output["instance_ids"], list)
    assert all(isinstance(i, str) for i in default_output["instance_ids"])


def test_predict_batch_lengths_are_all_aligned(default_output):
    n = len(default_output["feature_matrix"])
    assert (
        len(default_output["instance_ids"])
        == len(default_output["predictions"])
        == len(default_output["probabilities"])
        == n
    )


def test_predict_batch_default_ids_are_the_dataset_ids_for_the_test_split(default_output):
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    expected = df[INSTANCE_ID_COLUMN].loc[X_test.index].tolist()
    assert default_output["instance_ids"] == expected
    # These are genuine dataset identities, not positional placeholders.
    assert all(i.startswith("gc-") for i in default_output["instance_ids"])
    assert default_output["instance_ids"] != [f"row-{i:04d}" for i in range(200)]


def test_predict_batch_default_ids_are_unique(default_output):
    ids = default_output["instance_ids"]
    assert len(set(ids)) == len(ids)


def test_repeated_predict_batch_calls_return_identical_ids(default_output):
    again = predict_batch()
    assert again["instance_ids"] == default_output["instance_ids"]


def test_instance_id_aligns_with_the_scored_feature_row(default_output):
    """instance_ids[i] identifies the same raw record as feature_matrix row i."""
    df = load_dataset(DEFAULT_DATASET_PATH).set_index(INSTANCE_ID_COLUMN)
    fm = default_output["feature_matrix"]
    for i in (0, 1, 50, 199):
        rec_id = default_output["instance_ids"][i]
        original = df.loc[rec_id, FEATURE_COLUMNS]
        pd.testing.assert_series_equal(
            fm.iloc[i], original, check_names=False, check_dtype=False
        )


# =====================================================================
# 4. predict_batch() — custom input path
# =====================================================================


@pytest.fixture(scope="module")
def custom_rows():
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    ids = df[INSTANCE_ID_COLUMN].loc[X_test.index]
    frame = X_test.copy()
    frame[INSTANCE_ID_COLUMN] = ids.values
    return frame.reset_index(drop=True).head(10)


def test_custom_input_ids_are_preserved_when_supplied(custom_rows):
    result = predict_batch(feature_matrix=custom_rows)
    assert result["instance_ids"] == custom_rows[INSTANCE_ID_COLUMN].tolist()


def test_custom_instance_id_column_is_not_fed_to_the_model(custom_rows):
    result = predict_batch(feature_matrix=custom_rows)
    # Returned feature_matrix is the 20 model features only.
    assert list(result["feature_matrix"].columns) == FEATURE_COLUMNS
    assert INSTANCE_ID_COLUMN not in result["feature_matrix"].columns
    assert result["model_metadata"]["feature_names"] == FEATURE_COLUMNS

    # Predictions are identical whether or not the id column is present.
    without = predict_batch(feature_matrix=custom_rows.drop(columns=[INSTANCE_ID_COLUMN]))
    assert result["predictions"] == without["predictions"]
    assert result["probabilities"] == without["probabilities"]


def test_custom_input_without_ids_gets_deterministic_positional_placeholders(custom_rows):
    plain = custom_rows.drop(columns=[INSTANCE_ID_COLUMN])
    a = predict_batch(feature_matrix=plain)
    b = predict_batch(feature_matrix=plain)
    assert a["instance_ids"] == [f"row-{i:04d}" for i in range(len(plain))]
    assert a["instance_ids"] == b["instance_ids"]  # deterministic, not random


def test_custom_input_null_id_raises(custom_rows):
    bad = custom_rows.copy()
    bad.loc[0, INSTANCE_ID_COLUMN] = np.nan
    with pytest.raises(ValueError, match="null"):
        predict_batch(feature_matrix=bad)


def test_ids_are_not_generated_randomly_during_prediction(custom_rows):
    """No uuid/timestamp/randomness: same rows in same order -> same ids twice."""
    plain = custom_rows.drop(columns=[INSTANCE_ID_COLUMN])
    runs = [predict_batch(feature_matrix=plain)["instance_ids"] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_reordering_custom_rows_does_not_silently_reidentify(custom_rows):
    """Reversing supplied rows carries each row's own id with it, not by slot."""
    reversed_rows = custom_rows.iloc[::-1].reset_index(drop=True)
    forward = predict_batch(feature_matrix=custom_rows)
    backward = predict_batch(feature_matrix=reversed_rows)
    assert backward["instance_ids"] == list(reversed(forward["instance_ids"]))
    # And each id still lines up with its own prediction.
    fwd = dict(zip(forward["instance_ids"], forward["predictions"]))
    bwd = dict(zip(backward["instance_ids"], backward["predictions"]))
    assert fwd == bwd


# =====================================================================
# 5. Prediction semantics unchanged by identity metadata
# =====================================================================


def test_prediction_and_probability_semantics_are_unchanged(default_output):
    for pred, prob in zip(default_output["predictions"], default_output["probabilities"]):
        assert isinstance(pred, int) and pred in {0, 1}
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0
        assert (pred == 1) == (prob >= 0.5)


def test_default_output_matches_the_frozen_fixture_predictions(default_output):
    """Adding instance_id must not move any prediction for the same rows."""
    assert default_output["predictions"][:5] == [0, 0, 1, 0, 0]
    assert default_output["instance_ids"][:5] == [
        "gc-0030",
        "gc-0128",
        "gc-0289",
        "gc-0216",
        "gc-0966",
    ]
