"""Tests for app.explainability against the REAL credit model (Phase 1).

These tests assert three things the module must never regress on:
  1. It explains the real pipeline from app.models.model.load(), not a stub.
  2. Its output speaks only the 20-feature RAW schema, never the 61-column
     transformed one.
  3. SHAP stays mathematically tied to the model's own decision function.
"""

import numpy as np
import pandas as pd
import pytest

from app.explainability.explain import (
    RAW_FEATURES,
    _build_raw_feature_groups,
    _split_pipeline,
    explain,
)
from app.models.model import load
from app.models.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Real-model wiring
# ---------------------------------------------------------------------------


def test_raw_features_come_from_the_model_module():
    """The explained schema must be sourced from Namitha's module, not copied."""
    assert RAW_FEATURES == list(FEATURE_COLUMNS)
    assert len(RAW_FEATURES) == 20


def test_no_dummy_three_feature_model_remains():
    """Regression guard: the Phase 1 dummy model must be gone from explain().

    The old implementation fitted its own LogisticRegression on
    income/age/credit_history_len. None of those may appear in the output,
    and the module must expose no dummy-fitting helper.
    """
    import app.explainability.explain as module

    assert not hasattr(module, "_fit_dummy_model")
    assert not hasattr(module, "DEFAULT_FEATURES")

    result = explain(method="shap")
    features = set(result["global_importance"])
    assert features == set(RAW_FEATURES)
    for stale in ("income", "credit_history_len"):
        assert stale not in features


def test_shap_uses_the_real_loaded_pipeline(model_output):
    """SHAP contributions must reconstruct the real model's decision function.

    This is the strongest available check that the explanation is tied to the
    actual model: SHAP is additive, so for every row
    ``sum(contributions) + expected_value == decision_function(x)``. A stub or
    a mis-grouped aggregation could not satisfy this.
    """
    import shap

    pipeline = load()
    preprocessor, classifier = _split_pipeline(pipeline)

    result = explain(model_output=model_output, method="shap")

    rows = model_output["feature_matrix"]
    transformed = preprocessor.transform(rows)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    background = preprocessor.transform(_training_like(rows))
    if hasattr(background, "toarray"):
        background = background.toarray()
    masker = shap.maskers.Independent(background, max_samples=background.shape[0])
    expected_value = shap.LinearExplainer(classifier, masker).expected_value

    truth = classifier.decision_function(transformed)
    for i, item in enumerate(result["per_instance"]):
        reconstructed = sum(item["contributions"].values()) + float(expected_value)
        assert reconstructed == pytest.approx(float(truth[i]), abs=1e-9)


def _training_like(_rows):
    """The same training split explain() uses as SHAP background."""
    from app.models.preprocessing import (
        DEFAULT_DATASET_PATH,
        load_dataset,
        preprocess,
        split_data,
    )

    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    X_train, _, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    return X_train.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transformed -> raw mapping
# ---------------------------------------------------------------------------


def test_feature_groups_tile_the_transformed_space():
    """The 61 transformed columns must map onto the 20 raw features exactly once."""
    preprocessor, _ = _split_pipeline(load())
    groups = _build_raw_feature_groups(preprocessor)

    assert set(groups) == set(RAW_FEATURES)

    indices = sorted(i for idx in groups.values() for i in idx)
    n_transformed = preprocessor.transform(_training_like(None)).shape[1]
    assert indices == list(range(n_transformed)), "groups must tile without gaps or overlap"
    assert n_transformed > len(RAW_FEATURES), "preprocessing should expand the space"


def test_categorical_features_group_multiple_columns():
    """A one-hot categorical must own more than one transformed column."""
    preprocessor, _ = _split_pipeline(load())
    groups = _build_raw_feature_groups(preprocessor)

    widths = {f: len(groups[f]) for f in CATEGORICAL_FEATURES}
    assert all(w >= 2 for w in widths.values()), widths

    numeric = [f for f in RAW_FEATURES if f not in CATEGORICAL_FEATURES]
    assert all(len(groups[f]) == 1 for f in numeric)


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_output_never_leaks_transformed_column_names(method, model_output):
    """No ``cat__``/``num__`` prefixed name may reach a caller."""
    result = explain(model_output=model_output, method=method)
    names = set(result["global_importance"])
    for item in result["per_instance"]:
        names |= set(item["contributions"])
    assert names == set(RAW_FEATURES)
    assert not any(n.startswith(("cat__", "num__")) for n in names)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_output_contract(method, model_output):
    """Both methods return exactly the approved four keys and shapes."""
    result = explain(model_output=model_output, method=method)

    assert set(result) == {"method", "per_instance", "global_importance", "is_mock"}
    assert result["method"] == method
    assert result["is_mock"] is False

    assert len(result["per_instance"]) == len(model_output["feature_matrix"])
    for i, item in enumerate(result["per_instance"]):
        assert set(item) == {"row_index", "contributions"}
        assert item["row_index"] == i
        assert set(item["contributions"]) == set(RAW_FEATURES)
        for value in item["contributions"].values():
            assert isinstance(value, float)
            assert not pd.isna(value)

    assert set(result["global_importance"]) == set(RAW_FEATURES)
    for value in result["global_importance"].values():
        assert isinstance(value, float)
        assert value >= 0.0


def test_shap_and_lime_contracts_are_structurally_identical(model_output):
    """SHAP and LIME must be interchangeable in shape, key-for-key."""
    shap_result = explain(model_output=model_output, method="shap")
    lime_result = explain(model_output=model_output, method="lime")

    assert set(shap_result) == set(lime_result)
    assert list(shap_result["global_importance"]) == list(lime_result["global_importance"])
    assert len(shap_result["per_instance"]) == len(lime_result["per_instance"])
    for a, b in zip(shap_result["per_instance"], lime_result["per_instance"]):
        assert a["row_index"] == b["row_index"]
        assert set(a["contributions"]) == set(b["contributions"])


def test_default_call_explains_real_held_out_rows():
    """explain() with no arguments works and returns raw-schema explanations."""
    result = explain()
    assert result["method"] == "shap"
    assert result["is_mock"] is False
    assert len(result["per_instance"]) > 1
    assert set(result["global_importance"]) == set(RAW_FEATURES)


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_rows_get_row_specific_explanations(method, model_output):
    """Different applicants must receive different explanations.

    A constant per-row output would mean the module reports a global summary
    while claiming per-instance attribution.
    """
    result = explain(model_output=model_output, method=method)
    rows = [tuple(item["contributions"][f] for f in RAW_FEATURES) for item in result["per_instance"]]
    assert len(set(rows)) == len(rows), "every row produced an identical explanation"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_foreign_feature_names_raise_clear_error():
    """The old 3-feature dummy schema must now be rejected outright."""
    foreign = pd.DataFrame(
        {"income": [30000, 60000], "age": [25, 40], "credit_history_len": [2, 8]}
    )
    with pytest.raises(ValueError, match="Incompatible feature schema") as exc:
        explain(model_output={"feature_matrix": foreign})

    message = str(exc.value)
    assert "Expected 20 raw features" in message
    assert "status_checking_account" in message
    assert "income" in message


def test_wrong_feature_count_raises_clear_error():
    """A truncated but otherwise valid schema must be refused."""
    truncated = pd.DataFrame({f: [1] for f in RAW_FEATURES[:10]})
    with pytest.raises(ValueError, match="Incompatible feature schema") as exc:
        explain(model_output={"feature_matrix": truncated})
    assert "Missing from input" in str(exc.value)


def test_extra_feature_raises_clear_error(raw_rows):
    """Extra columns are a schema mismatch, not something to silently drop."""
    extra = raw_rows.copy()
    extra["unexpected_column"] = 1
    with pytest.raises(ValueError, match="Incompatible feature schema") as exc:
        explain(model_output={"feature_matrix": extra})
    assert "unexpected_column" in str(exc.value)


def test_declared_feature_names_must_match_matrix_columns():
    """A matrix mislabelled with a valid schema is caught, not trusted.

    This must reach the feature_names-vs-columns branch specifically, NOT the
    raw-schema check. So ``feature_names`` is the complete, valid 20-feature
    RAW_FEATURES list (it passes schema validation on its own) while the
    matrix carries entirely different columns. Without this branch, a caller
    could label any matrix with the right feature_names and have it accepted.

    The ``match=`` is deliberately specific: a generic ValueError assertion
    would also pass if the raw-schema check fired first, which is exactly the
    false positive this test previously had.
    """
    mislabelled_columns = pd.DataFrame(
        {"income": [30000, 60000], "age": [25, 40], "credit_history_len": [2, 8]}
    )
    model_output = {
        "feature_matrix": mislabelled_columns,
        "model_metadata": {"feature_names": list(RAW_FEATURES)},
    }

    with pytest.raises(
        ValueError, match=r"feature_names'\] does not match feature_matrix columns"
    ) as exc:
        explain(model_output=model_output)

    message = str(exc.value)
    assert "Incompatible feature schema" not in message, (
        "raw-schema validation fired first - this test is not reaching the "
        "feature_names-vs-columns branch it is meant to cover"
    )
    assert "feature_names:" in message
    assert "matrix columns:" in message


def test_reordered_columns_are_realigned_not_misattributed(raw_rows, model_output):
    """Column order must not change any contribution.

    The pipeline selects columns by name, but positional work downstream
    (LIME indices, SHAP grouping) would silently misattribute if ordering
    were mishandled.
    """
    shuffled = raw_rows[list(reversed(RAW_FEATURES))]
    baseline = explain(model_output=model_output, method="shap")
    reordered = explain(
        model_output={
            "feature_matrix": shuffled,
            "model_metadata": {"feature_names": list(reversed(RAW_FEATURES))},
        },
        method="shap",
    )
    for a, b in zip(baseline["per_instance"], reordered["per_instance"]):
        for feature in RAW_FEATURES:
            assert a["contributions"][feature] == pytest.approx(
                b["contributions"][feature], abs=1e-9
            )


def test_accepts_row_records_as_served_over_http(raw_rows, model_output):
    """The API serialises feature_matrix as row records; both forms must work."""
    records = raw_rows.to_dict(orient="records")
    from_records = explain(model_output={"feature_matrix": records}, method="shap")
    from_frame = explain(model_output=model_output, method="shap")

    for a, b in zip(from_records["per_instance"], from_frame["per_instance"]):
        for feature in RAW_FEATURES:
            assert a["contributions"][feature] == pytest.approx(
                b["contributions"][feature], abs=1e-9
            )


def test_lime_refuses_unseen_category_but_shap_explains_it(raw_rows):
    """Pin the deliberate SHAP/LIME divergence on out-of-vocabulary categories.

    The model's OneHotEncoder uses handle_unknown='ignore', so the pipeline
    tolerates an unseen category and SHAP can explain the row. LIME cannot:
    its raw-feature encoding has no integer code for that value, and the only
    ways to proceed (substituting a known category, or reporting a
    contribution anyway) would describe a row the surrogate never
    represented. Refusing is the documented, intentional behaviour.

    This test fails if either half of that contract silently changes.
    """
    rows = raw_rows.copy()
    rows.loc[rows.index[0], "purpose"] = "A999_NEVER_SEEN_IN_TRAINING"
    model_output = {"feature_matrix": rows}

    # SHAP: the real pipeline absorbs the unknown category, so this works.
    shap_result = explain(model_output=model_output, method="shap")
    assert len(shap_result["per_instance"]) == len(rows)
    assert shap_result["is_mock"] is False

    # LIME: refuses, and says why plus what to do instead.
    with pytest.raises(
        ValueError, match="LIME cannot explain an unseen categorical value"
    ) as exc:
        explain(model_output=model_output, method="lime")

    message = str(exc.value)
    assert "A999_NEVER_SEEN_IN_TRAINING" in message
    assert "purpose" in message
    assert "handle_unknown='ignore'" in message, "must explain why SHAP differs"


def test_unseen_category_is_never_silently_substituted(raw_rows):
    """An unseen category must not be quietly mapped onto a known one.

    If the encoder ever fell back to a default code, LIME would return a
    confident explanation of the wrong applicant. That must raise, not
    produce numbers.
    """
    rows = raw_rows.copy()
    rows.loc[rows.index[0], "savings_account"] = "NOT_A_REAL_CATEGORY"

    with pytest.raises(ValueError, match="LIME cannot explain an unseen"):
        explain(model_output={"feature_matrix": rows}, method="lime")


def test_invalid_method_raises_error():
    with pytest.raises(ValueError, match="Unsupported explainability method"):
        explain(method="not_a_method")


def test_non_dict_model_output_raises_error(raw_rows):
    with pytest.raises(ValueError, match="model_output must be a dict"):
        explain(model_output=raw_rows)


def test_empty_feature_matrix_raises_error():
    with pytest.raises(ValueError, match="empty"):
        explain(model_output={"feature_matrix": []})


def test_model_output_predictions_are_not_trusted(model_output):
    """predictions/probabilities in model_output must not affect the result.

    They are a compatibility input; the loaded pipeline is the sole source of
    truth. Contradictory values must be ignored, not blended in.
    """
    clean = explain(model_output=model_output, method="shap")

    tampered = dict(model_output)
    tampered["predictions"] = [1] * len(model_output["feature_matrix"])
    tampered["probabilities"] = [0.99] * len(model_output["feature_matrix"])
    tampered["model_metadata"] = dict(model_output["model_metadata"], model_type="xgboost")

    result = explain(model_output=tampered, method="shap")
    for a, b in zip(clean["per_instance"], result["per_instance"]):
        for feature in RAW_FEATURES:
            assert a["contributions"][feature] == pytest.approx(
                b["contributions"][feature], abs=1e-12
            )


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------


def test_shap_and_lime_are_on_different_scales(model_output):
    """Guard the documented scale difference so nobody averages the two.

    SHAP is log-odds; LIME is predicted probability. Their magnitudes are not
    interchangeable, and the module docstring says so.

    Note on what is deliberately NOT asserted here: ranking agreement between
    the two methods. An earlier version of this suite asserted the two
    produced identical global-importance rankings. That held for the old
    dummy model but is false for the real one (measured rank correlation
    ~0.5), so asserting it would be both wrong and brittle. The methods are
    two independent views, and divergence is expected rather than a defect.
    """
    shap_result = explain(model_output=model_output, method="shap")
    lime_result = explain(model_output=model_output, method="lime")

    shap_scale = sum(shap_result["global_importance"].values())
    lime_scale = sum(lime_result["global_importance"].values())
    assert shap_scale > lime_scale


def test_lime_is_deterministic(model_output):
    """Compliance evidence that changes between runs cannot be cited."""
    first = explain(model_output=model_output, method="lime")
    second = explain(model_output=model_output, method="lime")
    for a, b in zip(first["per_instance"], second["per_instance"]):
        assert a["contributions"] == pytest.approx(b["contributions"])


def test_global_importance_is_mean_absolute_contribution(model_output):
    """global_importance must be derivable from per_instance, not independent."""
    result = explain(model_output=model_output, method="shap")
    for feature in RAW_FEATURES:
        expected = np.mean(
            [abs(item["contributions"][feature]) for item in result["per_instance"]]
        )
        assert result["global_importance"][feature] == pytest.approx(expected, abs=1e-12)
