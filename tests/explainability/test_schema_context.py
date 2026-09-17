"""Tests for the parameterized schema / reference-data layer (owner: Manas, Step 3).

WHAT STEP 3 CHANGED
-------------------
The data-preparation helpers in ``app/explainability/explain.py`` used to read
the module-level German Credit schema and dataset directly. That made them
unusable for any other model: a foreign schema was rejected before any model
was touched, and the reference distribution was always German Credit's
training split no matter which model was being explained.

Those helpers now take a ``_SchemaContext``. This file tests both halves of
that change:

- the GENERALIZED half -- a foreign schema and a foreign reference
  distribution are handled correctly when a context supplies them;
- the UNCHANGED half -- with no adapter, every behaviour, error message, and
  number is exactly what Phase 1 produced.

The second half matters as much as the first. Step 3 is a refactor of a path
that four other modules already consume (the API, the report, the dashboard
panel, and evidence.py), so "it still works for German Credit" is a
requirement, not a courtesy.

WHAT STEP 3 DID NOT CHANGE
--------------------------
``explain()`` still takes no adapter and still explains the default German
Credit pipeline. The adapter-aware path exists at the helper level only.
Selecting the model and the explainer from an adapter is Step 4. Nothing here
calls the capability layer, and nothing here executes an adapter-aware SHAP
or LIME run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.explainability.explain import (
    DEFAULT_EXPLAIN_ROWS,
    RAW_FEATURES,
    _build_raw_feature_groups,
    _context_for,
    _default_feature_frame,
    _DEFAULT_CONTEXT,
    _load_fitted_model,
    _resolve_feature_frame,
    _split_pipeline,
    _training_frame,
    _validate_feature_groups,
    _validate_raw_schema,
    explain,
)
from app.models.model import MODEL_ID, ModelAdapter
from app.models.preprocessing import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    load_dataset,
    preprocess,
    split_data,
)
from app.models.registry import get_default_registry

SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"

# The synthetic bank's own 10-feature schema -- deliberately nothing like
# German Credit's 20, so a test using it cannot accidentally pass because the
# two schemas overlap.
BANK_FEATURES: List[str] = [
    "employment_type",
    "region",
    "loan_purpose",
    "age",
    "annual_income",
    "employment_years",
    "existing_loans",
    "credit_utilization_ratio",
    "late_payments_12m",
    "loan_amount",
]
BANK_CATEGORICAL: List[str] = ["employment_type", "region", "loan_purpose"]
BANK_NUMERIC: List[str] = [f for f in BANK_FEATURES if f not in BANK_CATEGORICAL]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAdapter(ModelAdapter):
    """A minimal adapter whose schema and reference data are set explicitly."""

    def __init__(
        self,
        *,
        model_id: str = "fake-bank-model",
        feature_names: Optional[List[str]] = None,
        background: Optional[Any] = None,
        artifact: Any = None,
    ) -> None:
        self.model_id = model_id
        self.model_version = "0.1.0"
        self.model_type = "fake_type"
        self.feature_names = list(BANK_FEATURES if feature_names is None else feature_names)
        self._background = background
        self._artifact = artifact

    @property
    def supports_probability(self) -> bool:
        return True

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def load_fitted_model(self) -> Any:
        return self._artifact

    def background_data(self) -> Optional[Any]:
        return self._background


def _explain_module():
    """The explain MODULE, not the explain function.

    ``app/explainability/__init__.py`` does ``from ...explain import explain``,
    which rebinds the package attribute ``explain`` from the submodule to the
    function. Anything reaching for the module by attribute silently gets the
    function instead, so it is resolved through sys.modules here.
    """
    from importlib import import_module

    return import_module("app.explainability.explain")


def _bank_frame(rows: int = 6) -> pd.DataFrame:
    """A reference batch in the synthetic bank's raw feature space."""
    return pd.DataFrame(
        {
            "employment_type": ["salaried", "self_employed"] * (rows // 2),
            "region": ["north", "south"] * (rows // 2),
            "loan_purpose": ["home", "auto"] * (rows // 2),
            "age": np.linspace(25, 60, rows),
            "annual_income": np.linspace(300000, 1800000, rows),
            "employment_years": np.linspace(1, 30, rows),
            "existing_loans": np.arange(rows) % 4,
            "credit_utilization_ratio": np.linspace(0.05, 0.95, rows),
            "late_payments_12m": np.arange(rows) % 3,
            "loan_amount": np.linspace(50000, 900000, rows),
        }
    )


@pytest.fixture
def bank_context():
    """A context for a foreign model, built from its adapter."""
    return _context_for(FakeAdapter(background=_bank_frame()))


@pytest.fixture(scope="module")
def registry():
    return get_default_registry()


# ===========================================================================
# 1 & 2. A foreign schema is accepted under its own context, rejected under
#        German Credit's. Same input, two contexts, opposite outcomes --
#        which is the whole point of parameterizing the schema.
# ===========================================================================


def test_foreign_schema_is_accepted_under_its_own_context(bank_context):
    """The 10-feature bank schema validates against the bank's own names."""
    _validate_raw_schema(list(BANK_FEATURES), context=bank_context)


def test_foreign_schema_is_accepted_in_any_column_order(bank_context):
    """Validation is on the feature SET; ordering is realigned, not rejected."""
    _validate_raw_schema(list(reversed(BANK_FEATURES)), context=bank_context)


def test_foreign_schema_is_rejected_under_the_german_credit_context():
    """The pre-Step-3 behaviour, still correct when no adapter is supplied."""
    with pytest.raises(ValueError, match="Incompatible feature schema") as exc:
        _validate_raw_schema(list(BANK_FEATURES), context=_DEFAULT_CONTEXT)

    message = str(exc.value)
    assert "Expected 20 raw features" in message
    assert "credit_utilization_ratio" in message


def test_default_context_error_still_names_german_credit():
    """Unchanged wording for the unchanged path, including the pointer file."""
    with pytest.raises(ValueError) as exc:
        _validate_raw_schema(["nonsense"], context=_DEFAULT_CONTEXT)

    message = str(exc.value)
    assert "Incompatible feature schema for the real credit model." in message
    assert "German Credit raw schema" in message
    assert "app/models/preprocessing.py FEATURE_COLUMNS" in message


def test_adapter_context_error_does_not_claim_german_credit(bank_context):
    """A foreign-schema failure must not send the operator to the wrong contract.

    Naming German Credit's FEATURE_COLUMNS while validating a bank model's
    schema would describe a contract this input was never meant to satisfy.
    """
    with pytest.raises(ValueError) as exc:
        _validate_raw_schema(["nonsense"], context=bank_context)

    message = str(exc.value)
    assert "German Credit" not in message
    assert "FEATURE_COLUMNS" not in message
    assert "fake-bank-model" in message
    assert "adapter.feature_names" in message


# ===========================================================================
# 3, 4, 5. Missing / extra / duplicate features all fail clearly
# ===========================================================================


def test_missing_foreign_feature_raises(bank_context):
    received = [f for f in BANK_FEATURES if f != "annual_income"]

    with pytest.raises(ValueError, match="Incompatible feature schema") as exc:
        _validate_raw_schema(received, context=bank_context)

    message = str(exc.value)
    assert "Missing from input: ['annual_income']" in message
    assert "Expected 10 raw features" in message


def test_extra_foreign_feature_raises(bank_context):
    """Extra columns are a schema mismatch, not something to silently drop."""
    with pytest.raises(ValueError, match="Incompatible feature schema") as exc:
        _validate_raw_schema(BANK_FEATURES + ["scraped_postcode"], context=bank_context)

    assert "Unexpected in input: ['scraped_postcode']" in str(exc.value)


def test_duplicate_foreign_column_raises(bank_context):
    """A duplicate is reported as a duplicate, not as a count mismatch.

    Without the explicit check this surfaced as 'Expected 10, Received 11'
    with the duplicate buried in a list -- technically a refusal, but it
    named the wrong problem.
    """
    with pytest.raises(ValueError, match="Duplicate feature column") as exc:
        _validate_raw_schema(BANK_FEATURES + ["age"], context=bank_context)

    message = str(exc.value)
    assert "['age']" in message
    assert "exactly once" in message


def test_duplicate_column_raises_on_the_default_context_too():
    with pytest.raises(ValueError, match="Duplicate feature column"):
        _validate_raw_schema(list(RAW_FEATURES) + [RAW_FEATURES[0]], context=_DEFAULT_CONTEXT)


def test_adapter_declaring_duplicate_feature_names_is_refused():
    """The context itself refuses an incoherent schema at construction."""
    with pytest.raises(ValueError, match="duplicate feature"):
        _context_for(FakeAdapter(feature_names=["age", "age", "region"]))


@pytest.mark.parametrize("declared", [[], None], ids=["empty", "none"])
def test_adapter_without_usable_feature_names_is_refused(declared):
    adapter = FakeAdapter(background=_bank_frame())
    adapter.feature_names = declared

    with pytest.raises(ValueError, match="feature_names"):
        _context_for(adapter)


# ===========================================================================
# 6, 7, 8. Adapter reference data is validated, never substituted
# ===========================================================================


def test_adapter_background_is_used_when_it_matches_the_schema(bank_context):
    """The adapter's own data becomes the reference distribution."""
    background = _training_frame(bank_context)

    assert list(background.columns) == BANK_FEATURES
    assert len(background) == 6
    pd.testing.assert_frame_equal(background, _bank_frame()[BANK_FEATURES])


def test_adapter_background_is_realigned_to_the_declared_order():
    """Column order in the reference data must not shift attributions."""
    shuffled = _bank_frame()[list(reversed(BANK_FEATURES))]
    context = _context_for(FakeAdapter(background=shuffled))

    assert list(_training_frame(context).columns) == BANK_FEATURES


def test_adapter_background_tolerates_extra_columns_but_drops_them():
    """Reference batches often carry an id or label column alongside features."""
    extra = _bank_frame()
    extra["instance_id"] = [f"row-{i}" for i in range(len(extra))]
    context = _context_for(FakeAdapter(background=extra))

    assert list(_training_frame(context).columns) == BANK_FEATURES


@pytest.mark.parametrize(
    "background, expected",
    [
        (None, "returned None"),
        (pd.DataFrame(), "is empty"),
        (pd.DataFrame(columns=BANK_FEATURES), "is empty"),
    ],
    ids=["none", "empty_frame", "no_rows"],
)
def test_empty_or_absent_adapter_background_is_rejected(background, expected):
    """An empty frame is not a reference distribution and must not pass as one."""
    context = _context_for(FakeAdapter(background=background))

    with pytest.raises(ValueError, match=expected):
        _training_frame(context)


def test_adapter_background_with_wrong_schema_is_rejected():
    """German Credit data offered as a bank model's reference must be refused."""
    german = load_dataset(DEFAULT_DATASET_PATH)
    X, _y, _, _ = preprocess(german)
    context = _context_for(FakeAdapter(background=X.head(5)))

    with pytest.raises(ValueError, match="missing raw feature") as exc:
        _training_frame(context)

    message = str(exc.value)
    assert "annual_income" in message
    assert "fake-bank-model" in message


def test_non_dataframe_adapter_background_is_rejected():
    context = _context_for(FakeAdapter(background=[[1, 2], [3, 4]]))

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        _training_frame(context)


def test_adapter_background_with_duplicate_columns_is_rejected():
    duplicated = pd.concat([_bank_frame(), _bank_frame()[["age"]]], axis=1)
    context = _context_for(FakeAdapter(background=duplicated))

    with pytest.raises(ValueError, match="duplicate column"):
        _training_frame(context)


def test_adapter_background_never_falls_back_to_german_credit():
    """The failure mode this step exists to prevent.

    Silently substituting German Credit's training split as a bank model's
    reference distribution would yield confident numbers describing a
    distribution the model was never fitted on. The refusal above is the
    point; this asserts no fallback happens on the way to it.
    """
    context = _context_for(FakeAdapter(background=None))

    with pytest.raises(ValueError):
        _training_frame(context)
    with pytest.raises(ValueError):
        _default_feature_frame(context)


def test_adapter_default_rows_come_from_the_adapters_own_data():
    """Default rows under an adapter context are that model's, not German Credit's."""
    context = _context_for(FakeAdapter(background=_bank_frame(rows=40)))
    frame = _default_feature_frame(context)

    assert list(frame.columns) == BANK_FEATURES
    assert len(frame) == DEFAULT_EXPLAIN_ROWS


def test_real_synthetic_bank_adapter_reference_data_validates(registry):
    """End to end on the real registry entry -- no network, no German Credit.

    Constructing a RESTAdapter makes no HTTP call and ``background_data()`` is
    a local frame, so this exercises the real adapter's real schema.
    """
    adapter = registry.get(SYNTHETIC_BANK_MODEL_ID)
    context = _context_for(adapter)

    assert context.names == BANK_FEATURES

    background = _training_frame(context)
    assert list(background.columns) == BANK_FEATURES
    assert len(background) == 50

    # The two schemas overlap only on 'age', so the bank's own columns are
    # proof the German Credit split was not substituted here.
    assert context.names != list(RAW_FEATURES)
    assert "annual_income" not in RAW_FEATURES
    assert "credit_utilization_ratio" not in RAW_FEATURES
    assert set(background.columns) & set(RAW_FEATURES) == {"age"}


def test_adapter_artifact_comes_from_the_adapter_not_the_default_model():
    sentinel = object()
    context = _context_for(FakeAdapter(background=_bank_frame(), artifact=sentinel))

    assert _load_fitted_model(context) is sentinel


def test_rest_adapter_without_an_artifact_still_raises_not_implemented(registry):
    """No German Credit fallback when a model has no local artifact.

    What to DO about such a model is the capability layer's decision (Step 2)
    and Step 4's execution. This only pins that the artifact loader does not
    quietly hand back a different model's pipeline.
    """
    context = _context_for(registry.get(SYNTHETIC_BANK_MODEL_ID))

    with pytest.raises(NotImplementedError):
        _load_fitted_model(context)


# ===========================================================================
# 9. The no-adapter path still uses the original dataset and background
# ===========================================================================


def _german_credit_splits():
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    return split_data(X, y, test_size=0.2, random_state=42)


def test_no_adapter_training_frame_is_the_german_credit_training_split():
    """Recomputed independently in the test, then compared frame-to-frame."""
    X_train, _, _, _ = _german_credit_splits()
    expected = X_train.reset_index(drop=True)

    pd.testing.assert_frame_equal(_training_frame(), expected)
    pd.testing.assert_frame_equal(_training_frame(_DEFAULT_CONTEXT), expected)


def test_no_adapter_default_frame_is_the_german_credit_test_split():
    _, X_test, _, _ = _german_credit_splits()
    expected = X_test.reset_index(drop=True).head(DEFAULT_EXPLAIN_ROWS)

    pd.testing.assert_frame_equal(_default_feature_frame(), expected)


def test_no_adapter_artifact_is_the_default_pipeline():
    from app.models.model import load as load_real_model

    assert type(_load_fitted_model()) is type(load_real_model())
    assert _split_pipeline(_load_fitted_model()) is not None


def test_default_context_schema_is_the_german_credit_schema():
    assert _DEFAULT_CONTEXT.names == list(FEATURE_COLUMNS)
    assert _DEFAULT_CONTEXT.adapter is None
    assert _context_for(None) is _DEFAULT_CONTEXT


# ===========================================================================
# 10, 11, 12. Feature grouping against a foreign preprocessor
# ===========================================================================


def _bank_preprocessor(remainder: str = "passthrough") -> ColumnTransformer:
    """A preprocessor shaped differently from German Credit's on purpose.

    German Credit's names BOTH column blocks explicitly. This one names only
    the categorical block and lets ``remainder`` carry the seven numerics,
    which is the shape a bank's own pipeline is likely to have.
    """
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BANK_CATEGORICAL),
        ],
        remainder=remainder,
    )
    ct.fit(_bank_frame())
    return ct


def test_feature_groups_handle_categorical_plus_passthrough_remainder():
    """The structural algorithm is unchanged and already handles this shape.

    sklearn materializes ``remainder='passthrough'`` as a fitted transformer
    in ``transformers_`` rather than the bare string, so it lands in the
    column-preserving branch: one input column, one output column. No
    special-casing was needed or added.
    """
    preprocessor = _bank_preprocessor()
    groups = _build_raw_feature_groups(preprocessor)

    assert set(groups) == set(BANK_FEATURES)
    for feature in BANK_CATEGORICAL:
        assert len(groups[feature]) == 2, "each bank categorical has two categories"
    for feature in BANK_NUMERIC:
        assert len(groups[feature]) == 1, "passthrough is column-preserving"


def test_feature_groups_validate_against_the_bank_schema_not_german_credits():
    """Requirement 10: the bank's preprocessing validates on its own 10 names.

    Under German Credit's 20 names the identical mapping is rejected -- which
    is exactly the coupling Step 3 removed.
    """
    preprocessor = _bank_preprocessor()
    transformed = preprocessor.transform(_bank_frame())
    groups = _build_raw_feature_groups(preprocessor)

    _validate_feature_groups(
        groups, transformed.shape[1], raw_feature_names=BANK_FEATURES
    )

    with pytest.raises(ValueError, match="not in the known raw schema"):
        _validate_feature_groups(
            groups, transformed.shape[1], raw_feature_names=RAW_FEATURES
        )


def test_feature_group_tiling_covers_exactly_the_transformed_range():
    """The invariant that keeps attributions attached to the right feature."""
    preprocessor = _bank_preprocessor()
    transformed = preprocessor.transform(_bank_frame())
    groups = _build_raw_feature_groups(preprocessor)

    covered = sorted(i for indices in groups.values() for i in indices)
    assert covered == list(range(transformed.shape[1]))

    flat = [i for indices in groups.values() for i in indices]
    assert len(flat) == len(set(flat)), "no transformed column claimed twice"


def test_uncovered_transformed_column_still_fails_loudly():
    """A mapping that under-covers the space must never be used silently."""
    preprocessor = _bank_preprocessor()
    transformed = preprocessor.transform(_bank_frame())
    groups = _build_raw_feature_groups(preprocessor)
    groups.pop("loan_amount")

    with pytest.raises(ValueError, match="Could not map transformed columns"):
        _validate_feature_groups(
            groups, transformed.shape[1], raw_feature_names=BANK_FEATURES
        )


def test_unknown_raw_feature_in_the_mapping_still_fails_loudly():
    preprocessor = _bank_preprocessor()
    transformed = preprocessor.transform(_bank_frame())
    groups = _build_raw_feature_groups(preprocessor)
    groups["scraped_postcode"] = [transformed.shape[1]]

    with pytest.raises(ValueError, match="not in the known raw schema"):
        _validate_feature_groups(
            groups, transformed.shape[1] + 1, raw_feature_names=BANK_FEATURES
        )


def test_feature_groups_still_tile_the_real_german_credit_preprocessor():
    """The unchanged path's mapping is still exact and still validates by default."""
    from app.models.model import load as load_real_model

    preprocessor, _ = _split_pipeline(load_real_model())
    groups = _build_raw_feature_groups(preprocessor)
    n_transformed = preprocessor.transform(_default_feature_frame()).shape[1]

    assert set(groups) == set(RAW_FEATURES)
    _validate_feature_groups(groups, n_transformed)
    _validate_feature_groups(groups, n_transformed, raw_feature_names=RAW_FEATURES)


def test_dropped_columns_are_excluded_from_the_mapping():
    """remainder='drop' means those columns never reach the model at all."""
    preprocessor = _bank_preprocessor(remainder="drop")
    transformed = preprocessor.transform(_bank_frame())
    groups = _build_raw_feature_groups(preprocessor)

    assert set(groups) == set(BANK_CATEGORICAL)
    _validate_feature_groups(
        groups, transformed.shape[1], raw_feature_names=BANK_FEATURES
    )


def test_scaler_block_is_column_preserving_under_a_foreign_schema():
    """An explicitly-named numeric block groups one-to-one, like German Credit's."""
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BANK_CATEGORICAL),
            ("num", StandardScaler(), BANK_NUMERIC),
        ],
        remainder="drop",
    )
    ct.fit(_bank_frame())
    groups = _build_raw_feature_groups(ct)

    assert set(groups) == set(BANK_FEATURES)
    _validate_feature_groups(
        groups, ct.transform(_bank_frame()).shape[1], raw_feature_names=BANK_FEATURES
    )


# ===========================================================================
# 13. Column reordering behaviour is preserved
# ===========================================================================


def test_reordering_realigns_to_the_default_schema_order():
    _, X_test, _, _ = _german_credit_splits()
    rows = X_test.reset_index(drop=True).head(3)
    shuffled = rows[list(reversed(RAW_FEATURES))]

    resolved = _resolve_feature_frame({"feature_matrix": shuffled})

    assert list(resolved.columns) == list(RAW_FEATURES)
    pd.testing.assert_frame_equal(resolved, rows[RAW_FEATURES].reset_index(drop=True))


def test_reordering_realigns_to_a_foreign_schema_order(bank_context):
    """Same realignment guarantee for an adapter's declared order."""
    shuffled = _bank_frame()[list(reversed(BANK_FEATURES))]

    resolved = _resolve_feature_frame(
        {"feature_matrix": shuffled}, context=bank_context
    )

    assert list(resolved.columns) == BANK_FEATURES
    pd.testing.assert_frame_equal(
        resolved, _bank_frame()[BANK_FEATURES].reset_index(drop=True)
    )


def test_foreign_row_records_are_accepted_like_german_credit_ones(bank_context):
    """The API serialises feature_matrix as row records; both forms must work."""
    records = _bank_frame().to_dict(orient="records")

    resolved = _resolve_feature_frame(
        {"feature_matrix": records}, context=bank_context
    )

    assert list(resolved.columns) == BANK_FEATURES
    assert len(resolved) == 6


def test_resolve_rejects_a_foreign_matrix_under_the_default_context():
    """Requirement: explain()'s public no-adapter path still gates the schema."""
    with pytest.raises(ValueError, match="Incompatible feature schema"):
        _resolve_feature_frame({"feature_matrix": _bank_frame()})


# ===========================================================================
# 14. The no-adapter explain() path is numerically unchanged
# ===========================================================================


def test_explain_accepts_an_adapter_as_a_keyword_only_parameter():
    """UPDATED (Phase 4): the adapter parameter now exists.

    This test previously asserted the opposite -- that Step 3 had not yet
    added it. Keyword-only is deliberate: every existing positional call
    ``explain(model_output, method)`` keeps working untouched.
    """
    import inspect

    signature = inspect.signature(explain)
    assert list(signature.parameters) == ["model_output", "method", "adapter"]

    adapter_param = signature.parameters["adapter"]
    assert adapter_param.kind is inspect.Parameter.KEYWORD_ONLY
    assert adapter_param.default is None


def test_no_adapter_shap_is_still_exactly_additive():
    """The strongest available correctness check on the unchanged path.

    sum(contributions) + expected_value == decision_function(x) is the
    property that makes raw-feature aggregation valid. If the refactor had
    disturbed the background, the grouping, or the column order, this is
    where it would show.
    """
    from app.models.model import load as load_real_model

    pipeline = load_real_model()
    rows = _default_feature_frame().head(5)
    result = explain(model_output={"feature_matrix": rows}, method="shap")

    preprocessor, classifier = _split_pipeline(pipeline)
    background = preprocessor.transform(_training_frame())
    expected_value = float(
        np.mean(classifier.decision_function(np.asarray(background)))
    )
    actual = classifier.decision_function(
        np.asarray(preprocessor.transform(rows))
    )

    for i, item in enumerate(result["per_instance"]):
        total = sum(item["contributions"].values()) + expected_value
        assert total == pytest.approx(float(actual[i]), abs=1e-9)


def test_no_adapter_shap_global_importance_matches_the_pre_step3_values():
    """Regression pin: values measured on this artifact BEFORE Step 3 landed.

    Recorded from the default ``explain(method='shap')`` run on the committed
    German Credit artifact. Any drift in the background split, the masker, the
    grouping, or the column order moves these numbers.
    """
    result = explain(method="shap")
    importance = result["global_importance"]

    assert set(importance) == set(RAW_FEATURES)
    assert importance["status_checking_account"] == pytest.approx(
        0.6148128368804278, abs=1e-11
    )
    assert importance["duration_months"] == pytest.approx(
        0.22136306321736124, abs=1e-12
    )


def test_no_adapter_explain_output_shape_is_unchanged():
    result = explain(method="shap")

    assert set(result) == {"method", "per_instance", "global_importance", "is_mock"}
    assert result["method"] == "shap"
    assert result["is_mock"] is False
    assert len(result["per_instance"]) == DEFAULT_EXPLAIN_ROWS
    assert "model_id" not in result, "adding model_id is Step 4's change, not Step 3's"
    for item in result["per_instance"]:
        assert set(item["contributions"]) == set(RAW_FEATURES)


def test_no_adapter_lime_constants_are_unchanged():
    """Seed and sample count drive LIME's numbers; both must stay pinned."""
    import inspect

    explain_module = _explain_module()

    assert explain_module.LIME_NUM_SAMPLES == 5000
    assert explain_module.DEFAULT_EXPLAIN_ROWS == 20

    source = inspect.getsource(explain_module._explain_lime)
    assert "random_state=42" in source
    assert 'class_names=["good_credit", "bad_credit"]' in source


def test_no_adapter_lime_is_deterministic_and_raw_scoped():
    rows = _default_feature_frame().head(2)
    first = explain(model_output={"feature_matrix": rows}, method="lime")
    second = explain(model_output={"feature_matrix": rows}, method="lime")

    assert first["global_importance"] == second["global_importance"]
    assert set(first["global_importance"]) == set(RAW_FEATURES)


@pytest.mark.parametrize(
    "bad_method, match",
    [("permutation", "Unsupported explainability method"), ("", "Unsupported")],
)
def test_existing_public_errors_are_unchanged(bad_method, match):
    with pytest.raises(ValueError, match=match):
        explain(method=bad_method)


def test_existing_non_dict_and_empty_matrix_errors_are_unchanged():
    with pytest.raises(ValueError, match="model_output must be a dict"):
        explain(model_output="not a dict")
    with pytest.raises(ValueError, match="empty"):
        explain(model_output={"feature_matrix": pd.DataFrame()})


def test_german_credit_public_path_is_untouched_by_context_plumbing():
    """Explicitly passing the default context must equal passing nothing."""
    rows = _default_feature_frame().head(3)

    implicit = _resolve_feature_frame({"feature_matrix": rows})
    explicit = _resolve_feature_frame(
        {"feature_matrix": rows}, context=_DEFAULT_CONTEXT
    )

    pd.testing.assert_frame_equal(implicit, explicit)


# ===========================================================================
# The no-adapter path must stay decoupled from capability routing
# ===========================================================================


def test_no_adapter_path_does_not_consult_the_capability_layer():
    """UPDATED (Phase 4): the module now routes on capabilities WITH an adapter.

    This test previously asserted the module never imported the capability
    layer at all. That is no longer the contract -- what still must hold is
    narrower and more useful: the DEFAULT path must not go through capability
    routing, so its behaviour cannot be changed by a routing decision.

    Asserted on the source of ``explain()`` itself, where the adapter branch
    returns before any default-path work begins.
    """
    import inspect

    source = inspect.getsource(_explain_module().explain)

    adapter_branch = source.index("if adapter is not None")
    default_work = source.index("_resolve_feature_frame")
    assert adapter_branch < default_work, (
        "the adapter branch must return before the default path runs, so the "
        "no-adapter path never touches capability routing"
    )
    assert "select_explainer" not in source, (
        "explain() itself must not route; it delegates to "
        "_explain_with_adapter only when an adapter was supplied"
    )


def test_capability_layer_still_names_no_concrete_estimator_class():
    """Requirement 35, re-asserted after the execution layer started using it.

    The execution layer now imports capability.py's structural probes. Those
    probes must still be structural -- if an estimator class name had crept in
    to make routing easier, the model-agnostic guarantee would be gone.
    """
    import inspect

    from app.explainability import capability

    source = inspect.getsource(capability)
    for token in (
        "LogisticRegression",
        "RandomForestClassifier",
        "XGBClassifier",
        "logistic_regression",
        "random_forest",
        "xgboost",
    ):
        assert token not in source, f"capability.py references {token!r}"
    assert "from sklearn" not in source
    assert "import sklearn" not in source
