"""Explainability module (owner: Manas).

SHAP and LIME explanations for whichever credit-scoring model is being
assured. There is no dummy model anywhere here: every number is derived from
a real fitted model's own coefficients, its own trees, or its own
``predict_proba``. This module never trains a model.

TWO ENTRY PATHS
---------------
1. ``explain(...)`` -- the DEFAULT path. Explains the German Credit pipeline
   from ``app.models.model.load()``, gates input on that schema, and behaves
   exactly as it did in Phase 1. Four other modules consume it, so it is a
   compatibility surface and its numbers are held byte-identical by test.
2. ``explain(..., adapter=X)`` -- the ADAPTER-AWARE path. The adapter is the
   sole source of the model, its raw schema, its reference data and its
   identity, so a request to explain model X can never return model Y's
   numbers under X's label.

EXPLAINER SELECTION IS CAPABILITY-BASED, NOT CLASS-BASED
--------------------------------------------------------
``app/explainability/capability.py`` decides which explainer applies, from
the model's OBSERVABLE STRUCTURE -- does it expose fitted coefficients, an
ensemble, a booster; does the adapter promise probabilities, reference data,
batch scoring. No estimator class is imported or type-checked, and neither
``model_type`` nor ``model_id`` is ever read to pick a code path. A model
family this repository has never seen is routed correctly if it exposes
recognisable structure, and routed to an approximate black-box explainer if
it does not -- never rejected for being unfamiliar.

This module only EXECUTES that decision.

FEATURE SPACES (there are two, and they must not be confused)
-------------------------------------------------------------
- RAW: the columns the model accepts as input -- the 20 German Credit
  columns by default, or ``adapter.feature_names`` otherwise. This is the
  only space that ever appears in this module's output.
- TRANSFORMED: whatever the model's preprocessing produces (e.g. 61 columns
  for German Credit: one-hot for 13 categoricals + scaled for 7 numerics).
  An internal detail, deliberately never exposed to callers.

Exact SHAP works in the transformed space and is aggregated back to raw
features by summing the columns each raw feature produced -- valid because
SHAP values are additive by construction. LIME and KernelSHAP work directly
in the raw space. All paths return raw feature names.

SCALE IS A PROPERTY OF THE EXPLAINER, NOT OF THE METHOD
-------------------------------------------------------
This is the easiest thing here to get wrong. "SHAP" does not imply one scale:

- LinearExplainer -> LOG-ODDS. Additive to the decision function:
  ``sum(contributions) + expected_value == decision_function(x)``.
- TreeExplainer   -> PROBABILITY. Additive to ``predict_proba``. Verified at
  runtime rather than assumed, because a gradient-boosted model's tree SHAP
  is on the raw margin instead -- emitting margin numbers labelled
  'probability' would be a silent, plausible-looking lie.
- KernelExplainer -> PROBABILITY, approximate. A sampled estimate of the same
  quantity, computed through the prediction interface only.
- LIME            -> PROBABILITY, surrogate. Local surrogate weights, not an
  attribution of the model itself.

So two models' SHAP values can be in different units. Every adapter-aware
result carries its own ``scale``; read it rather than inferring one from
``method``.

SHAP AND LIME ARE NOT INTERCHANGEABLE
-------------------------------------
Two separate warnings, and the second is the easier one to get wrong:

1. MAGNITUDES are not comparable. Never plot the two on a shared axis, and
   never average, subtract, or otherwise combine them.
2. RANKINGS can also diverge. Measured on the German Credit model, the two
   methods' global-importance rank correlation is only about 0.5 -- they
   agree loosely, not closely. Treat SHAP and LIME as two independent views.
   A feature ranking highly under BOTH is NOT thereby corroborated: the two
   methods answer different questions (exact additive attribution vs. local
   surrogate fit), so agreement is informative but never confirmatory, and
   disagreement is expected rather than a defect.

WHEN A MODEL CANNOT HONESTLY BE EXPLAINED
-----------------------------------------
The adapter-aware path returns a structured unavailable result --
``available=False``, empty containers, and ``limitations`` saying why. Never
``None``, never a bare exception for the caller to interpret, and never a
zero-filled table, which would read as "no feature mattered" (a claim about
the model) instead of "this could not be computed" (a fact about tooling).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from app.explainability.capability import (
    EXPLAINER_KERNEL,
    EXPLAINER_LIME,
    EXPLAINER_LINEAR,
    EXPLAINER_TREE,
    ExplainerCapability,
    ExplainerDecision,
    detect as detect_capability,
    final_estimator,
    fitted_column_transformer,
    is_pipeline_like,
    select_explainer,
)
from app.models.model import load as load_real_model
from app.models.preprocessing import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    load_dataset,
    preprocess,
    split_data,
)

# The raw feature space this module explains. Sourced from Namitha's module
# so the two can never drift apart.
RAW_FEATURES: List[str] = list(FEATURE_COLUMNS)

# Rows explained when the caller supplies no data. LIME costs ~0.25s per row
# against the real pipeline, so the default sample is deliberately small;
# callers wanting more pass their own feature_matrix.
DEFAULT_EXPLAIN_ROWS = 20

# LIME's own default. Kept explicit so the cost/stability tradeoff is visible.
LIME_NUM_SAMPLES = 5000

SUPPORTED_METHODS = ("shap", "lime")

# Project-wide label convention: 0 = GOOD, 1 = BAD = the positive class, and
# every probability in this platform is P(class == 1) == P(BAD).
#
# Declared here rather than imported from app/models/preprocessing.py on
# purpose: it is a PLATFORM convention that the German Credit module and
# app/synthetic_bank/data_generator.py both independently state, not a German
# Credit schema detail. Importing it from the German Credit module would
# re-couple the adapter-aware path to that one model. A test asserts this
# constant still agrees with both modules, so the duplication cannot drift.
POSITIVE_CLASS = 1

# KernelSHAP cost is (background rows x nsamples) model calls per explained
# row, so both are bounded explicitly. These are the knobs that decide whether
# a black-box explanation is minutes or hours, and they are the reason
# capability.py refuses KernelSHAP for an adapter that cannot batch.
KERNEL_BACKGROUND_SAMPLES = 25
KERNEL_NUM_SAMPLES = 100

# Rows explained by a black-box (KernelSHAP) run when the caller supplies none.
# Far below DEFAULT_EXPLAIN_ROWS because each row costs thousands of model
# calls rather than one matrix multiply.
KERNEL_DEFAULT_ROWS = 3

# The schema-source sentence used when no adapter is supplied. Preserved
# verbatim so the default path's error text is byte-identical to Phase 1's.
_DEFAULT_SCHEMA_SOURCE = (
    "Explainability explains the real model from app.models.model.load(), "
    "which accepts the German Credit raw schema defined in "
    "app/models/preprocessing.py FEATURE_COLUMNS."
)


@dataclass(frozen=True)
class _SchemaContext:
    """Which raw feature space is being explained, and where its data lives.

    WHY THIS EXISTS (Step 3)
    ------------------------
    The data-preparation helpers in this module used to read the module-level
    ``RAW_FEATURES`` and the German Credit dataset directly, which made them
    unusable for any other model: a foreign schema was rejected before any
    model was touched, and the reference distribution was always German
    Credit's training split regardless of which model was being explained.

    Every such helper now takes a context instead. The context bundles the
    three things that actually vary between models -- the raw feature names,
    the reference data, and the fitted artifact -- so generalising them did
    not require threading an adapter through every signature.

    ``adapter is None`` is the DEFAULT path and is deliberately privileged:
    it reproduces Phase 1 behaviour exactly, down to the wording of the
    errors. An adapter context is built by ``_context_for()`` and reaches
    here from ``explain(adapter=...)``.

    ``subject`` and ``schema_source`` exist so a schema error describes the
    model actually being validated. A foreign-schema failure must not tell
    the operator about German Credit's FEATURE_COLUMNS -- that would name the
    wrong contract and send them to the wrong file.
    """

    feature_names: Tuple[str, ...]
    subject: str
    schema_source: str
    adapter: Optional[Any] = None

    @property
    def names(self) -> List[str]:
        """The raw feature names as a fresh, ordered, mutable list."""
        return list(self.feature_names)


_DEFAULT_CONTEXT = _SchemaContext(
    feature_names=tuple(RAW_FEATURES),
    subject="the real credit model",
    schema_source=_DEFAULT_SCHEMA_SOURCE,
    adapter=None,
)


def _context_for(adapter: Optional[Any] = None) -> _SchemaContext:
    """Build the context for ``adapter``, or the German Credit default.

    Passing ``None`` returns the shared default context, so every existing
    caller keeps Phase 1 behaviour without opting in to anything.
    """
    if adapter is None:
        return _DEFAULT_CONTEXT

    model_id = str(getattr(adapter, "model_id", "") or "<unknown>")
    declared = getattr(adapter, "feature_names", None)
    if declared is None or isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ValueError(
            f"Adapter for model '{model_id}' declares no usable feature_names "
            f"(got {type(declared).__name__}). Explainability cannot determine "
            "which raw feature space to explain."
        )
    names = [str(name) for name in declared]
    if not names:
        raise ValueError(
            f"Adapter for model '{model_id}' declares an empty feature_names "
            "list. Explainability cannot explain a model with no features."
        )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Adapter for model '{model_id}' declares duplicate feature "
            f"name(s): {duplicates}. Raw feature names must be unique -- "
            "duplicates make positional attribution ambiguous."
        )

    return _SchemaContext(
        feature_names=tuple(names),
        subject=f"model '{model_id}'",
        schema_source=(
            "This input was validated against the raw feature schema declared "
            f"by the adapter for model '{model_id}' (adapter.feature_names)."
        ),
        adapter=adapter,
    )


def _split_pipeline(pipeline: Pipeline) -> Tuple[Any, Any]:
    """Return the pipeline's (preprocessor, classifier) steps.

    Raises ValueError if the model does not have the two-step shape this
    module depends on -- that is a model-contract change, not a data problem.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError(
            f"Expected app.models.model.load() to return an sklearn Pipeline, "
            f"got {type(pipeline).__name__}."
        )
    steps = dict(pipeline.named_steps)
    missing = [s for s in ("preprocessor", "classifier") if s not in steps]
    if missing:
        raise ValueError(
            f"Model pipeline is missing expected step(s): {missing}. "
            f"Found steps: {list(steps)}. Explainability depends on the "
            "'preprocessor'/'classifier' structure defined in app/models/model.py."
        )
    return steps["preprocessor"], steps["classifier"]


def _build_raw_feature_groups(preprocessor: Any) -> Dict[str, List[int]]:
    """Map each raw feature to the transformed column indices it produced.

    Built from the fitted ColumnTransformer's own structure -- the ordered
    ``transformers_`` list and each OneHotEncoder's ``categories_`` -- rather
    than by parsing generated column-name strings. Raw feature names and
    category codes both contain underscores (``status_checking_account``,
    ``A11``), so string splitting would be ambiguous and fragile.

    Returns an ordered {raw_feature: [transformed indices]} mapping whose
    index ranges tile the transformed space exactly once.
    """
    groups: Dict[str, List[int]] = {}
    position = 0
    for _name, transformer, columns in preprocessor.transformers_:
        if transformer in ("drop", "passthrough") or columns is None:
            continue
        categories = getattr(transformer, "categories_", None)
        if categories is not None:
            # One-hot: each input column expands to len(categories) columns.
            for column, cats in zip(columns, categories):
                width = len(cats)
                groups[column] = list(range(position, position + width))
                position += width
        else:
            # Scalers and friends are column-preserving: one in, one out.
            for column in columns:
                groups[column] = [position]
                position += 1
    return groups


def _validate_feature_groups(
    groups: Dict[str, List[int]],
    n_transformed: int,
    *,
    raw_feature_names: Optional[Sequence[str]] = None,
) -> None:
    """Fail loudly if the raw->transformed mapping does not tile the space.

    A silent mismatch here would misattribute contributions to the wrong
    feature, which is exactly the class of bug this module must never ship.

    ``raw_feature_names`` is the schema the preprocessor's input columns are
    checked against. It defaults to the German Credit schema, so existing
    two-argument calls behave exactly as before; an adapter-aware caller
    passes its own model's names instead. That default was previously
    hardcoded, which is what made a foreign preprocessor fail here even when
    its mapping tiled the transformed space perfectly.

    The tiling invariant itself is schema-independent and is enforced for
    every model: the covered indices must be exactly ``range(n_transformed)``,
    so every transformed column is claimed by exactly one raw feature.
    """
    known = RAW_FEATURES if raw_feature_names is None else [str(n) for n in raw_feature_names]

    covered = sorted(i for indices in groups.values() for i in indices)
    if covered != list(range(n_transformed)):
        raise ValueError(
            "Could not map transformed columns back to raw features: the "
            f"mapping covers {len(covered)} column(s) but the preprocessor "
            f"produced {n_transformed}. The model's preprocessing structure "
            "has changed -- see app/models/model.py build_pipeline()."
        )
    unknown = [f for f in groups if f not in known]
    if unknown:
        raise ValueError(
            f"Preprocessor references features not in the known raw schema: {unknown}."
        )


def _coerce_feature_frame(raw_matrix: Any) -> pd.DataFrame:
    """Accept a DataFrame or a list of row records and return a DataFrame.

    The API serialises ``feature_matrix`` as row records over HTTP (see
    docs/module-interfaces.md, "feature_matrix over HTTP"), so both forms
    reach this module in practice.
    """
    if isinstance(raw_matrix, pd.DataFrame):
        frame = raw_matrix.copy()
    else:
        try:
            frame = pd.DataFrame(raw_matrix)
        except Exception as exc:
            raise ValueError(
                f"Could not interpret feature_matrix as tabular data: {exc}"
            ) from exc
    if frame.empty:
        raise ValueError("feature_matrix is empty -- nothing to explain.")
    return frame


def _validate_raw_schema(
    received: List[str],
    *,
    context: _SchemaContext = _DEFAULT_CONTEXT,
) -> None:
    """Raise unless ``received`` is exactly ``context``'s raw feature set.

    The pipeline selects its columns by name. Explaining a different feature
    set would produce numbers that look real but describe nothing, so we
    refuse loudly instead (CLAUDE.md section 6: mock results must never be
    presented as real results).

    The comparison is on the feature SET, not its order: a caller may supply
    the right columns in any order and is realigned by
    ``_resolve_feature_frame()``. Duplicates are rejected separately, and
    first, because the set comparison alone reports a duplicated column as a
    puzzling count mismatch rather than as the duplicate it actually is.

    Defaults to the German Credit context, so the message an existing caller
    sees is unchanged. Under an adapter context the wording names that model
    and its adapter instead -- sending an operator to German Credit's
    FEATURE_COLUMNS for a schema that was never meant to match it would point
    at the wrong contract and the wrong file.
    """
    expected = context.names
    received = list(received)

    duplicates = sorted({name for name in received if received.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate feature column(s) in the input for {context.subject}: "
            f"{duplicates}.\n"
            f"  Received {len(received)} features: {received}\n"
            "Each raw feature must appear exactly once. A duplicated column "
            "makes it ambiguous which values the model was shown, and which "
            "column an attribution belongs to."
        )

    if set(received) == set(expected) and len(received) == len(expected):
        return

    missing = sorted(set(expected) - set(received))
    unexpected = sorted(set(received) - set(expected))
    raise ValueError(
        f"Incompatible feature schema for {context.subject}.\n"
        f"  Expected {len(expected)} raw features: {expected}\n"
        f"  Received {len(received)} features: {received}\n"
        f"  Missing from input: {missing}\n"
        f"  Unexpected in input: {unexpected}\n"
        f"{context.schema_source}"
    )


def _adapter_reference_frame(context: _SchemaContext) -> pd.DataFrame:
    """The adapter's own reference/background data, validated.

    This is the reference distribution an adapter-aware SHAP/LIME run will
    attribute against, so it is validated rather than trusted. There is
    deliberately NO fallback to the German Credit training split: silently
    substituting one model's data as another model's reference distribution
    would produce confident numbers describing a distribution the model was
    never fitted on -- the exact failure mode Step 1 documented.

    Extra columns are tolerated and dropped (reference batches often carry an
    id or label column alongside the features); MISSING ones are fatal.
    """
    adapter = context.adapter
    if adapter is None:  # pragma: no cover - guarded by both callers
        raise ValueError("_adapter_reference_frame() requires an adapter context.")

    background = adapter.background_data()
    if background is None:
        raise ValueError(
            f"No reference/background data is available for {context.subject}: "
            "adapter.background_data() returned None. SHAP needs a reference "
            "distribution to attribute against and LIME needs a sampling "
            "distribution to perturb within, so neither can run."
        )
    if not isinstance(background, pd.DataFrame):
        raise ValueError(
            f"Reference/background data for {context.subject} must be a pandas "
            f"DataFrame, got {type(background).__name__}. Explainability needs "
            "named columns to align the reference distribution with the "
            "model's raw feature schema."
        )
    if background.empty:
        raise ValueError(
            f"Reference/background data for {context.subject} is empty. An "
            "empty frame is not a reference distribution; any attribution "
            "computed against it would describe nothing."
        )

    columns = list(background.columns)
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(
            f"Reference/background data for {context.subject} has duplicate "
            f"column(s): {duplicates}. Each raw feature must appear exactly "
            "once for the reference distribution to be unambiguous."
        )

    missing = [f for f in context.names if f not in background.columns]
    if missing:
        raise ValueError(
            f"Reference/background data for {context.subject} is missing raw "
            f"feature(s) the model expects: {missing}.\n"
            f"  Expected: {context.names}\n"
            f"  Background columns: {columns}\n"
            "The reference distribution must cover the model's whole raw "
            "feature schema."
        )

    return background[context.names].reset_index(drop=True)


def _default_feature_frame(context: _SchemaContext = _DEFAULT_CONTEXT) -> pd.DataFrame:
    """Rows explained when the caller supplies none.

    For the default (no-adapter) context this is unchanged: the same held-out
    test split ``predict_batch()`` defaults to, so a default explanation
    describes the same rows the model module reports on.

    For an adapter context there is no German Credit fallback -- the rows come
    from that model's own reference data, which is the only data this module
    knows is in that model's feature space.
    """
    if context.adapter is not None:
        return _adapter_reference_frame(context).head(DEFAULT_EXPLAIN_ROWS)

    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    return X_test.reset_index(drop=True).head(DEFAULT_EXPLAIN_ROWS)


def _training_frame(context: _SchemaContext = _DEFAULT_CONTEXT) -> pd.DataFrame:
    """SHAP's background and LIME's sampling distribution.

    Default (no-adapter) context: the German Credit training split, exactly as
    before. Adapter context: that adapter's own validated reference data.
    """
    if context.adapter is not None:
        return _adapter_reference_frame(context)

    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    X_train, _, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    return X_train.reset_index(drop=True)


def _load_fitted_model(context: _SchemaContext = _DEFAULT_CONTEXT) -> Any:
    """The fitted artifact to explain.

    Default context: the persisted German Credit pipeline, via Namitha's
    ``load()``. Adapter context: the adapter's own artifact.

    A ``NotImplementedError`` from an adapter with no local artifact is left
    to propagate rather than being converted into a German Credit fallback.
    Deciding what to do about a model with no inspectable artifact belongs to
    the capability layer (``app/explainability/capability.py``) and to Step 4,
    not here.
    """
    if context.adapter is None:
        return load_real_model()

    model = context.adapter.load_fitted_model()
    if model is None:
        raise ValueError(
            f"Adapter for {context.subject} returned no fitted model from "
            "load_fitted_model(); there is nothing to explain."
        )
    return model


def _resolve_feature_frame(
    model_output: Optional[Dict[str, Any]],
    *,
    context: _SchemaContext = _DEFAULT_CONTEXT,
) -> pd.DataFrame:
    """Decide which rows to explain, validating any caller-supplied matrix.

    Validation and realignment are both done against ``context``'s schema
    rather than the module-level German Credit one, so the same code path
    serves any model. The default context keeps existing behaviour.
    """
    if model_output is not None and not isinstance(model_output, dict):
        raise ValueError(
            "model_output must be a dict shaped like app.models.model.predict_batch() "
            f"output, got {type(model_output).__name__}."
        )

    supplied = model_output.get("feature_matrix") if isinstance(model_output, dict) else None
    if supplied is None:
        return _default_feature_frame(context)

    frame = _coerce_feature_frame(supplied)

    # Prefer the declared feature_names so a mislabelled matrix is caught,
    # rather than trusting the DataFrame's own columns implicitly.
    declared = None
    metadata = model_output.get("model_metadata")
    if isinstance(metadata, dict):
        declared = metadata.get("feature_names")
    received = list(declared) if declared else list(frame.columns)

    _validate_raw_schema(received, context=context)

    if declared is not None and set(declared) != set(frame.columns):
        raise ValueError(
            "model_metadata['feature_names'] does not match feature_matrix columns.\n"
            f"  feature_names: {sorted(declared)}\n"
            f"  matrix columns: {sorted(frame.columns)}"
        )

    # Reorder to the model's training order. The ColumnTransformer selects
    # by name, but ordering the frame explicitly keeps downstream positional
    # work (LIME indices, SHAP grouping) unambiguous.
    return frame[context.names].reset_index(drop=True)


def _as_dense(matrix: Any) -> np.ndarray:
    """OneHotEncoder may return sparse output depending on its settings."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _package(method: str, contributions: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """Build the approved output dict from an (n_rows, n_features) matrix."""
    per_instance = [
        {
            "row_index": int(i),
            "contributions": {
                feature: float(contributions[i, j])
                for j, feature in enumerate(feature_names)
            },
        }
        for i in range(contributions.shape[0])
    ]
    global_importance = {
        feature: float(np.mean(np.abs(contributions[:, j])))
        for j, feature in enumerate(feature_names)
    }
    return {
        "method": method,
        "per_instance": per_instance,
        "global_importance": global_importance,
        "is_mock": False,
    }


def _explain_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    context: _SchemaContext = _DEFAULT_CONTEXT,
) -> Dict[str, Any]:
    """Exact SHAP values for the real model, aggregated to raw features.

    STRATEGY -- why the Pipeline is split rather than passed whole:
    ``shap.LinearExplainer`` rejects an sklearn ``Pipeline``
    (``InvalidModelError``). It is also the *right* explainer for the final
    estimator, which is a LogisticRegression -- for a linear model SHAP is
    exact, not approximated. So the pipeline is split:

        raw X --[preprocessor.transform]--> transformed (61)
                                              |
                              LinearExplainer(classifier)
                                              |
                                    phi over 61 columns
                                              |
                              group-sum by source raw feature
                                              |
                                    phi over 20 raw features

    The preprocessor is the model's OWN fitted transformer and the classifier
    is the model's OWN fitted estimator, so the explanation stays tied to the
    real model's predictions.

    AGGREGATION -- why summing is correct, not a heuristic:
    SHAP values are additive by construction:
    ``sum(phi) + expected_value == decision_function(x)``. Summing the columns
    produced by one raw feature therefore yields exactly that raw feature's
    contribution, and additivity survives the aggregation intact (asserted to
    ~1e-9 in the test suite). One-hot columns from a single categorical
    feature are mutually exclusive, so their group sum is that feature's
    contribution for the category the row actually holds.

    Contributions are on the LOG-ODDS scale.
    """
    preprocessor, classifier = _split_pipeline(pipeline)
    raw_features = context.names

    groups = _build_raw_feature_groups(preprocessor)
    background = _as_dense(preprocessor.transform(_training_frame(context)))
    _validate_feature_groups(groups, background.shape[1], raw_feature_names=raw_features)

    transformed = _as_dense(preprocessor.transform(X))

    # An explicit masker keeps the background deterministic instead of letting
    # SHAP silently subsample it to its default 100 rows.
    masker = shap.maskers.Independent(background, max_samples=background.shape[0])
    explainer = shap.LinearExplainer(classifier, masker)
    values = explainer(transformed).values

    ordered = [f for f in raw_features if f in groups]
    aggregated = np.column_stack(
        [values[:, groups[feature]].sum(axis=1) for feature in ordered]
    )
    return _package("shap", aggregated, ordered)


def _lime_category_lookup(preprocessor: Any) -> Dict[str, List[str]]:
    """Category values per categorical feature, taken from the fitted encoder.

    Read off the model's own OneHotEncoder rather than recomputed from the
    dataset, so LIME's category space is exactly the one the model was fitted
    on -- including any category absent from the rows being explained.
    """
    lookup: Dict[str, List[str]] = {}
    for _name, transformer, columns in preprocessor.transformers_:
        categories = getattr(transformer, "categories_", None)
        if categories is None:
            continue
        for column, cats in zip(columns, categories):
            lookup[column] = [str(c) for c in cats]
    return lookup


def _explain_lime(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    context: _SchemaContext = _DEFAULT_CONTEXT,
) -> Dict[str, Any]:
    """LIME explanations for the real model, in the raw feature space.

    STRATEGY:
    LIME needs a numeric matrix, but 13 of the 20 raw features are string
    categoricals. Those are encoded to integer category codes for LIME and
    declared via ``categorical_features``/``categorical_names``, so LIME
    perturbs them by swapping categories rather than by interpolating
    meaningless values between codes. The prediction function decodes each
    perturbed row back to the original string categories and calls the REAL
    pipeline's ``predict_proba``, so LIME explains the actual model end to end.

    DISCRETIZATION (intentional):
    ``discretize_continuous=True`` applies to the 7 numeric features only;
    declared categorical features are never discretized. A numeric
    contribution is therefore the effect of the row's VALUE BAND (e.g.
    "credit_amount <= 1360") rather than a per-unit slope, and is reported
    under the bare feature name. This is kept deliberately: it produces
    genuine per-row attributions. Disabling it would make the surrogate
    return near-identical weights for every row, defeating the purpose of
    ``per_instance``.

    UNSEEN CATEGORIES -- DELIBERATE DIVERGENCE FROM SHAP:
    The model's OneHotEncoder uses ``handle_unknown='ignore'``, so the
    pipeline itself tolerates a category it was never fitted on (it encodes
    as all-zeros for that feature). SHAP therefore explains such a row
    without complaint. LIME here does NOT: it raises ValueError.

    That asymmetry is intentional, not an oversight. LIME's raw-feature
    encoding needs an integer code per category, and an unseen value has
    none. The only ways to continue would be to map it to some known
    category or to report a contribution for it anyway -- both would produce
    a confident-looking explanation of a row the surrogate never actually
    represented. Refusing is deterministic, explicit, and cannot mislead
    (CLAUDE.md section 6: mock results must never be presented as real
    results). This does not change the model's behaviour, only what this
    module is willing to claim about it.

    Contributions are on the PREDICTED-PROBABILITY scale for the "bad credit"
    class (label 1) -- not comparable to SHAP's log-odds magnitudes, and its
    feature ranking may diverge from SHAP's (see the module docstring).
    """
    preprocessor, _classifier = _split_pipeline(pipeline)
    categories = _lime_category_lookup(preprocessor)
    raw_features = context.names

    categorical = [f for f in raw_features if f in categories]
    categorical_idx = [raw_features.index(f) for f in categorical]
    numeric = [f for f in raw_features if f not in categories]

    code_maps = {f: {value: i for i, value in enumerate(categories[f])} for f in categorical}

    def encode(frame: pd.DataFrame) -> np.ndarray:
        encoded = frame[raw_features].copy()
        for feature in categorical:
            mapped = encoded[feature].astype(str).map(code_maps[feature])
            if mapped.isna().any():
                unseen = sorted(set(encoded.loc[mapped.isna(), feature].astype(str)))
                raise ValueError(
                    "LIME cannot explain an unseen categorical value.\n"
                    f"  Feature: '{feature}'\n"
                    f"  Unseen value(s): {unseen}\n"
                    f"  Categories the model was fitted on: {categories[feature]}\n"
                    "LIME's raw-feature encoding requires an integer code per "
                    "category, and an unseen value has none. Substituting a known "
                    "category or reporting a contribution anyway would describe a "
                    "row the surrogate never represented, so this is refused "
                    "instead.\n"
                    "Note: method='shap' CAN explain this row -- the model's "
                    "OneHotEncoder uses handle_unknown='ignore'. Use SHAP for rows "
                    "with out-of-vocabulary categories."
                )
            encoded[feature] = mapped.astype(int)
        return encoded.to_numpy(dtype=float)

    def decode_and_predict(array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(array, columns=raw_features)
        for feature in categorical:
            codes = np.clip(
                np.round(frame[feature].to_numpy()).astype(int),
                0,
                len(categories[feature]) - 1,
            )
            frame[feature] = [categories[feature][c] for c in codes]
        for feature in numeric:
            frame[feature] = pd.to_numeric(frame[feature])
        return pipeline.predict_proba(frame)

    training = encode(_training_frame(context))
    target = encode(X)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training,
        feature_names=raw_features,
        categorical_features=categorical_idx,
        categorical_names={i: categories[raw_features[i]] for i in categorical_idx},
        class_names=["good_credit", "bad_credit"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    contributions = np.zeros((target.shape[0], len(raw_features)), dtype=float)
    for row in range(target.shape[0]):
        explanation = explainer.explain_instance(
            target[row],
            decode_and_predict,
            labels=(1,),
            num_features=len(raw_features),
            num_samples=LIME_NUM_SAMPLES,
        )
        for index, weight in explanation.as_map().get(1, []):
            contributions[row, int(index)] = float(weight)

    return _package("lime", contributions, raw_features)


# ===========================================================================
# ADAPTER-AWARE EXECUTION
#
# Everything below runs ONLY when explain() is given an adapter. The adapter
# is then the sole source of the model, its schema, its reference data and its
# identity -- the default German Credit loader is never consulted, so a
# request to explain model X can never come back with model Y's numbers
# wearing X's label.
#
# WHICH explainer runs is decided entirely by
# app/explainability/capability.py from the model's observable structure. This
# layer only EXECUTES that decision. No estimator class is imported or
# type-checked here, and neither model_type nor model_id is ever read to pick
# a code path.
# ===========================================================================


def _decompose(artifact: Any) -> Tuple[Optional[Any], Any]:
    """Locate (preprocessing, final estimator) structurally.

    NO STEP NAMES. ``_split_pipeline()`` above requires steps literally called
    'preprocessor'/'classifier', which is fine for the German Credit pipeline
    this module was written for but excludes every model that names its steps
    anything else. Position is used instead: the final step predicts, and
    everything before it transforms.

    The prefix is obtained by slicing, which works for a chain of any length
    and composes the steps in order. Returns ``(None, artifact)`` for a bare
    estimator -- there is then no transformation, so raw space and model space
    are the same space.
    """
    if not is_pipeline_like(artifact):
        return None, artifact

    estimator = final_estimator(artifact)
    try:
        prefix = artifact[:-1]
    except Exception:
        return None, estimator
    if not hasattr(prefix, "transform"):
        return None, estimator
    return prefix, estimator


def _raw_feature_groups(
    artifact: Any,
    preprocessing: Optional[Any],
    context: _SchemaContext,
) -> Optional[Dict[str, List[int]]]:
    """Raw-feature -> transformed-column-index mapping, or None if unsafe.

    Reuses the existing ``_build_raw_feature_groups()`` -- the grouping
    algorithm is not duplicated for the adapter path.

    RETURNS None RATHER THAN A GUESS. The mapping is index-based against the
    column transformer's own output, so it is only valid if nothing reshapes
    the columns afterwards. A chain of [ColumnTransformer, Scaler, estimator]
    would still produce a tidy-looking mapping whose indices no longer point
    where they claim. Rather than aggregate along a mapping that might be
    silently shifted, this reports "cannot map" and the caller degrades to a
    black-box explainer that needs no mapping at all.
    """
    if preprocessing is None:
        # Bare estimator: the model consumes raw features directly, so the
        # mapping is the identity.
        return {name: [i] for i, name in enumerate(context.names)}

    transformer = fitted_column_transformer(artifact)
    if transformer is None:
        return None

    steps = getattr(preprocessing, "steps", None)
    if steps is None or len(steps) != 1 or steps[0][1] is not transformer:
        # More than one preprocessing step, or the column transformer is not
        # the one doing the final reshaping: indices cannot be trusted.
        return None

    return _build_raw_feature_groups(transformer)


def _positive_class_index(estimator: Any, n_outputs: int) -> int:
    """Which output axis carries P(class == 1) == P(BAD).

    Looked up in ``classes_`` rather than assumed to be index 1. A model whose
    ``classes_`` is ordered differently -- or which was fitted on labels in
    another order -- would otherwise have its GOOD-class contributions
    reported as BAD-class contributions, sign-flipped and completely wrong,
    with nothing in the output to reveal it.

    A single-output model needs no selection. Anything else without a usable
    ``classes_`` raises instead of guessing.
    """
    if n_outputs == 1:
        return 0

    # NOTE: classes_ is typically a numpy array, so `x or []` would raise on
    # its ambiguous truth value -- check for None explicitly instead.
    declared = getattr(estimator, "classes_", None)
    classes = [] if declared is None else list(declared)
    if POSITIVE_CLASS in classes:
        index = classes.index(POSITIVE_CLASS)
        if index < n_outputs:
            return index
        raise ValueError(
            f"Model's classes_ places the positive class {POSITIVE_CLASS} at "
            f"index {index}, but the explainer produced only {n_outputs} "
            "output(s). Refusing to guess which output is P(BAD)."
        )

    raise ValueError(
        f"Cannot identify which of the explainer's {n_outputs} outputs is "
        f"P(class == {POSITIVE_CLASS}) == P(BAD): the model exposes "
        f"classes_={classes or None}. Guessing an index could report the "
        "favourable class's contributions as the unfavourable class's, so "
        "this is refused."
    )


def _select_class_axis(values: Any, base_values: Any, estimator: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a possibly multi-class SHAP result to the positive class.

    SHAP returns one of three shapes depending on the model family: a list of
    per-class arrays, a 3-D array with classes last, or a 2-D array for a
    single-output model. All three are handled by shape, not by model type.
    """
    if isinstance(values, list):
        index = _positive_class_index(estimator, len(values))
        chosen = np.asarray(values[index])
        base = np.asarray(base_values)
        base = base[index] if base.ndim and base.shape[0] == len(values) else base
        return chosen, np.atleast_1d(base).ravel()

    array = np.asarray(values)
    base = np.asarray(base_values)
    if array.ndim == 3:
        index = _positive_class_index(estimator, array.shape[2])
        chosen = array[:, :, index]
        base = base[:, index] if base.ndim == 2 else np.atleast_1d(base).ravel()
        return chosen, np.atleast_1d(base).ravel()

    return array, np.atleast_1d(base).ravel()


def _aggregate_to_raw(
    values: np.ndarray,
    groups: Dict[str, List[int]],
    context: _SchemaContext,
) -> Tuple[np.ndarray, List[str]]:
    """Sum transformed-column contributions into their source raw features.

    Valid because SHAP values are additive by construction: the columns one
    raw feature produced sum to exactly that feature's contribution.
    """
    ordered = [f for f in context.names if f in groups]
    aggregated = np.column_stack(
        [values[:, groups[feature]].sum(axis=1) for feature in ordered]
    )
    return aggregated, ordered


def _encode_raw_space(
    frame: pd.DataFrame,
    context: _SchemaContext,
    categories: Dict[str, List[Any]],
) -> np.ndarray:
    """Encode a raw frame to a numeric matrix for a black-box explainer.

    KernelSHAP and LIME both need numbers, but a raw schema may contain string
    categoricals. Categories are encoded to integer codes using ``categories``
    -- derived from the reference data, which is the only category vocabulary
    available for a model whose internals cannot be inspected.

    An unseen category raises rather than being mapped to a neighbouring code:
    a substituted category would describe a row the explainer never actually
    perturbed around.
    """
    encoded = frame[context.names].copy()
    for feature, values in categories.items():
        lookup = {str(value): code for code, value in enumerate(values)}
        mapped = encoded[feature].astype(str).map(lookup)
        if mapped.isna().any():
            unseen = sorted(set(encoded.loc[mapped.isna(), feature].astype(str)))
            raise ValueError(
                "Cannot explain a categorical value absent from the reference "
                f"data.\n  Feature: '{feature}'\n  Unseen value(s): {unseen}\n"
                f"  Reference categories: {values}\n"
                "A black-box explainer's category vocabulary comes from the "
                "reference distribution; substituting a known category would "
                "describe a row the explainer never perturbed around."
            )
        encoded[feature] = mapped.astype(int)
    return encoded.to_numpy(dtype=float)


def _raw_space_categories(
    background: pd.DataFrame, context: _SchemaContext
) -> Dict[str, List[Any]]:
    """Category vocabulary per non-numeric raw feature, from reference data.

    Detected by dtype, not by a configured categorical list, so it works for
    any adapter's schema. Sorted for determinism -- the codes must be the same
    on every run or two explanations of the same row would differ.
    """
    categories: Dict[str, List[Any]] = {}
    for feature in context.names:
        column = background[feature]
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            continue
        categories[feature] = sorted({str(v) for v in column.dropna().unique()})
    return categories


def _decode_raw_space(
    array: np.ndarray,
    context: _SchemaContext,
    categories: Dict[str, List[Any]],
    dtypes: Dict[str, Any],
) -> pd.DataFrame:
    """Inverse of ``_encode_raw_space``: numeric matrix back to a raw frame.

    Perturbed category codes are clipped into range, because a perturbation
    explainer legitimately proposes values between codes; the clip maps each
    back onto a real category the model can actually score.
    """
    frame = pd.DataFrame(np.asarray(array, dtype=float), columns=context.names)
    for feature, values in categories.items():
        codes = np.clip(
            np.round(frame[feature].to_numpy()).astype(int), 0, len(values) - 1
        )
        frame[feature] = [values[c] for c in codes]
    for feature in context.names:
        if feature in categories:
            continue
        target = dtypes.get(feature)
        if target is not None and pd.api.types.is_integer_dtype(target):
            frame[feature] = np.round(frame[feature]).astype(target)
    return frame


def _adapter_probability_fn(adapter: Any, context: _SchemaContext):
    """A 1-D P(BAD) callable over the adapter's raw feature space.

    This is the ONLY way a black-box explanation reaches the model: through
    the adapter's own prediction interface, never by inspecting internals.
    The adapter contract already guarantees a 1-D positive-class vector, so no
    class-axis selection is needed or attempted here.
    """

    def probability(frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(adapter.predict_proba(frame[context.names]), dtype=float)
        values = values.ravel()
        if values.shape[0] != len(frame):
            raise ValueError(
                f"Adapter returned {values.shape[0]} probabilities for "
                f"{len(frame)} row(s). A misaligned probability vector would "
                "attribute one row's score to another."
            )
        return values

    return probability


def _run_linear_shap(
    artifact: Any, X: pd.DataFrame, context: _SchemaContext
) -> Tuple[np.ndarray, List[str]]:
    """Exact additive attribution for a linear final estimator (log-odds)."""
    preprocessing, estimator = _decompose(artifact)
    groups = _raw_feature_groups(artifact, preprocessing, context)
    if groups is None:
        raise _DecompositionUnsafe(
            "the fitted column transformer could not be mapped to raw features"
        )

    background_frame = _training_frame(context)
    if preprocessing is None:
        background = np.asarray(background_frame.to_numpy(dtype=float))
        transformed = np.asarray(X.to_numpy(dtype=float))
    else:
        background = _as_dense(preprocessing.transform(background_frame))
        transformed = _as_dense(preprocessing.transform(X))

    _validate_feature_groups(
        groups, background.shape[1], raw_feature_names=context.names
    )

    masker = shap.maskers.Independent(background, max_samples=background.shape[0])
    values = shap.LinearExplainer(estimator, masker)(transformed).values
    values, _base = _select_class_axis(values, 0.0, estimator)
    return _aggregate_to_raw(values, groups, context)


def _run_tree_shap(
    artifact: Any, X: pd.DataFrame, context: _SchemaContext
) -> Tuple[np.ndarray, List[str]]:
    """Exact additive attribution for a tree/ensemble final estimator.

    SCALE VERIFICATION, NOT ASSUMPTION: tree SHAP is additive to the model's
    output, but WHICH output depends on the family -- probabilities for a
    forest of probability-leaf trees, raw margin for a gradient-boosted model.
    The capability layer labels this decision ``scale='probability'``, so that
    claim is checked here against the estimator's own ``predict_proba`` and
    fails loudly if it does not hold. Emitting margin-scale numbers labelled
    'probability' would be a silent, plausible-looking lie.
    """
    preprocessing, estimator = _decompose(artifact)
    groups = _raw_feature_groups(artifact, preprocessing, context)
    if groups is None:
        raise _DecompositionUnsafe(
            "the fitted column transformer could not be mapped to raw features"
        )

    if preprocessing is None:
        transformed = np.asarray(X.to_numpy(dtype=float))
        n_transformed = transformed.shape[1]
    else:
        transformed = _as_dense(preprocessing.transform(X))
        n_transformed = _as_dense(
            preprocessing.transform(_training_frame(context))
        ).shape[1]

    _validate_feature_groups(groups, n_transformed, raw_feature_names=context.names)

    explanation = shap.TreeExplainer(estimator)(transformed)
    values, base = _select_class_axis(
        explanation.values, explanation.base_values, estimator
    )

    _verify_probability_scale(values, base, estimator, transformed)
    return _aggregate_to_raw(values, groups, context)


def _verify_probability_scale(
    values: np.ndarray, base: np.ndarray, estimator: Any, transformed: np.ndarray
) -> None:
    """Fail unless the tree contributions really are on the probability scale."""
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            "Tree SHAP was selected but the estimator exposes no "
            "predict_proba(), so its contributions cannot be confirmed to be "
            "on the probability scale."
        )

    proba = np.asarray(estimator.predict_proba(transformed))
    index = _positive_class_index(estimator, proba.shape[1])
    expected = proba[:, index]
    actual = values.sum(axis=1) + np.asarray(base).ravel()

    largest = float(np.max(np.abs(actual - expected))) if len(expected) else 0.0
    if largest > 1e-6:
        raise ValueError(
            "Tree SHAP contributions are NOT additive to predict_proba "
            f"(largest deviation {largest:.3e}). The capability layer labels "
            "this explanation scale='probability', and these values do not "
            "satisfy that claim -- they are most likely on the model's raw "
            "margin (log-odds) scale, as gradient-boosted models produce. "
            "Refusing rather than mislabelling the scale."
        )


def _run_kernel_shap(
    adapter: Any, X: pd.DataFrame, context: _SchemaContext
) -> Tuple[np.ndarray, List[str]]:
    """Approximate, model-agnostic attribution through the prediction interface.

    The model's internals are never touched: KernelSHAP perturbs raw features
    and observes the adapter's probability output. That is what makes it work
    for a remote model and an unrecognised estimator family alike, and also
    what makes it approximate -- it estimates the same quantity exact SHAP
    would compute, by sampling.

    Cost is bounded by KERNEL_BACKGROUND_SAMPLES x KERNEL_NUM_SAMPLES model
    calls per explained row, issued as batched calls. This is precisely why
    capability.py requires ``batch_scoring`` before allowing this path.
    """
    background_frame = _training_frame(context)
    categories = _raw_space_categories(background_frame, context)
    dtypes = {f: background_frame[f].dtype for f in context.names}

    background = _encode_raw_space(background_frame, context, categories)
    if background.shape[0] > KERNEL_BACKGROUND_SAMPLES:
        # Deterministic stride, not a random sample: two runs must agree.
        step = background.shape[0] // KERNEL_BACKGROUND_SAMPLES
        background = background[::step][:KERNEL_BACKGROUND_SAMPLES]

    target = _encode_raw_space(X, context, categories)
    probability = _adapter_probability_fn(adapter, context)

    def scores(array: np.ndarray) -> np.ndarray:
        frame = _decode_raw_space(array, context, categories, dtypes)
        return probability(frame)

    explainer = shap.KernelExplainer(scores, background)
    values = np.asarray(
        explainer.shap_values(target, nsamples=KERNEL_NUM_SAMPLES, silent=True)
    )
    if values.ndim == 3:
        values = values[:, :, _positive_class_index(adapter, values.shape[2])]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values, context.names


def _run_adapter_lime(
    adapter: Any, X: pd.DataFrame, context: _SchemaContext
) -> Tuple[np.ndarray, List[str]]:
    """Local surrogate weights, computed through the adapter's probabilities.

    PROBABILITY SHAPE: the adapter contract returns a 1-D ``P(BAD)`` vector,
    but LIME's classification mode requires a per-class matrix. The vector is
    widened to ``[[1 - p, p], ...]`` -- column 0 = GOOD, column 1 = BAD --
    and ``labels=(1,)`` selects BAD. Handing LIME the 1-D vector directly, or
    widening it in the other column order, would silently explain the
    favourable class while labelling it the unfavourable one.
    """
    background_frame = _training_frame(context)
    categories = _raw_space_categories(background_frame, context)
    dtypes = {f: background_frame[f].dtype for f in context.names}

    training = _encode_raw_space(background_frame, context, categories)
    target = _encode_raw_space(X, context, categories)

    categorical_idx = [context.names.index(f) for f in categories]
    probability = _adapter_probability_fn(adapter, context)

    def predict_two_column(array: np.ndarray) -> np.ndarray:
        frame = _decode_raw_space(array, context, categories, dtypes)
        p_bad = probability(frame)
        return np.column_stack([1.0 - p_bad, p_bad])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training,
        feature_names=context.names,
        categorical_features=categorical_idx,
        categorical_names={i: categories[context.names[i]] for i in categorical_idx},
        class_names=["good_credit", "bad_credit"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    contributions = np.zeros((target.shape[0], len(context.names)), dtype=float)
    for row in range(target.shape[0]):
        explanation = explainer.explain_instance(
            target[row],
            predict_two_column,
            labels=(POSITIVE_CLASS,),
            num_features=len(context.names),
            num_samples=LIME_NUM_SAMPLES,
        )
        for index, weight in explanation.as_map().get(POSITIVE_CLASS, []):
            contributions[row, int(index)] = float(weight)

    return contributions, context.names


class _DecompositionUnsafe(Exception):
    """Raised internally when a model's structure cannot be mapped safely.

    Never escapes ``explain()``: it is caught and converted into a degraded
    black-box decision, or into a structured unavailable result.
    """


def _identity(capability: ExplainerCapability) -> Dict[str, Any]:
    """Identity fields, taken only from the adapter the caller passed in."""
    return {
        "model_id": capability.model_id,
        "model_type": capability.model_type,
    }


def _execution_provenance(
    decision: ExplainerDecision, n_background: int
) -> Dict[str, Any]:
    """How the numbers were produced, for the evidence layer to carry.

    ``n_samples`` and ``random_seed`` are None for the exact explainers: they
    draw no samples and have no seed, so reporting a number there would imply
    a sampling process that did not happen. A reader can therefore tell a
    sampled estimate from an exact decomposition from the provenance alone.
    """
    sampled = {
        EXPLAINER_KERNEL: KERNEL_NUM_SAMPLES,
        EXPLAINER_LIME: LIME_NUM_SAMPLES,
    }.get(decision.explainer)
    return {
        "n_background": int(n_background),
        "n_samples": sampled,
        "random_seed": 42 if sampled is not None else None,
    }


def _adapter_result(
    method: str,
    decision: ExplainerDecision,
    capability: ExplainerCapability,
    context: _SchemaContext,
    contributions: np.ndarray,
    feature_names: List[str],
    extra_limitations: Sequence[str] = (),
    n_background: int = 0,
) -> Dict[str, Any]:
    """The adapter-aware output: the legacy four keys plus honest labelling."""
    result = _package(method, contributions, feature_names)
    result.update(_identity(capability))
    result.update(
        {
            "available": True,
            "explainer": decision.explainer,
            "scale": decision.scale,
            "fidelity": decision.fidelity,
            "feature_space": list(context.names),
            "capabilities": capability.to_dict(),
            "limitations": list(decision.limitations) + list(extra_limitations),
            "integration_type": capability.integration_type,
            **_execution_provenance(decision, n_background),
        }
    )
    return result


def _unavailable_result(
    method: str,
    decision: ExplainerDecision,
    capability: ExplainerCapability,
    context: _SchemaContext,
    extra_limitations: Sequence[str] = (),
) -> Dict[str, Any]:
    """A first-class 'cannot explain this' result.

    NOT an exception, NOT None, and NOT a zero-filled explanation. Empty
    containers plus ``available=False`` plus the reason -- a table of zeros
    would render as "no feature mattered", which is a claim about the model
    rather than an admission about the tooling.

    ``is_mock`` stays False: nothing was fabricated here. is_mock describes
    whether numbers are stand-ins, not whether they exist.
    """
    return {
        "method": method,
        "per_instance": [],
        "global_importance": {},
        "is_mock": False,
        "available": False,
        **_identity(capability),
        "explainer": None,
        "scale": None,
        "fidelity": None,
        "feature_space": list(context.names),
        "capabilities": capability.to_dict(),
        "limitations": list(decision.limitations) + list(extra_limitations),
        "integration_type": capability.integration_type,
        "n_background": 0,
        "n_samples": None,
        "random_seed": None,
    }


def _reject_identity_conflict(model_output: Any, capability: ExplainerCapability) -> None:
    """Refuse a caller-declared identity that contradicts the adapter's.

    Identity is DERIVED from the adapter, never accepted from the caller. If
    a caller nonetheless declares one and it disagrees, that is a real
    disagreement about which model is being explained -- silently preferring
    the adapter's would leave the caller believing its own label was honoured.
    """
    if not isinstance(model_output, dict):
        return
    metadata = model_output.get("model_metadata")
    if not isinstance(metadata, dict):
        return

    for field, actual in (
        ("model_id", capability.model_id),
        ("model_type", capability.model_type),
    ):
        declared = metadata.get(field)
        if declared is not None and str(declared) != str(actual):
            raise ValueError(
                f"Conflicting {field}: model_output['model_metadata']"
                f"['{field}'] is {declared!r} but the adapter being explained "
                f"reports {actual!r}. Explainability derives identity from the "
                "adapter, so it will not relabel this model's explanation as "
                "another model's. Pass the matching adapter, or drop the "
                f"'{field}' from model_metadata."
            )


def _explain_with_adapter(
    model_output: Optional[Dict[str, Any]], method: str, adapter: Any
) -> Dict[str, Any]:
    """Explain exactly the model ``adapter`` wraps, or say why it cannot."""
    context = _context_for(adapter)
    capability = detect_capability(adapter)
    _reject_identity_conflict(model_output, capability)

    decision = select_explainer(capability, method)
    if not decision.available:
        return _unavailable_result(method, decision, capability, context)

    extra: List[str] = []
    black_box = decision.explainer in (EXPLAINER_KERNEL, EXPLAINER_LIME)
    rows = KERNEL_DEFAULT_ROWS if decision.explainer == EXPLAINER_KERNEL else None

    X = _resolve_feature_frame(model_output, context=context)
    if rows is not None and len(X) > rows:
        X = X.head(rows)
        extra.append(
            f"Explained the first {rows} row(s) only: each black-box row costs "
            f"up to {KERNEL_BACKGROUND_SAMPLES * KERNEL_NUM_SAMPLES} model "
            "calls. Pass fewer rows explicitly to control this."
        )

    n_background = len(_training_frame(context))

    if not black_box:
        artifact = _load_fitted_model(context)
        runner = (
            _run_linear_shap
            if decision.explainer == EXPLAINER_LINEAR
            else _run_tree_shap
        )
        try:
            contributions, names = runner(artifact, X, context)
        except _DecompositionUnsafe as exc:
            # The structure the capability layer judged mappable turned out not
            # to be. Degrade to black-box ONLY if the capability record allows
            # it -- never silently keep the 'exact' label.
            return _degrade_to_black_box(
                model_output, method, adapter, capability, context, str(exc)
            )
        return _adapter_result(
            method, decision, capability, context, contributions, names, extra,
            n_background=n_background,
        )

    if decision.explainer == EXPLAINER_KERNEL:
        contributions, names = _run_kernel_shap(adapter, X, context)
        n_background = min(n_background, KERNEL_BACKGROUND_SAMPLES)
    else:
        contributions, names = _run_adapter_lime(adapter, X, context)
    return _adapter_result(
        method, decision, capability, context, contributions, names, extra,
        n_background=n_background,
    )


def _degrade_to_black_box(
    model_output: Optional[Dict[str, Any]],
    method: str,
    adapter: Any,
    capability: ExplainerCapability,
    context: _SchemaContext,
    reason: str,
) -> Dict[str, Any]:
    """Fall back to a black-box explainer when exact decomposition is unsafe.

    Re-decides from a capability record with the mapping marked unavailable,
    so the fallback is the capability layer's decision rather than this
    layer's improvisation -- and so the result carries 'approximate', never a
    stale 'exact'. If the capability record does not permit a black-box run
    either, the result is a structured unavailable.
    """
    import dataclasses

    degraded = dataclasses.replace(capability, has_raw_transformed_mapping=False)
    decision = select_explainer(degraded, method)
    note = (
        f"Exact decomposition was not possible ({reason}), so an approximate "
        "black-box explanation was produced instead."
    )
    if not decision.available:
        return _unavailable_result(method, decision, degraded, context, [note])

    X = _resolve_feature_frame(model_output, context=context)
    if decision.explainer == EXPLAINER_KERNEL and len(X) > KERNEL_DEFAULT_ROWS:
        X = X.head(KERNEL_DEFAULT_ROWS)

    n_background = len(_training_frame(context))
    if decision.explainer == EXPLAINER_KERNEL:
        contributions, names = _run_kernel_shap(adapter, X, context)
        n_background = min(n_background, KERNEL_BACKGROUND_SAMPLES)
    else:
        contributions, names = _run_adapter_lime(adapter, X, context)
    return _adapter_result(
        method, decision, degraded, context, contributions, names, [note],
        n_background=n_background,
    )


def explain(
    model_output: Optional[Dict[str, Any]] = None,
    method: str = "shap",
    *,
    adapter: Optional[Any] = None,
) -> Dict[str, Any]:
    """Explain the real credit model's predictions with SHAP or LIME.

    Public entry point for the explainability module (owner: Manas).

    The model explained is always the trained pipeline returned by
    ``app.models.model.load()``. This module never trains and never
    substitutes a stand-in model, so ``is_mock`` is False in the strong sense:
    real algorithm, real model, real data path.

    Args:
        model_output: Optional dict shaped like ``predict_batch()`` output.
            It is a COMPATIBILITY INPUT that selects WHICH ROWS to explain --
            not the source of the explanation. Only ``feature_matrix`` and
            ``model_metadata['feature_names']`` are read; ``predictions``,
            ``probabilities`` and ``model_metadata['model_type']`` are ignored,
            because the loaded pipeline supplies all of those itself.
            ``feature_matrix`` may be a DataFrame or a list of row records, and
            must carry exactly the 20 raw German Credit features. Any other
            schema raises ValueError rather than returning numbers the model
            cannot legitimately produce. When None, a sample of the held-out
            test split is explained.
        method: "shap" (default) or "lime".
        adapter: optional ``ModelAdapter``. When supplied it is the SOLE
            source of the model, its raw feature schema, its reference data
            and its identity -- the default German Credit loader is never
            consulted, so a request to explain model X cannot return model Y's
            numbers under X's label. Which explainer runs is decided by
            ``app/explainability/capability.py`` from the model's observable
            structure, never from its class or ``model_type``. When omitted,
            behaviour is exactly as it was before adapters existed.

    Returns:
        dict: the approved interface shape, with raw feature names only:
            {
                "method": "shap" | "lime",
                "per_instance": [{"row_index": 0, "contributions": {...}}, ...],
                "global_importance": {...},
                "is_mock": False
            }

        With an ``adapter``, the same four keys plus the labelling needed to
        read the numbers honestly: ``model_id``, ``model_type``, ``available``,
        ``explainer``, ``scale``, ``fidelity``, ``feature_space``,
        ``capabilities`` and ``limitations``.

        SCALE IS NOT IMPLIED BY THE METHOD. Linear SHAP is log-odds; tree and
        kernel SHAP are probability; LIME is probability. Read ``scale`` from
        the result rather than inferring it from ``method``.

        When a model cannot honestly be explained, the result is a structured
        unavailable -- ``available=False``, empty ``per_instance`` and
        ``global_importance``, and ``limitations`` saying why. Never an
        exception for the caller to interpret, and never a zero-filled table
        that would read as "no feature mattered".

    Raises:
        ValueError: if ``method`` is unsupported, if ``model_output`` supplies a
            feature schema the real model does not accept, or if the model
            pipeline's structure is not the one this module depends on.
        FileNotFoundError: if no trained model artifact exists. Train one with
            ``python -m app.models.train``.

    Note:
        SHAP returns log-odds contributions; LIME returns predicted-probability
        contributions. Their magnitudes are not comparable, and their feature
        rankings can also diverge (measured rank correlation ~0.5 on the
        current model). Read them as two independent views; do not treat
        agreement between them as corroboration. See the module docstring.

        The two methods also differ on unseen categorical values: SHAP
        tolerates them (the model uses handle_unknown='ignore'), while LIME
        raises. See ``_explain_lime``.
    """
    method_lower = str(method).lower()
    if method_lower not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported explainability method: '{method}'. "
            "Supported methods are 'shap' and 'lime'."
        )

    if adapter is not None:
        return _explain_with_adapter(model_output, method_lower, adapter)

    # DEFAULT PATH -- unchanged from Phase 1, deliberately. The German Credit
    # pipeline is loaded, the German Credit schema gates the input, and the
    # numbers are byte-for-byte what they always were. Four other modules
    # already consume this path, so it is a compatibility surface, not just a
    # default.
    context = _DEFAULT_CONTEXT

    X = _resolve_feature_frame(model_output, context=context)
    pipeline = _load_fitted_model(context)

    if method_lower == "shap":
        return _explain_shap(pipeline, X, context=context)
    return _explain_lime(pipeline, X, context=context)
