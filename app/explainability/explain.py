"""Phase 1 Explainability module (owner: Manas).

SHAP and LIME explanations for the REAL credit-scoring model owned by
Namitha (`app/models/model.py`).

WHAT THIS MODULE EXPLAINS
-------------------------
The trained sklearn ``Pipeline`` loaded via ``app.models.model.load()``.
There is no dummy model anywhere in this module: every number returned
here is derived from that pipeline's own coefficients or its own
``predict_proba``. This module never trains a model -- if the artifact is
missing, ``load()`` raises with instructions to run
``python -m app.models.train``.

FEATURE SPACES (there are two, and they must not be confused)
-------------------------------------------------------------
- RAW: the 20 German Credit columns the pipeline accepts as input
  (``app.models.preprocessing.FEATURE_COLUMNS``). This is the only space
  that ever appears in this module's output.
- TRANSFORMED: the 61 columns the pipeline's ColumnTransformer produces
  (one-hot for 13 categorical + scaled for 7 numeric). This space is an
  internal detail and is deliberately never exposed to callers.

SHAP works in the transformed space (where the classifier is linear and
SHAP is exact) and is then aggregated back to the 20 raw features. LIME
works directly in the raw space. Both return raw feature names.

SCALE AND RANKING -- SHAP AND LIME ARE NOT INTERCHANGEABLE
-----------------------------------------------------------
- SHAP contributions are on the model's LOG-ODDS scale and are exactly
  additive: ``sum(contributions) + expected_value == decision_function(x)``.
- LIME contributions are on the PREDICTED-PROBABILITY scale and are local
  surrogate weights, not an exact decomposition.

Two separate warnings follow, and the second is the easier one to get wrong:

1. MAGNITUDES are not comparable. Never plot the two on a shared axis, and
   never average, subtract, or otherwise combine them.
2. RANKINGS can also diverge. Measured on the current model, the two methods'
   global-importance rank correlation is only about 0.5 -- they agree loosely,
   not closely. Treat SHAP and LIME as two independent views of the model.
   A feature ranking highly under BOTH is NOT thereby corroborated: the two
   methods answer different questions (exact additive attribution vs. local
   surrogate fit), so agreement is informative but never confirmatory, and
   disagreement is expected rather than a defect.
"""

from typing import Any, Dict, List, Optional, Tuple

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

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


def _validate_feature_groups(groups: Dict[str, List[int]], n_transformed: int) -> None:
    """Fail loudly if the raw->transformed mapping does not tile the space.

    A silent mismatch here would misattribute contributions to the wrong
    feature, which is exactly the class of bug this module must never ship.
    """
    covered = sorted(i for indices in groups.values() for i in indices)
    if covered != list(range(n_transformed)):
        raise ValueError(
            "Could not map transformed columns back to raw features: the "
            f"mapping covers {len(covered)} column(s) but the preprocessor "
            f"produced {n_transformed}. The model's preprocessing structure "
            "has changed -- see app/models/model.py build_pipeline()."
        )
    unknown = [f for f in groups if f not in RAW_FEATURES]
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


def _validate_raw_schema(received: List[str]) -> None:
    """Raise unless ``received`` is exactly the real model's raw feature set.

    The pipeline selects its columns by name. Explaining a different feature
    set would produce numbers that look real but describe nothing, so we
    refuse loudly instead (CLAUDE.md section 6: mock results must never be
    presented as real results).
    """
    if set(received) == set(RAW_FEATURES) and len(received) == len(RAW_FEATURES):
        return
    missing = sorted(set(RAW_FEATURES) - set(received))
    unexpected = sorted(set(received) - set(RAW_FEATURES))
    raise ValueError(
        "Incompatible feature schema for the real credit model.\n"
        f"  Expected {len(RAW_FEATURES)} raw features: {RAW_FEATURES}\n"
        f"  Received {len(received)} features: {list(received)}\n"
        f"  Missing from input: {missing}\n"
        f"  Unexpected in input: {unexpected}\n"
        "Explainability explains the real model from app.models.model.load(), "
        "which accepts the German Credit raw schema defined in "
        "app/models/preprocessing.py FEATURE_COLUMNS."
    )


def _default_feature_frame() -> pd.DataFrame:
    """Rows explained when the caller supplies none.

    Uses the same held-out test split ``predict_batch()`` defaults to, so a
    default explanation describes the same rows the model module reports on.
    """
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    return X_test.reset_index(drop=True).head(DEFAULT_EXPLAIN_ROWS)


def _training_frame() -> pd.DataFrame:
    """The training split: SHAP background and LIME's sampling distribution."""
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    X_train, _, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    return X_train.reset_index(drop=True)


def _resolve_feature_frame(model_output: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Decide which rows to explain, validating any caller-supplied matrix."""
    if model_output is not None and not isinstance(model_output, dict):
        raise ValueError(
            "model_output must be a dict shaped like app.models.model.predict_batch() "
            f"output, got {type(model_output).__name__}."
        )

    supplied = model_output.get("feature_matrix") if isinstance(model_output, dict) else None
    if supplied is None:
        return _default_feature_frame()

    frame = _coerce_feature_frame(supplied)

    # Prefer the declared feature_names so a mislabelled matrix is caught,
    # rather than trusting the DataFrame's own columns implicitly.
    declared = None
    metadata = model_output.get("model_metadata")
    if isinstance(metadata, dict):
        declared = metadata.get("feature_names")
    received = list(declared) if declared else list(frame.columns)

    _validate_raw_schema(received)

    if declared is not None and set(declared) != set(frame.columns):
        raise ValueError(
            "model_metadata['feature_names'] does not match feature_matrix columns.\n"
            f"  feature_names: {sorted(declared)}\n"
            f"  matrix columns: {sorted(frame.columns)}"
        )

    # Reorder to the pipeline's training order. The ColumnTransformer selects
    # by name, but ordering the frame explicitly keeps downstream positional
    # work (LIME indices, SHAP grouping) unambiguous.
    return frame[RAW_FEATURES].reset_index(drop=True)


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


def _explain_shap(pipeline: Pipeline, X: pd.DataFrame) -> Dict[str, Any]:
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

    groups = _build_raw_feature_groups(preprocessor)
    background = _as_dense(preprocessor.transform(_training_frame()))
    _validate_feature_groups(groups, background.shape[1])

    transformed = _as_dense(preprocessor.transform(X))

    # An explicit masker keeps the background deterministic instead of letting
    # SHAP silently subsample it to its default 100 rows.
    masker = shap.maskers.Independent(background, max_samples=background.shape[0])
    explainer = shap.LinearExplainer(classifier, masker)
    values = explainer(transformed).values

    ordered = [f for f in RAW_FEATURES if f in groups]
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


def _explain_lime(pipeline: Pipeline, X: pd.DataFrame) -> Dict[str, Any]:
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

    categorical = [f for f in RAW_FEATURES if f in categories]
    categorical_idx = [RAW_FEATURES.index(f) for f in categorical]
    numeric = [f for f in RAW_FEATURES if f not in categories]

    code_maps = {f: {value: i for i, value in enumerate(categories[f])} for f in categorical}

    def encode(frame: pd.DataFrame) -> np.ndarray:
        encoded = frame[RAW_FEATURES].copy()
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
        frame = pd.DataFrame(array, columns=RAW_FEATURES)
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

    training = encode(_training_frame())
    target = encode(X)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training,
        feature_names=RAW_FEATURES,
        categorical_features=categorical_idx,
        categorical_names={i: categories[RAW_FEATURES[i]] for i in categorical_idx},
        class_names=["good_credit", "bad_credit"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    contributions = np.zeros((target.shape[0], len(RAW_FEATURES)), dtype=float)
    for row in range(target.shape[0]):
        explanation = explainer.explain_instance(
            target[row],
            decode_and_predict,
            labels=(1,),
            num_features=len(RAW_FEATURES),
            num_samples=LIME_NUM_SAMPLES,
        )
        for index, weight in explanation.as_map().get(1, []):
            contributions[row, int(index)] = float(weight)

    return _package("lime", contributions, RAW_FEATURES)


def explain(model_output: Optional[Dict[str, Any]] = None, method: str = "shap") -> Dict[str, Any]:
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

    Returns:
        dict: the approved interface shape, with raw feature names only:
            {
                "method": "shap" | "lime",
                "per_instance": [{"row_index": 0, "contributions": {...}}, ...],
                "global_importance": {...},
                "is_mock": False
            }

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

    X = _resolve_feature_frame(model_output)
    pipeline = load_real_model()

    if method_lower == "shap":
        return _explain_shap(pipeline, X)
    return _explain_lime(pipeline, X)
