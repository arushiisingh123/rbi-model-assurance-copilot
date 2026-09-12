"""Credit-scoring model training, evaluation, persistence, and prediction (owner: Namitha).

Implements Phase 1 scikit-learn Logistic Regression pipeline for RBI compliance
model risk assurance, operating on the UCI Statlog (German Credit Data) dataset.

Public model contract (Phase 1)
-------------------------------
- ``load(path=...)`` -> fitted ``sklearn.pipeline.Pipeline``. Raises
  ``FileNotFoundError`` if the artifact is absent (deterministic, no
  implicit training).
- ``predict_batch(feature_matrix: pandas.DataFrame | None)`` -> shared dict
  (see the function docstring). Internally always a ``pandas.DataFrame``;
  serialization to ``list[dict]`` is the API layer's concern, not this
  module's.
- ``predict_batch()`` output carries ``instance_ids`` (Phase 3, additive):
  a batch-aligned ``list[str]`` of stable per-record identifiers, position
  ``i`` describing the same record as ``predictions[i]`` /
  ``probabilities[i]`` / ``feature_matrix`` row ``i``. Identity metadata
  only -- never a model feature. See ``app.models.preprocessing`` for the
  deterministic scheme.
- Expected input: the 20 RAW German-Credit features named and ordered by
  ``app.models.preprocessing.FEATURE_COLUMNS``. The pipeline one-hot expands
  categoricals internally; those expanded columns are NOT part of the input
  contract and are never exposed as ``feature_names``.
- Labels: ``0`` = GOOD (low risk), ``1`` = BAD (high risk) = positive class.
  ``probabilities[i]`` is ``P(class == 1) == P(BAD)``. The favorable credit
  outcome is label ``0``; consumers must not assume the favorable label is
  ``1``. This is echoed in ``model_metadata["label_semantics"]``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.models.preprocessing import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATASET_PATH,
    FAVORABLE_OUTCOME_LABEL,
    FEATURE_COLUMNS,
    INSTANCE_ID_COLUMN,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    load_dataset,
    make_fallback_instance_ids,
    preprocess,
    split_data,
)

DEFAULT_MODEL_ARTIFACT_PATH = "app/models/artifacts/credit_model.joblib"
MODEL_VERSION = "0.1.0"
MODEL_TYPE = "logistic_regression"

# Project-agreed target label semantics, surfaced in model_metadata so every
# downstream consumer reads the same contract instead of assuming a polarity.
# 0 = GOOD (low risk), 1 = BAD (high risk / default) = positive class.
LABEL_SEMANTICS: Dict[str, Any] = {
    "0": "GOOD - low credit risk",
    "1": "BAD - high credit risk / likely default",
    "positive_class": POSITIVE_CLASS,
    "probabilities_represent": "P(class == 1) = P(BAD / high credit risk)",
    "favorable_outcome_label": FAVORABLE_OUTCOME_LABEL,
}


def _positive_class_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(class == 1) = P(BAD) regardless of ``model.classes_`` ordering.

    sklearn orders ``predict_proba`` columns by ``model.classes_``. For a
    ``{0, 1}`` target that is already ``[0, 1]`` so column ``1`` is the
    positive class, but this looks the column up explicitly so the contract
    ("probabilities mean P(BAD)") cannot silently break.
    """
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if POSITIVE_CLASS not in classes:
        raise ValueError(
            f"Model was not trained with the positive class {POSITIVE_CLASS}; "
            f"classes_={classes}."
        )
    return proba[:, classes.index(POSITIVE_CLASS)]


def build_pipeline(
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """Build an sklearn Pipeline combining preprocessing and LogisticRegression.

    Categorical features are one-hot encoded; numeric features are scaled
    with StandardScaler to ensure smooth and deterministic convergence.

    Parameters
    ----------
    categorical_cols : Optional[List[str]]
        Names of categorical columns to encode.
    numeric_cols : Optional[List[str]]
        Names of numeric columns to scale.
    random_state : int, default=42
        Random seed for classifier reproducibility.
    max_iter : int, default=1000
        Maximum iterations for logistic regression solver.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES
    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            (
                "num",
                StandardScaler(),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train(
    dataset_path: str = DEFAULT_DATASET_PATH,
    save_path: Optional[str] = DEFAULT_MODEL_ARTIFACT_PATH,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train the credit scoring model on the approved dataset.

    Performs data loading, preprocessing, stratified 80/20 train/test splitting,
    fits the pipeline strictly on the training set, evaluates on the held-out test
    set, and optionally persists the trained pipeline.

    Parameters
    ----------
    dataset_path : str
        Path to the German Credit dataset CSV.
    save_path : Optional[str]
        Path to save the trained model artifact. If None, model is not saved.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary with training status, fitted model pipeline, and evaluation metrics.
    """
    df = load_dataset(dataset_path)
    X, y, cat_cols, num_cols = preprocess(df)

    if y is None:
        raise ValueError("Cannot train model without target column 'credit_risk'.")

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=random_state
    )

    model = build_pipeline(
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        random_state=random_state,
    )

    # Train strictly on the training set
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    metrics = evaluate(model, X_test, y_test)

    saved_result = None
    if save_path is not None:
        saved_result = save(model, path=save_path)

    return {
        "status": "trained",
        "model": model,
        "metrics": metrics,
        "dataset_path": dataset_path,
        "save_result": saved_result,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }


def evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a trained model on a held-out test dataset.

    Parameters
    ----------
    model : Any
        Fitted sklearn Pipeline or estimator.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True binary labels (0 = good credit, 1 = bad credit / high risk).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing accuracy, precision, recall, f1, roc_auc,
        n_test_samples, and is_mock=False.
    """
    y_pred = model.predict(X_test)
    y_prob = _positive_class_probabilities(model, X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "n_test_samples": int(len(y_test)),
        "is_mock": False,
    }


def evaluate_current_model() -> Dict[str, Any]:
    """Held-out evaluation metrics for the current persisted default model.

    Additive Phase 4 (D1(a)) helper. Exposes the same metrics ``evaluate()``
    already computes -- accuracy, precision, recall, f1, roc_auc -- as a
    property of the *current trained model artifact*, not of any particular
    prediction batch. All metric calculation is delegated to ``evaluate()``;
    nothing is recalculated or reimplemented here.

    Deliberately separate from ``predict_batch()``: ``predict_batch()`` scores
    whatever ``feature_matrix`` a caller supplies, which may carry no
    ground-truth labels at all, so it has no ``y_test`` to evaluate against
    and does not gain a metrics field. This function instead regenerates the
    same deterministic held-out split ``predict_batch()``'s own default demo
    path uses (``split_data(..., test_size=0.2, random_state=42)`` on the
    approved dataset), so the numbers it returns describe the model's
    held-out performance -- never a prediction batch's.

    Returns
    -------
    Dict[str, Any]
        Exactly ``evaluate()``'s return shape: accuracy, precision, recall,
        f1, roc_auc, n_test_samples, is_mock.
    """
    model = _get_or_train_default_model()
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, y_test = split_data(X, y, test_size=0.2, random_state=42)
    return evaluate(model, X_test, y_test)


def save(
    model: Any,
    path: str = DEFAULT_MODEL_ARTIFACT_PATH,
) -> Dict[str, str]:
    """Persist a trained model pipeline to disk using joblib.

    Parameters
    ----------
    model : Any
        Fitted sklearn Pipeline to serialize.
    path : str
        Target file path for the .joblib artifact.

    Returns
    -------
    Dict[str, str]
        Dictionary containing status and saved file path.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    joblib.dump(model, path)
    return {"status": "saved", "path": path}


def _artifact_feature_schema_error(model: Any) -> Optional[str]:
    """Return a human-readable reason a loaded artifact fails the current
    raw feature schema, or ``None`` if it matches.

    A joblib artifact deserializes cleanly even when it was trained on an
    older or different ``FEATURE_COLUMNS`` set; that mismatch would otherwise
    only surface as an opaque sklearn column error deep inside ``predict``.
    Downstream modules (explainability, fairness, drift, API) consume this
    model's output, so the check belongs here at the model boundary.

    Lenient by design: an estimator without ``feature_names_in_`` (it was not
    fitted on a named DataFrame) is not rejected -- only a definite name
    mismatch is.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    actual = list(names)
    if set(actual) != set(FEATURE_COLUMNS) or len(actual) != len(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS) - set(actual))
        unexpected = sorted(set(actual) - set(FEATURE_COLUMNS))
        return (
            f"expected {len(FEATURE_COLUMNS)} raw features "
            f"(app.models.preprocessing.FEATURE_COLUMNS), got {len(actual)} "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return None


def load(
    path: str = DEFAULT_MODEL_ARTIFACT_PATH,
) -> Pipeline:
    """Load a trained model pipeline from disk using joblib.

    Parameters
    ----------
    path : str
        Path to the saved .joblib artifact.

    Returns
    -------
    Pipeline
        Deserialized sklearn Pipeline.

    Raises
    ------
    FileNotFoundError
        If the model artifact does not exist on disk.
    ValueError
        If the artifact loads but was trained on a feature schema that no
        longer matches ``app.models.preprocessing.FEATURE_COLUMNS`` (e.g. a
        stale local artifact left over from an earlier schema). The message
        names the differing columns instead of letting a confusing sklearn
        error surface later during prediction.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. "
            "Please train and save the model first using train() or 'python -m app.models.train'."
        )
    model = joblib.load(path)

    reason = _artifact_feature_schema_error(model)
    if reason is not None:
        raise ValueError(
            f"Model artifact at '{path}' does not match the current feature "
            f"schema: {reason}. Retrain with 'python -m app.models.train'."
        )
    return model


def _get_or_train_default_model() -> Pipeline:
    """Return the persisted default model, training + saving it once if absent.

    Deterministic: training uses ``random_state=42`` on the fixed approved
    dataset, so a rebuilt artifact is identical to a previously saved one.
    Only ``predict_batch()`` uses this convenience path -- ``load()`` itself
    never trains implicitly and raises ``FileNotFoundError`` on a missing
    artifact. The generated artifact lives under the git-ignored
    ``app/models/artifacts/`` directory and is never committed.
    """
    if os.path.exists(DEFAULT_MODEL_ARTIFACT_PATH):
        try:
            return load(DEFAULT_MODEL_ARTIFACT_PATH)
        except Exception:
            # Corrupt, unreadable, or stale-schema artifact (load() now raises
            # ValueError for a feature-schema mismatch): fall back to a fresh
            # deterministic rebuild so predict_batch() self-heals rather than
            # handing downstream modules output from a stale model.
            pass

    # Train and save default model if artifact not present
    result = train(
        dataset_path=DEFAULT_DATASET_PATH,
        save_path=DEFAULT_MODEL_ARTIFACT_PATH,
    )
    return result["model"]


def predict_batch(
    feature_matrix: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run batch prediction on input feature matrix adhering to the frozen interface.

    Follows the shared module interface contract from docs/module-interfaces.md:
    {
        "predictions": [0, 1, 0, ...],       # 0 = GOOD, 1 = BAD (positive class)
        "probabilities": [0.12, 0.81, ...],  # probabilities[i] = P(class 1) = P(BAD)
        "instance_ids": ["gc-0007", ...],    # additive: stable per-record identity,
                                             #   aligned 1:1 with predictions / probabilities
                                             #   / feature_matrix rows. NOT a model feature.
        "feature_matrix": <pandas.DataFrame>,  # the 20 RAW features, FEATURE_COLUMNS order
        "model_metadata": {
            "model_type": "logistic_regression",
            "version": "0.1.0",
            "trained_on": "data/german_credit/german_credit.csv",
            "feature_names": [...],          # 20 RAW feature names (not one-hot columns)
            "label_semantics": {...}         # additive: 0/1 meaning, positive & favorable class
        },
        "is_mock": False
    }

    ``feature_matrix`` is returned as a ``pandas.DataFrame`` (the in-process
    representation). The API layer serializes it to ``list[dict]`` at the
    HTTP boundary; this function does not.

    Record identity (``instance_ids``)
    ----------------------------------
    The returned dict carries ``instance_ids``: a ``list[str]`` aligned 1:1
    with ``predictions`` / ``probabilities`` / ``feature_matrix`` rows.

    - ``feature_matrix is None`` (default demo path): identity is the
      deterministic ``instance_id`` attached by ``load_dataset()`` (``gc-NNNN``
      from the fixed CSV row order), carried through the stratified split by
      value so a shuffled test row keeps its original identifier.
    - Caller supplies ``feature_matrix`` **with** an ``instance_id`` column:
      those values are preserved verbatim (cast to ``str``), never fed to the
      model, and never replaced. Null values raise ``ValueError``.
    - Caller supplies ``feature_matrix`` **without** an ``instance_id`` column:
      deterministic positional placeholders (``row-NNNN``) derived from the
      supplied batch order. No random IDs are ever generated.

    ``instance_id`` is identity metadata, not a predictive feature: it is
    excluded from ``FEATURE_COLUMNS`` and never reaches the pipeline.

    Parameters
    ----------
    feature_matrix : Optional[pd.DataFrame]
        Input features DataFrame. If None, uses the held-out test split of the
        German Credit dataset for a runnable demonstration. May optionally
        carry an ``instance_id`` column (see "Record identity" above).

    Returns
    -------
    Dict[str, Any]
        Shared dictionary containing predictions, probabilities, batch-aligned
        instance_ids, raw feature matrix, metadata, and is_mock=False.
    """
    model = _get_or_train_default_model()

    if feature_matrix is None:
        # Provide held-out test split as sensible default
        df = load_dataset(DEFAULT_DATASET_PATH)
        X, y, _, _ = preprocess(df)
        id_series = df[INSTANCE_ID_COLUMN]
        _, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
        # Align identity to the shuffled test rows BEFORE reset_index: the
        # pre-reset index labels are only a join key back to the identifiers
        # captured at load; the identity itself is the instance_id value.
        instance_ids: List[str] = id_series.loc[X_test.index].astype(str).tolist()
        feature_matrix = X_test.reset_index(drop=True)
    else:
        if not isinstance(feature_matrix, pd.DataFrame):
            raise TypeError(
                f"Expected feature_matrix to be a pandas DataFrame, got {type(feature_matrix)}."
            )
        feature_matrix = feature_matrix.copy().reset_index(drop=True)
        if INSTANCE_ID_COLUMN in feature_matrix.columns:
            supplied_ids = feature_matrix[INSTANCE_ID_COLUMN]
            if supplied_ids.isnull().any():
                raise ValueError(
                    f"feature_matrix '{INSTANCE_ID_COLUMN}' column contains null "
                    "values; supply a valid identifier for every row or omit the "
                    "column entirely."
                )
            instance_ids = supplied_ids.astype(str).tolist()
            # Identity metadata must never be handed to the model.
            feature_matrix = feature_matrix.drop(columns=[INSTANCE_ID_COLUMN])
        else:
            instance_ids = make_fallback_instance_ids(len(feature_matrix))

    # Ensure required features exist in input matrix
    missing_cols = [c for c in FEATURE_COLUMNS if c not in feature_matrix.columns]
    if missing_cols:
        raise ValueError(
            f"Input feature_matrix is missing required columns: {missing_cols}"
        )

    # Reorder columns to guarantee exact match with training pipeline
    scored_features = feature_matrix[FEATURE_COLUMNS].copy()

    raw_preds = model.predict(scored_features)
    raw_probs = _positive_class_probabilities(model, scored_features)

    predictions: List[int] = [int(p) for p in raw_preds]
    probabilities: List[float] = [float(p) for p in raw_probs]

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "instance_ids": instance_ids,
        "feature_matrix": scored_features,
        "model_metadata": {
            "model_type": MODEL_TYPE,
            "version": MODEL_VERSION,
            "trained_on": DEFAULT_DATASET_PATH,
            "feature_names": list(scored_features.columns),
            "label_semantics": LABEL_SEMANTICS,
        },
        "is_mock": False,
    }
