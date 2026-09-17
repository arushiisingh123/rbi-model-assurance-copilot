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

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

from app.fairness.fairness import DEFAULT_PROTECTED_ATTRIBUTE
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
MODEL_ID = "german-credit-logistic-regression"
MODEL_VERSION = "0.1.0"
MODEL_TYPE = "logistic_regression"

# Phase 5C -- second supported model (Random Forest). Distinct artifact path
# so training/loading RF can never overwrite the LR artifact above.
RF_MODEL_ARTIFACT_PATH = "app/models/artifacts/credit_model_random_forest.joblib"
RF_MODEL_ID = "german-credit-random-forest"
RF_MODEL_VERSION = "0.1.0"
RF_MODEL_TYPE = "random_forest"

class ProbabilityCapabilityUnavailable(Exception):
    """Raised when an operation requires probability output but the model does not support it."""
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
    if not hasattr(model, "predict_proba"):
        raise ProbabilityCapabilityUnavailable(
            "This model does not support probability predictions."
        )
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if POSITIVE_CLASS not in classes:
        raise ValueError(
            f"Model was not trained with the positive class {POSITIVE_CLASS}; "
            f"classes_={classes}."
        )
    return proba[:, classes.index(POSITIVE_CLASS)]


def _build_preprocessor(
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> ColumnTransformer:
    """Shared preprocessing step: one-hot encode categoricals, scale numerics.

    Used by every model-type pipeline builder (``build_pipeline()`` for
    Logistic Regression, ``build_rf_pipeline()`` for Random Forest, and any
    future model) so preprocessing behavior never diverges by model type.
    """
    return ColumnTransformer(
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

    preprocessor = _build_preprocessor(categorical_cols, numeric_cols)

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


def build_rf_pipeline(
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: Optional[int] = 10,
    min_samples_leaf: int = 2,
) -> Pipeline:
    """Build an sklearn Pipeline combining preprocessing and RandomForestClassifier.

    Phase 5C: the second supported model. Reuses the exact same preprocessing
    approach as ``build_pipeline()`` (one-hot encoding + scaling via
    ``_build_preprocessor()``) so the two models differ only in classifier.
    StandardScaler is a no-op for tree splits but is kept for consistency
    with the shared preprocessing step rather than diverging per model type.

    Hyperparameters are conservative, fixed defaults -- no tuning performed
    in Phase 5C.

    Parameters
    ----------
    categorical_cols : Optional[List[str]]
        Names of categorical columns to encode.
    numeric_cols : Optional[List[str]]
        Names of numeric columns to scale.
    random_state : int, default=42
        Random seed for classifier reproducibility.
    n_estimators : int, default=200
        Number of trees in the forest.
    max_depth : Optional[int], default=10
        Maximum tree depth (bounded to limit overfitting on 800 training rows).
    min_samples_leaf : int, default=2
        Minimum samples required at a leaf node.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES
    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURES

    preprocessor = _build_preprocessor(categorical_cols, numeric_cols)

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
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
        Dictionary containing accuracy, precision, recall, f1, roc_auc
        (Optional[float], None if model does not support probability
        predictions), n_test_samples, and is_mock=False.

    Notes
    -----
    accuracy/precision/recall/f1 never need probabilities; only roc_auc
    does. A model without predict_proba still receives real hard-label
    metrics, with roc_auc explicitly set to None rather than failing the
    entire evaluation.
    """
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        y_prob = _positive_class_probabilities(model, X_test)
        roc_auc: Optional[float] = float(roc_auc_score(y_test, y_prob))
    except ProbabilityCapabilityUnavailable:
        roc_auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "n_test_samples": int(len(y_test)),
        "is_mock": False,
    }



def evaluate_current_model(adapter: Optional[Any] = None) -> Dict[str, Any]:
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

    Parameters
    ----------
    adapter : Optional[Any]
        Phase 5D, additive. Omitted (the default), byte-identical to the
        pre-existing default Logistic Regression path. Supplied (a
        ``ModelAdapter`` instance, e.g. ``RandomForestAdapter``), the
        held-out metrics are computed for that adapter's own fitted model
        (``adapter.load_fitted_model()``) over the same deterministic
        German Credit split, using ``adapter.feature_names`` for column
        selection/order -- so metrics can never silently describe a
        different model than the one the adapter wraps. No new metric
        calculation is introduced; both paths delegate to the same
        ``evaluate()``.

        This function's held-out split is German Credit's own test split --
        it has no meaning for an adapter whose schema isn't German Credit's
        (e.g. a RESTAdapter for an external model). Raises ``ValueError``
        for such an adapter rather than a confusing ``KeyError``/
        ``NotImplementedError`` from deep inside column selection or
        ``load_fitted_model()``; callers that want to support arbitrary
        adapters must catch this and degrade (e.g. to ``None`` metrics),
        not assume every adapter can be evaluated this way.

    Returns
    -------
    Dict[str, Any]
        Exactly ``evaluate()``'s return shape: accuracy, precision, recall,
        f1, roc_auc, n_test_samples, is_mock.

    Raises
    ------
    ValueError
        If ``adapter`` is supplied and its feature schema does not match
        German Credit's (``FEATURE_COLUMNS``).
    """
    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    _, X_test, _, y_test = split_data(X, y, test_size=0.2, random_state=42)

    if adapter is None:
        model = _get_or_train_default_model()
        return evaluate(model, X_test, y_test)

    if set(adapter.feature_names) != set(FEATURE_COLUMNS):
        raise ValueError(
            f"evaluate_current_model() cannot compute held-out metrics for "
            f"adapter '{adapter.model_id}': its feature schema does not "
            f"match German Credit's (FEATURE_COLUMNS), so German Credit's "
            f"held-out test split has no meaning for it. This function only "
            f"supports adapters trained on the German Credit schema (e.g. "
            f"LogisticRegressionAdapter, RandomForestAdapter)."
        )

    model = adapter.load_fitted_model()
    return evaluate(model, X_test[list(adapter.feature_names)], y_test)


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


def _get_or_train_default_rf_model() -> Pipeline:
    """Return the persisted default Random Forest model, training + saving it
    once if absent.

    Phase 5C counterpart to ``_get_or_train_default_model()``. Deliberately
    does not call ``train()`` (which is hardcoded to
    ``build_pipeline()``/Logistic Regression and is kept unchanged per the
    approved Phase 5C plan); instead it repeats the same
    load -> preprocess -> split -> fit -> evaluate -> save sequence with
    ``build_rf_pipeline()``, on the same deterministic dataset and split
    (``random_state=42``, canonical 80/20 stratified split) so the RF
    artifact is reproducible exactly like the LR one. Saves to
    ``RF_MODEL_ARTIFACT_PATH`` -- the LR artifact is never touched.
    """
    if os.path.exists(RF_MODEL_ARTIFACT_PATH):
        try:
            return load(RF_MODEL_ARTIFACT_PATH)
        except Exception:
            # Corrupt, unreadable, or stale-schema artifact: fall back to a
            # fresh deterministic rebuild, mirroring the LR self-heal above.
            pass

    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, cat_cols, num_cols = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42
    )

    model = build_rf_pipeline(
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        random_state=42,
    )
    model.fit(X_train, y_train)
    save(model, path=RF_MODEL_ARTIFACT_PATH)
    return model


class ModelAdapter(ABC):
    """Model-facing boundary so downstream assurance code (explainability,
    orchestration) does not need separate per-model-type interfaces.

    A concrete adapter owns: model identity (``model_id``, ``model_version``,
    ``model_type``), the expected raw feature schema, prediction execution,
    probability capability, and internal -- never serialized -- access to the
    fitted model/artifact and explainability background/reference data.

    ``predict_proba`` contract: returns the 1-D positive-class probability
    vector ``P(class == 1) == P(BAD)``, one value per row -- NOT sklearn's
    raw 2-column ``predict_proba`` matrix. Every adapter (the current
    LogisticRegression one, and any future model type) must return
    probabilities in this same shape and polarity, so callers never need
    per-model-type branching.
    """

    model_id: str
    model_version: str
    model_type: str
    feature_names: List[str]
    integration_type: str = "in_process"
    # Which raw feature (in feature_names) is this model's fairness protected
    # attribute, if any. None means "not declared" -- orchestration must
    # treat fairness as not applicable/PENDING for this adapter rather than
    # guessing or falling back to another model's protected attribute.
    protected_attribute: Optional[str] = None

    @property
    def input_schema(self) -> Dict[str, Dict[str, str]]:
        """Expected feature types for the model's raw inputs."""
        return {
            col: {
                "type": "categorical" if col in CATEGORICAL_FEATURES else "numeric"
            }
            for col in self.feature_names
        }

    @property
    def capabilities(self) -> Dict[str, bool]:
        """Adapter capability flags."""
        return {
            "predict_proba": self.supports_probability,
            "batch": True,
            "explainability": True,
        }

    def health(self) -> Dict[str, Any]:
        """Liveness/readiness check. In-process adapters just confirm the fitted
        model is loaded, e.g. return {"status": "ok"}."""
        return {"status": "ok"}

    def metadata(self) -> Dict[str, Any]:
        """Return {model_id, model_version, model_type, integration_type,
        capabilities} — the same shape /models/{model_id} will serialize."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "integration_type": self.integration_type,
            "capabilities": self.capabilities,
        }

    @property
    @abstractmethod
    def supports_probability(self) -> bool:
        """Whether this adapter can provide probability predictions.

        Internal capability signal only -- never serialized into
        ``model_metadata`` or any shared/API schema.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions (0 = GOOD, 1 = BAD) for X."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(class == 1) == P(BAD) as a 1-D array, one value per row.

        Must raise ``ProbabilityCapabilityUnavailable`` when
        ``supports_probability`` is False.
        """

    @abstractmethod
    def load_fitted_model(self) -> Any:
        """Return the underlying fitted model/artifact (internal use only)."""

    @abstractmethod
    def background_data(self) -> Optional[pd.DataFrame]:
        """Return reference/background data for explainability (internal use only)."""


class LogisticRegressionAdapter(ModelAdapter):
    """Adapter wrapping the existing Logistic Regression pipeline (Phase 5A/5B).

    Wraps the same fitted ``sklearn.pipeline.Pipeline`` produced by
    ``train()``/``load()`` -- no new model, no changed preprocessing, no
    changed target polarity. Existing LR behavior and P(BAD) probability
    semantics are unchanged; this only adds a uniform access boundary
    around them.
    """

    model_type = MODEL_TYPE
    protected_attribute = DEFAULT_PROTECTED_ATTRIBUTE

    def __init__(self, fitted_model: Any, background: Optional[pd.DataFrame] = None):
        self.model_id = MODEL_ID
        self.model_version = MODEL_VERSION
        self.feature_names: List[str] = list(FEATURE_COLUMNS)
        self._fitted_model = fitted_model
        self._background = background

    @property
    def supports_probability(self) -> bool:
        return hasattr(self._fitted_model, "predict_proba")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._fitted_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _positive_class_probabilities(self._fitted_model, X)

    def load_fitted_model(self) -> Any:
        return self._fitted_model

    def background_data(self) -> Optional[pd.DataFrame]:
        return self._background

    @classmethod
    def load_default(cls) -> "LogisticRegressionAdapter":
        """Build an adapter around the current persisted default LR model.

        ``background_data()`` on the result is the canonical deterministic
        training split (``split_data(..., test_size=0.2, random_state=42)``
        on the approved dataset) -- the same split used everywhere else in
        this module, not a newly invented sample.
        """
        model = _get_or_train_default_model()
        df = load_dataset(DEFAULT_DATASET_PATH)
        X, y, _, _ = preprocess(df)
        X_train, _, _, _ = split_data(X, y, test_size=0.2, random_state=42)
        background = X_train.reset_index(drop=True)
        return cls(model, background=background)


class RandomForestAdapter(ModelAdapter):
    """Adapter wrapping the Random Forest pipeline (Phase 5C).

    Same shape as ``LogisticRegressionAdapter``: wraps a fitted
    ``sklearn.pipeline.Pipeline`` (preprocessing + ``RandomForestClassifier``)
    trained on the same dataset, same 20 raw features, same target polarity,
    and the same deterministic 80/20 split. ``predict_proba()`` returns the
    same 1-D ``P(class == 1) == P(BAD)`` vector as every other adapter, via
    the same shared ``_positive_class_probabilities()`` helper.
    """

    model_type = RF_MODEL_TYPE
    protected_attribute = DEFAULT_PROTECTED_ATTRIBUTE

    def __init__(self, fitted_model: Any, background: Optional[pd.DataFrame] = None):
        self.model_id = RF_MODEL_ID
        self.model_version = RF_MODEL_VERSION
        self.feature_names: List[str] = list(FEATURE_COLUMNS)
        self._fitted_model = fitted_model
        self._background = background

    @property
    def supports_probability(self) -> bool:
        return hasattr(self._fitted_model, "predict_proba")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._fitted_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _positive_class_probabilities(self._fitted_model, X)

    def load_fitted_model(self) -> Any:
        return self._fitted_model

    def background_data(self) -> Optional[pd.DataFrame]:
        return self._background

    @classmethod
    def load_default(cls) -> "RandomForestAdapter":
        """Build an adapter around the current persisted default RF model.

        ``background_data()`` on the result is the same canonical
        deterministic training split (``split_data(..., test_size=0.2,
        random_state=42)`` on the approved dataset) that
        ``LogisticRegressionAdapter.load_default()`` uses -- the same
        source, not a separately derived sample.
        """
        model = _get_or_train_default_rf_model()
        df = load_dataset(DEFAULT_DATASET_PATH)
        X, y, _, _ = preprocess(df)
        X_train, _, _, _ = split_data(X, y, test_size=0.2, random_state=42)
        background = X_train.reset_index(drop=True)
        return cls(model, background=background)


def predict_batch(
    feature_matrix: Optional[pd.DataFrame] = None,
    *,
    adapter: Optional[ModelAdapter] = None,
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
    adapter : Optional[ModelAdapter]
        Phase 5A/5B, additive. When omitted (the default), behavior is
        exactly the current Logistic Regression path: same predictions,
        probabilities, instance_ids, feature_matrix content/order, and
        ``model_metadata`` keys/values as before this parameter existed.
        When supplied, prediction execution and the raw feature schema are
        delegated to the adapter instead of the default LR model, but the
        returned dict's top-level keys and ``model_metadata`` key shape are
        identical either way -- the adapter's ``model_id`` is never added to
        ``model_metadata`` (it stays available as a property on the adapter
        itself for orchestration to read separately). If the adapter exposes
        a ``trained_on`` attribute, that value is used for
        ``model_metadata["trained_on"]`` instead of the German Credit default
        -- an adapter trained on other data (e.g. a REST-backed model) should
        not be reported as trained on German Credit. Adapters without a
        ``trained_on`` attribute are unaffected.

    Returns
    -------
    Dict[str, Any]
        Shared dictionary containing predictions, probabilities, batch-aligned
        instance_ids, raw feature matrix, metadata, and is_mock=False.
    """
    if adapter is None:
        model = _get_or_train_default_model()
        schema = FEATURE_COLUMNS
    else:
        schema = list(adapter.feature_names)

    if feature_matrix is None:
        if adapter is not None and set(schema) != set(FEATURE_COLUMNS):
            # The adapter's schema is not German Credit's (e.g. a RESTAdapter
            # for an external model with its own feature space) -- the
            # German Credit test split below cannot serve as demo input for
            # it. Fall back to the adapter's own background/reference data
            # instead. Adapters whose schema matches German Credit's (the
            # default LR path, and RandomForestAdapter) are unaffected and
            # keep using the test split exactly as before this branch existed.
            demo_data = adapter.background_data()
            if demo_data is None or len(demo_data) == 0:
                raise ValueError(
                    f"No feature_matrix supplied and adapter '{adapter.model_id}' "
                    "has no background data to use as default demo input."
                )
            instance_ids = make_fallback_instance_ids(len(demo_data))
            feature_matrix = demo_data.reset_index(drop=True)
        else:
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
    missing_cols = [c for c in schema if c not in feature_matrix.columns]
    if missing_cols:
        raise ValueError(
            f"Input feature_matrix is missing required columns: {missing_cols}"
        )

    # Reorder columns to guarantee exact match with training pipeline
    scored_features = feature_matrix[schema].copy()

    if adapter is None:
        raw_preds = model.predict(scored_features)
        raw_probs = _positive_class_probabilities(model, scored_features)
        meta_model_type = MODEL_TYPE
        meta_model_version = MODEL_VERSION
    else:
        raw_preds = adapter.predict(scored_features)
        raw_probs = adapter.predict_proba(scored_features)
        meta_model_type = adapter.model_type
        meta_model_version = adapter.model_version

    predictions: List[int] = [int(p) for p in raw_preds]
    probabilities: List[float] = [float(p) for p in raw_probs]

    # An adapter trained on data other than German Credit (e.g. a REST-backed
    # model) may expose its own `trained_on` provenance string; fall back to
    # the German Credit default otherwise. `trained_on` is intentionally NOT
    # part of the abstract ModelAdapter contract -- adapters that don't
    # define it (LR, RF, and any adapter written before this existed) are
    # unaffected and keep reporting DEFAULT_DATASET_PATH exactly as before.
    meta_trained_on = (
        getattr(adapter, "trained_on", DEFAULT_DATASET_PATH)
        if adapter is not None
        else DEFAULT_DATASET_PATH
    )

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "instance_ids": instance_ids,
        "feature_matrix": scored_features,
        "model_metadata": {
            "model_type": meta_model_type,
            "version": meta_model_version,
            "trained_on": meta_trained_on,
            "feature_names": list(scored_features.columns),
            "label_semantics": LABEL_SEMANTICS,
        },
        "is_mock": False,
    }
