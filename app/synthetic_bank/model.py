"""Synthetic bank credit model: training, persistence, evaluation (owner: Manas).

Deliberately NOT another sklearn Pipeline like app/models/model.py's LR/RF --
an XGBoost classifier, trained on the synthetic bank's own schema
(app.synthetic_bank.data_generator), so it genuinely stresses the
ModelAdapter/RESTAdapter contract against a different model implementation.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from app.synthetic_bank.data_generator import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    TARGET_COLUMN,
    generate_customers,
)

MODEL_ID = "synthetic-bank-credit-v1"
MODEL_VERSION = "1.0.0"
MODEL_TYPE = "xgboost"

DEFAULT_ARTIFACT_PATH = "app/synthetic_bank/artifacts/model.joblib"


def _build_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )
    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train(random_state: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """Generate synthetic data, fit an XGBoost pipeline, evaluate on a held-out split.

    Deterministic given ``random_state`` -- same seed produces the same
    generated data, the same stratified split, and the same fitted model.
    """
    df = generate_customers(n=1000, random_state=random_state)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = _build_pipeline(random_state=random_state)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    return model, metrics


def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Held-out evaluation metrics -- same metric set as app/models/model.py's
    evaluate(), reimplemented locally (this module does not import from or
    modify app/models/model.py)."""
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    y_prob = proba[:, classes.index(POSITIVE_CLASS)]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "n_test_samples": int(len(y_test)),
        "is_mock": False,
    }


def save(model: Any, path: str = DEFAULT_ARTIFACT_PATH) -> Dict[str, str]:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    joblib.dump(model, path)
    return {"status": "saved", "path": path}


def load(path: str = DEFAULT_ARTIFACT_PATH) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Synthetic bank model artifact not found at '{path}'. "
            "Train and save it first using train()/save()."
        )
    return joblib.load(path)


def _get_or_train_default_model() -> Any:
    """Load the persisted default model, training + saving it once if absent.

    Same lazy-singleton-on-disk pattern as
    app.models.model._get_or_train_default_model(): deterministic
    (random_state=42), so a rebuilt artifact is identical to a previously
    saved one.
    """
    if os.path.exists(DEFAULT_ARTIFACT_PATH):
        try:
            return load(DEFAULT_ARTIFACT_PATH)
        except Exception:
            pass

    model, _metrics = train(random_state=42)
    save(model, path=DEFAULT_ARTIFACT_PATH)
    return model


def predict_one(model: Any, features: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single flat feature dict, returning the /score response shape.

    Used by app.synthetic_bank.service's POST /score handler.
    """
    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
    prediction = int(model.predict(row)[0])
    proba = model.predict_proba(row)
    classes = list(model.classes_)
    probability = float(proba[0, classes.index(POSITIVE_CLASS)])
    return {
        "prediction": prediction,
        "probability": probability,
        "model_version": MODEL_VERSION,
    }
