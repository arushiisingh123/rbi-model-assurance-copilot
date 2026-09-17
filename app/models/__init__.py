"""Model & Data module for AI Model Risk & Assurance Copilot (owner: Namitha).

Exposes model training, evaluation, persistence, and batch prediction, plus
the authoritative RAW feature schema and target label semantics so downstream
modules can read the model contract without guessing.
"""

from app.models.model import (
    LABEL_SEMANTICS,
    MODEL_ID,
    MODEL_TYPE,
    MODEL_VERSION,
    RF_MODEL_ID,
    RF_MODEL_TYPE,
    RF_MODEL_VERSION,
    LogisticRegressionAdapter,
    ModelAdapter,
    RandomForestAdapter,
    evaluate,
    evaluate_current_model,
    load,
    predict_batch,
    save,
    train,
)
from app.models.preprocessing import (
    CATEGORICAL_FEATURES,
    FAVORABLE_OUTCOME_LABEL,
    FEATURE_COLUMNS,
    INSTANCE_ID_COLUMN,
    LABEL_BAD,
    LABEL_GOOD,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    TARGET_COLUMN,
    load_dataset,
    make_dataset_instance_ids,
    make_fallback_instance_ids,
    preprocess,
    split_data,
)
from app.models.registry import (
    ModelNotFoundError,
    ModelRegistry,
    get_default_registry,
    reset_default_registry,
)
from app.models.rest_adapter import RESTAdapter, RESTAdapterError

__all__ = [
    "train",
    "predict_batch",
    "evaluate",
    "evaluate_current_model",
    "save",
    "load",
    "load_dataset",
    "preprocess",
    "split_data",
    # RAW feature schema (authoritative, model-input contract)
    "FEATURE_COLUMNS",
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "TARGET_COLUMN",
    # Stable per-record identity (Phase 3) -- metadata, NOT a model feature
    "INSTANCE_ID_COLUMN",
    "make_dataset_instance_ids",
    "make_fallback_instance_ids",
    # Target label semantics (0 = GOOD, 1 = BAD = positive class)
    "LABEL_GOOD",
    "LABEL_BAD",
    "POSITIVE_CLASS",
    "FAVORABLE_OUTCOME_LABEL",
    "LABEL_SEMANTICS",
    # Model identity
    "MODEL_ID",
    "MODEL_TYPE",
    "MODEL_VERSION",
    # Model adapter boundary (Phase 5A/5B)
    "ModelAdapter",
    "LogisticRegressionAdapter",
    # Second model: Random Forest (Phase 5C)
    "RF_MODEL_ID",
    "RF_MODEL_TYPE",
    "RF_MODEL_VERSION",
    "RandomForestAdapter",
    # Model registry (Phase 5)
    "ModelNotFoundError",
    "ModelRegistry",
    "get_default_registry",
    "reset_default_registry",
    # REST adapter (Phase 5)
    "RESTAdapter",
    "RESTAdapterError",
]
