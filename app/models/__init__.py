"""Model & Data module for AI Model Risk & Assurance Copilot (owner: Namitha).

Exposes model training, evaluation, persistence, and batch prediction, plus
the authoritative RAW feature schema and target label semantics so downstream
modules can read the model contract without guessing.
"""

from app.models.model import (
    LABEL_SEMANTICS,
    MODEL_TYPE,
    MODEL_VERSION,
    evaluate,
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

__all__ = [
    "train",
    "predict_batch",
    "evaluate",
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
    "MODEL_TYPE",
    "MODEL_VERSION",
]
