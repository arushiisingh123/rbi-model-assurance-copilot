"""Synthetic bank reference model (Owner: Manas, covering Namitha's Model/Data role).

A second, independently-trained credit-scoring model with its own schema,
served over its own HTTP API (app.synthetic_bank.service), used to prove the
ModelAdapter / ModelRegistry / RESTAdapter architecture against a model this
codebase never trained in-process. Kept separate from app/models/ (the
German Credit reference implementation) rather than modifying it.
"""
from app.synthetic_bank.data_generator import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    generate_customers,
)
from app.synthetic_bank.model import (
    MODEL_ID,
    MODEL_TYPE,
    MODEL_VERSION,
    load,
    save,
    train,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "generate_customers",
    "MODEL_ID",
    "MODEL_VERSION",
    "MODEL_TYPE",
    "train",
    "save",
    "load",
]
