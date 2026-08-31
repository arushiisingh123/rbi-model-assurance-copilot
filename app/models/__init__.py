"""Model & Data module for AI Model Risk & Assurance Copilot (owner: Namitha).

Exposes model training, evaluation, persistence, and batch prediction.
"""

from app.models.model import (
    evaluate,
    load,
    predict_batch,
    save,
    train,
)
from app.models.preprocessing import (
    load_dataset,
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
]
