"""Identity grouping and structural category extraction for fairness evaluation.

Note on sensitive feature representation:
The categories processed here are raw combined personal-status-and-sex categories
(e.g., Attribute 9 in UCI German Credit). They represent combined marital/personal
status and sex, NOT a clean, standalone sex or gender field. Binary sex derivation
is explicitly deferred. No meaning table or codebook mapping is assumed or applied.
"""

from typing import Any, List, Tuple
import numpy as np
import pandas as pd


def get_observed_categories(sensitive_feature: Any) -> List[Any]:
    """Extract the exact set of observed, non-null categories from sensitive_feature.

    Returns the unique categories present in the input without translating or
    remapping category codes.
    """
    if sensitive_feature is None:
        return []

    series = pd.Series(sensitive_feature).dropna()
    if series.empty:
        return []

    # Return unique categories preserving type/order
    unique_vals = pd.unique(series)
    return [v for v in unique_vals if pd.notna(v)]


def validate_fairness_inputs(
    predictions: Any,
    sensitive_feature: Any,
) -> Tuple[pd.Series, pd.Series]:
    """Validate and align predictions and sensitive_feature inputs.

    Raises:
        ValueError: if either input is None, lengths mismatch, or inputs are empty.

    Returns:
        Tuple of (clean_predictions, clean_sensitive_feature) with missing/NaN
        rows dropped and matching index.
    """
    if predictions is None or sensitive_feature is None:
        raise ValueError("predictions and sensitive_feature must not be None")

    pred_series = pd.Series(predictions)
    sens_series = pd.Series(sensitive_feature)

    if len(pred_series) != len(sens_series):
        raise ValueError(
            f"Length mismatch: predictions has length {len(pred_series)}, "
            f"sensitive_feature has length {len(sens_series)}"
        )

    if len(pred_series) == 0:
        raise ValueError("Input arrays must not be empty")

    # Drop rows where either is NaN / null
    valid_mask = pred_series.notna() & sens_series.notna()
    clean_pred = pred_series[valid_mask].reset_index(drop=True)
    clean_sens = sens_series[valid_mask].reset_index(drop=True)

    if len(clean_pred) == 0:
        raise ValueError("No valid rows remaining after dropping null values")

    return clean_pred, clean_sens
