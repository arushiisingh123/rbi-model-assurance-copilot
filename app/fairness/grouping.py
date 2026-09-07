"""Identity grouping and structural category extraction for fairness evaluation.

Note on sensitive feature representation:
The categories processed here are raw combined personal-status-and-sex categories
(e.g., Attribute 9 in UCI German Credit). They represent combined marital/personal
status and sex, NOT a clean, standalone sex or gender field. Binary sex derivation
is explicitly deferred. No meaning table or codebook mapping is assumed or applied.

Note on pairing:
``predictions`` and ``sensitive_feature`` describe the same records in the same
order, and are paired strictly by POSITION -- element i of one refers to the same
record as element i of the other. A pandas index is never used to pair them; see
``_as_positional_series`` for why, and for what happens when an input carries a
non-positional index.
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


def _as_positional_series(values: Any, argument_name: str) -> pd.Series:
    """Return ``values`` as a Series whose index is exactly ``0..n-1``.

    Pairing here is positional, but pandas aligns binary operations on the
    *index*, not on position. A pandas input carrying any other index would
    therefore be re-paired by index behind the caller's back, and where two
    indexes only partially overlap the non-matching rows are silently dropped --
    yielding a valid-looking fairness result computed from mismatched pairs on a
    subset of the data.

    Rather than guess which pairing was meant, any pandas input whose index is
    not already ``0..n-1`` is rejected. The caller then states the intent
    explicitly: ``.reset_index(drop=True)`` keeps the current order, and
    ``.to_numpy()`` drops the index entirely.
    """
    if isinstance(values, pd.Series):
        if not values.index.equals(pd.RangeIndex(len(values))):
            raise ValueError(
                f"{argument_name} has a non-positional index, so predictions and "
                "sensitive_feature cannot be paired safely. They are paired by "
                "position (element i to element i), but pandas would align them by "
                "index and could silently drop or mis-pair records. Pass "
                f"{argument_name}.reset_index(drop=True) if the current order is "
                f"correct, or {argument_name}.to_numpy() to drop the index."
            )
        # Normalise e.g. Index([0, 1, ...]) to a true RangeIndex so both operands
        # share an identical index for the null mask below.
        return values.reset_index(drop=True)
    return pd.Series(values)


def validate_fairness_inputs(
    predictions: Any,
    sensitive_feature: Any,
) -> Tuple[pd.Series, pd.Series]:
    """Validate and positionally pair predictions and sensitive_feature.

    The two inputs must describe the same records in the same order. Pairing is
    strictly positional; a pandas index is never used to align them.

    Raises:
        ValueError: if either input is None; if either is a pandas Series whose
            index is not ``0..n-1``; if the lengths differ; if the inputs are
            empty; or if no rows survive null removal.

    Returns:
        Tuple of (clean_predictions, clean_sensitive_feature), positionally
        paired, with a row dropped only when one of its own two values is null.
    """
    if predictions is None or sensitive_feature is None:
        raise ValueError("predictions and sensitive_feature must not be None")

    pred_series = _as_positional_series(predictions, "predictions")
    sens_series = _as_positional_series(sensitive_feature, "sensitive_feature")

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
