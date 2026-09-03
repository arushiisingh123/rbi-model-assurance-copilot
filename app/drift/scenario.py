"""Drift scenario generator for model assurance demonstration.

NOTE: The dataset outputs produced by this module are SYNTHETIC, constructed
under controlled deterministic shifts to demonstrate drift detection capabilities
(e.g., PSI and KS calculations). They do NOT represent observed drift in any
real-world lending population.
"""

from typing import List, Tuple, Union
import pandas as pd


def build_drift_scenario(
    reference: pd.DataFrame,
    *,
    shift_features: Union[str, List[str]],
    shift_amount: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic drifted current dataset from a reference dataset.

    Applies a deterministic mean-shift offset to the selected feature(s):
        current[feature] = reference[feature] + shift_amount * reference[feature].std()

    This transformation is purely deterministic and does not use random sampling,
    ensuring exact reproducibility. The input DataFrame is never mutated.

    Args:
        reference: Reference baseline DataFrame.
        shift_features: Column name (str) or list of column names to shift.
        shift_amount: Multiplier for feature standard deviation to offset.
        seed: Seed parameter preserved for deterministic interface compatibility.

    Returns:
        Tuple of (reference_copy, current_data) as DataFrames.

    Raises:
        ValueError: If reference is not a pandas DataFrame, or if shift_features
                    contain columns not found in reference.
    """
    if not isinstance(reference, pd.DataFrame):
        raise ValueError("reference must be a pandas DataFrame")

    if isinstance(shift_features, str):
        features_to_shift = [shift_features]
    elif isinstance(shift_features, (list, tuple)):
        features_to_shift = list(shift_features)
    else:
        raise ValueError("shift_features must be a string or list of strings")

    missing = [f for f in features_to_shift if f not in reference.columns]
    if missing:
        raise ValueError(f"Features not found in reference DataFrame: {missing}")

    # Create deep copies to prevent mutation of the input frame
    ref_copy = reference.copy(deep=True)
    cur_copy = reference.copy(deep=True)

    for feature in features_to_shift:
        cur_copy[feature] = cur_copy[feature] + shift_amount * ref_copy[feature].std()

    return ref_copy, cur_copy
