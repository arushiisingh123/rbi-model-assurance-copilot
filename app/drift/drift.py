"""Drift evaluation module (owner: Arushi).

Calculates Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS)
statistic across numeric features using pure pandas and numpy (no scipy).
Thresholds are imported from app.config.thresholds.

Aggregation:
  psi = MAX across features
  ks_statistic = MAX across features
  These maxima are evaluated INDEPENDENTLY and may originate from different features.
  The drift report status is driven strictly by classify_psi(psi); KS statistic
  never affects the status.
"""

from typing import Any, List, Optional
import numpy as np
import pandas as pd

from app.config.thresholds import (
    STATUS_PENDING,
    classify_psi,
)

EPSILON = 1e-6


def _compute_feature_psi(ref_vals: np.ndarray, cur_vals: np.ndarray) -> float:
    """Compute PSI for a single numeric feature using reference quantiles.

    10 quantile bin edges are derived from reference only. Degenerate edges are
    collapsed via np.unique, outer edges are set to -inf and +inf, and zero bins
    are substituted with epsilon (1e-6).
    """
    if len(ref_vals) == 0 or len(cur_vals) == 0:
        return 0.0

    # If all reference values are constant, PSI is 0.0
    if np.all(ref_vals == ref_vals[0]):
        return 0.0

    quantiles = np.linspace(0, 1, 11)
    bin_edges = np.quantile(ref_vals, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Expand outer boundaries to capture out-of-range current values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_vals, bins=bin_edges)

    ref_pct = ref_counts / len(ref_vals)
    cur_pct = cur_counts / len(cur_vals)

    # Substitute epsilon for zero-frequency bins to prevent division by zero / log(0)
    ref_pct = np.where(ref_pct == 0, EPSILON, ref_pct)
    cur_pct = np.where(cur_pct == 0, EPSILON, cur_pct)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, psi_val))


def _compute_feature_ks(ref_vals: np.ndarray, cur_vals: np.ndarray) -> float:
    """Compute exact two-sample Kolmogorov-Smirnov test statistic using np.searchsorted."""
    if len(ref_vals) == 0 or len(cur_vals) == 0:
        return 0.0

    ref_sorted = np.sort(ref_vals)
    cur_sorted = np.sort(cur_vals)

    pooled = np.sort(np.concatenate([ref_sorted, cur_sorted]))
    ecdf_ref = np.searchsorted(ref_sorted, pooled, side="right") / len(ref_sorted)
    ecdf_cur = np.searchsorted(cur_sorted, pooled, side="right") / len(cur_sorted)

    ks_stat = float(np.max(np.abs(ecdf_ref - ecdf_cur)))
    return ks_stat


def drift_report(
    reference_data: Any = None,
    current_data: Any = None,
) -> dict:
    """Evaluate population stability index (PSI) and Kolmogorov-Smirnov (KS) drift.

    Args:
        reference_data: Baseline reference pandas DataFrame.
        current_data: Monitored current pandas DataFrame.

    Returns:
        Dictionary with keys:
            features_evaluated (list of str): Common numeric column names evaluated.
            psi (float): Maximum PSI across all evaluated features.
            ks_statistic (float): Maximum KS statistic across all evaluated features.
            status (str): PASS, WARNING, FAIL, or PENDING.
            is_mock (bool): False for real calculation.

    Raises:
        ValueError: If either reference_data or current_data is None or not a DataFrame.
    """
    if reference_data is None or current_data is None:
        raise ValueError("reference_data and current_data must not be None")

    if not isinstance(reference_data, pd.DataFrame) or not isinstance(current_data, pd.DataFrame):
        raise ValueError("reference_data and current_data must be pandas DataFrames")

    if reference_data.empty or current_data.empty:
        return {
            "features_evaluated": [],
            "psi": 0.0,
            "ks_statistic": 0.0,
            "status": STATUS_PENDING,
            "is_mock": False,
        }

    # Find common numeric columns preserving reference column ordering
    common_cols = [
        col
        for col in reference_data.columns
        if col in current_data.columns
        and pd.api.types.is_numeric_dtype(reference_data[col])
        and pd.api.types.is_numeric_dtype(current_data[col])
    ]

    if not common_cols:
        return {
            "features_evaluated": [],
            "psi": 0.0,
            "ks_statistic": 0.0,
            "status": STATUS_PENDING,
            "is_mock": False,
        }

    feature_psis = []
    feature_kss = []

    for col in common_cols:
        ref_series = reference_data[col].dropna().to_numpy(dtype=float)
        cur_series = current_data[col].dropna().to_numpy(dtype=float)

        feat_psi = _compute_feature_psi(ref_series, cur_series)
        feat_ks = _compute_feature_ks(ref_series, cur_series)

        feature_psis.append(feat_psi)
        feature_kss.append(feat_ks)

    # Aggregate MAX independently across features
    max_psi = float(max(feature_psis)) if feature_psis else 0.0
    max_ks = float(max(feature_kss)) if feature_kss else 0.0

    # Status is driven strictly by PSI
    status = classify_psi(max_psi)

    return {
        "features_evaluated": common_cols,
        "psi": round(max_psi, 4),
        "ks_statistic": round(max_ks, 4),
        "status": status,
        "is_mock": False,
    }
