"""Drift evaluation module (owner: Arushi).

Calculates Population Stability Index (PSI) and the two-sample
Kolmogorov-Smirnov (KS) statistic across numeric features using pure pandas and
numpy (no scipy). Thresholds are imported from app.config.thresholds -- this
module never defines its own.

WHAT IS AND IS NOT COVERED
    Only columns that are numeric (and non-boolean) in BOTH frames are
    evaluated. Everything else is skipped and is NOT covered by the reported
    PSI/KS values:

      - categorical / text columns are excluded (they are not treated as numeric);
      - boolean columns are excluded (semantically categorical, not continuous);
      - columns present in only one of the two frames are excluded;
      - non-finite values (NaN, +inf, -inf) are excluded per feature, because
        they cannot be placed in a quantile bin and would otherwise corrupt PSI;
      - a feature left with no usable values after that cleaning is excluded
        entirely rather than silently contributing a zero.

    ``features_evaluated`` lists exactly the features that were actually
    evaluated, so it never claims coverage the calculation did not provide.

AGGREGATION
    psi          = MAX across evaluated features
    ks_statistic = MAX across evaluated features
    The two maxima are evaluated INDEPENDENTLY and may originate from different
    features -- on the real German Credit train/test split the maximum PSI comes
    from ``age`` while the maximum KS comes from ``installment_rate``.

    ``per_feature`` (Phase 4, additive) exposes the per-feature PSI and KS
    values behind those maxima, aligned index-for-index with
    ``features_evaluated`` and rounded to the same precision. It is
    informational only: there is no per-feature status and no per-feature
    threshold, and ``status`` continues to come from the aggregate PSI alone.
    Because rounding is monotonic, the maximum per-feature value always equals
    the reported aggregate.

STATUS
    Status is driven strictly by classify_psi(psi). The KS statistic is an
    analytical output only and never contributes a severity of its own -- no KS
    threshold exists (docs/thresholds.md).

Note on is_mock:
    ``is_mock=False`` means the PSI/KS arithmetic is real. It says nothing about
    where the data came from. Drift computed against a dataset produced by
    app.drift.scenario.build_drift_scenario is SYNTHETIC and demonstrates that
    detection works -- it must never be presented as observed drift in a real
    lending population.
"""

from typing import Any, List

import numpy as np
import pandas as pd

from app.config.thresholds import (
    STATUS_PENDING,
    classify_psi,
)

EPSILON = 1e-6

# Number of decimal places the reported metrics are rounded to. The status is
# derived from the rounded (reported) PSI so the number shown in a report and
# the status attached to it can never disagree -- this matters most at the
# 0.10 boundary, where an unrounded 0.09999 would print as "0.1" (documented
# WARNING) while being classified PASS.
_ROUNDING_DP = 4


def _is_evaluable_numeric(series: pd.Series) -> bool:
    """True for continuous numeric columns only.

    Booleans are excluded: pandas reports them as numeric, but a boolean flag is
    semantically categorical and quantile-binning it is meaningless.
    """
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _finite_values(series: pd.Series) -> np.ndarray:
    """Return the finite float values of ``series``.

    Drops NaN and +/-inf. Infinities must be removed before quantile binning:
    np.quantile over infinities produces NaN bin edges and a meaningless PSI,
    which would otherwise surface as a spurious drift FAIL.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _compute_feature_psi(ref_vals: np.ndarray, cur_vals: np.ndarray) -> float:
    """Compute PSI for a single numeric feature using reference quantiles.

    10 quantile bin edges are derived from the reference only. Degenerate
    (duplicate) edges are collapsed via np.unique, the outer edges are widened
    to -inf/+inf so out-of-range current values are still captured, and
    zero-frequency bins are substituted with epsilon (1e-6) to avoid log(0).

    Inputs are expected to be finite (see _finite_values).
    """
    if len(ref_vals) == 0 or len(cur_vals) == 0:
        return 0.0

    # A constant reference distribution has no spread to bin against.
    if np.all(ref_vals == ref_vals[0]):
        return 0.0

    quantiles = np.linspace(0, 1, 11)
    bin_edges = np.quantile(ref_vals, quantiles)
    # Heavily tied data produces duplicate edges; collapse them.
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_vals, bins=bin_edges)

    ref_pct = ref_counts / len(ref_vals)
    cur_pct = cur_counts / len(cur_vals)

    ref_pct = np.where(ref_pct == 0, EPSILON, ref_pct)
    cur_pct = np.where(cur_pct == 0, EPSILON, cur_pct)

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, psi_val))


def _compute_feature_ks(ref_vals: np.ndarray, cur_vals: np.ndarray) -> float:
    """Compute the exact two-sample Kolmogorov-Smirnov statistic.

    max |ECDF_ref - ECDF_cur| evaluated on the pooled sample, via
    np.searchsorted. No scipy dependency.
    """
    if len(ref_vals) == 0 or len(cur_vals) == 0:
        return 0.0

    ref_sorted = np.sort(ref_vals)
    cur_sorted = np.sort(cur_vals)

    pooled = np.sort(np.concatenate([ref_sorted, cur_sorted]))
    ecdf_ref = np.searchsorted(ref_sorted, pooled, side="right") / len(ref_sorted)
    ecdf_cur = np.searchsorted(cur_sorted, pooled, side="right") / len(cur_sorted)

    return float(np.max(np.abs(ecdf_ref - ecdf_cur)))


def _pending_result() -> dict:
    """Neutral result used when no feature could be evaluated.

    Absent or unusable data is not evidence of drift, so the status is PENDING
    rather than FAIL, and the metrics are neutral zeros.
    """
    return {
        "features_evaluated": [],
        "psi": 0.0,
        "ks_statistic": 0.0,
        "status": STATUS_PENDING,
        "is_mock": False,
        # Nothing was evaluated, so there is no per-feature detail to report.
        # An empty list says that; a zero-valued entry would invent a measurement.
        "per_feature": [],
    }


def drift_report(
    reference_data: Any = None,
    current_data: Any = None,
) -> dict:
    """Evaluate PSI and KS drift between a reference and a current dataset.

    Args:
        reference_data: Baseline reference pandas DataFrame.
        current_data: Monitored current pandas DataFrame.

    Returns:
        Dictionary with keys:
            features_evaluated (list of str): exactly the features actually
                evaluated (see module docstring for what is excluded).
            psi (float): maximum PSI across evaluated features.
            ks_statistic (float): maximum KS statistic across evaluated features.
            status (str): PASS, WARNING, FAIL, or PENDING, derived solely from
                the reported PSI via app.config.thresholds.classify_psi.
            is_mock (bool): False -- the calculation is real.
            per_feature (list of dict): one entry per evaluated feature, in the
                same order as features_evaluated, each with ``feature``,
                ``psi``, and ``ks_statistic`` at the same rounding precision as
                the aggregates. Informational detail only -- it carries no
                status and no threshold of its own. Empty when nothing could be
                evaluated.

    Raises:
        ValueError: If either input is None or is not a pandas DataFrame.

    Notes:
        PENDING is returned when no feature could be evaluated -- empty frames,
        no shared numeric columns, or every shared numeric column left with no
        usable (finite) values. Missing data is never reported as FAIL.
    """
    if reference_data is None or current_data is None:
        raise ValueError("reference_data and current_data must not be None")

    if not isinstance(reference_data, pd.DataFrame) or not isinstance(
        current_data, pd.DataFrame
    ):
        raise ValueError("reference_data and current_data must be pandas DataFrames")

    if reference_data.empty or current_data.empty:
        return _pending_result()

    # Shared continuous numeric columns, preserving reference column ordering.
    candidate_cols = [
        col
        for col in reference_data.columns
        if col in current_data.columns
        and _is_evaluable_numeric(reference_data[col])
        and _is_evaluable_numeric(current_data[col])
    ]

    if not candidate_cols:
        return _pending_result()

    evaluated_cols: List[str] = []
    feature_psis: List[float] = []
    feature_kss: List[float] = []

    for col in candidate_cols:
        ref_vals = _finite_values(reference_data[col])
        cur_vals = _finite_values(current_data[col])

        # Nothing usable left on either side -> the feature was not evaluated,
        # so it must not be listed as evaluated either.
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        evaluated_cols.append(col)
        feature_psis.append(_compute_feature_psi(ref_vals, cur_vals))
        feature_kss.append(_compute_feature_ks(ref_vals, cur_vals))

    if not evaluated_cols:
        return _pending_result()

    # MAX aggregation, evaluated independently per metric.
    max_psi = round(float(max(feature_psis)), _ROUNDING_DP)
    max_ks = round(float(max(feature_kss)), _ROUNDING_DP)

    # Classify the reported value so the PSI shown and the status agree.
    status = classify_psi(max_psi)

    # Per-feature detail (Phase 4, additive): the same values the maxima were
    # taken from, at the same precision, in the same order as
    # features_evaluated. Rounding is monotonic, so max(per-feature psi) equals
    # the reported psi and max(per-feature ks_statistic) equals the reported
    # ks_statistic -- the detail can never disagree with the aggregate it
    # explains. No status and no threshold is attached to a per-feature value.
    per_feature = [
        {
            "feature": col,
            "psi": round(float(psi_val), _ROUNDING_DP),
            "ks_statistic": round(float(ks_val), _ROUNDING_DP),
        }
        for col, psi_val, ks_val in zip(evaluated_cols, feature_psis, feature_kss)
    ]

    return {
        "features_evaluated": evaluated_cols,
        "psi": max_psi,
        "ks_statistic": max_ks,
        "status": status,
        "is_mock": False,
        "per_feature": per_feature,
    }
