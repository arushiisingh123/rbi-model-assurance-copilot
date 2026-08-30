"""Fairness evaluation module (owner: Arushi).

Calculates demographic parity difference and disparate impact ratio across
categories using pure pandas and numpy. Thresholds are imported from
app.config.thresholds.

Note on sensitive feature representation:
The sensitive feature reflects raw combined categories (e.g. Attribute 9 of
German Credit data). It is not a standalone gender/sex column.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from app.config.thresholds import (
    STATUS_PENDING,
    classify_disparate_impact,
)
from app.fairness.grouping import validate_fairness_inputs

DEFAULT_PROTECTED_ATTRIBUTE = "personal_status_and_sex"


def fairness_report(
    predictions: Any = None,
    sensitive_feature: Any = None,
    favorable_label: int = 1,
) -> dict:
    """Calculate demographic parity difference and disparate impact ratio.

    Args:
        predictions: Model predictions (binary or discrete labels).
        sensitive_feature: Protected / sensitive attribute categories.
        favorable_label: Value of prediction considered favorable (default 1).

    Returns:
        Dictionary with keys:
            protected_attribute (str): Name or default identifier of the attribute.
            demographic_parity_diff (float): max selection rate - min selection rate.
            disparate_impact_ratio (float): min selection rate / max selection rate.
            status (str): PASS, WARNING, FAIL, or PENDING.
            is_mock (bool): False for real calculation.
    """
    # Determine protected attribute name before input cleaning
    if isinstance(sensitive_feature, pd.Series) and sensitive_feature.name:
        attr_name = str(sensitive_feature.name)
        # Guard against hardcoded "gender" name
        if attr_name.lower() == "gender":
            attr_name = DEFAULT_PROTECTED_ATTRIBUTE
    else:
        attr_name = DEFAULT_PROTECTED_ATTRIBUTE

    clean_preds, clean_sens = validate_fairness_inputs(predictions, sensitive_feature)

    groups = pd.unique(clean_sens)

    # Fewer than 2 distinct groups -> PENDING, neutral metrics
    if len(groups) < 2:
        return {
            "protected_attribute": attr_name,
            "demographic_parity_diff": 0.0,
            "disparate_impact_ratio": 1.0,
            "status": STATUS_PENDING,
            "is_mock": False,
        }

    # Calculate selection rate per group: P(pred == favorable_label | Group == g)
    selection_rates = []
    for g in groups:
        group_mask = clean_sens == g
        group_preds = clean_preds[group_mask]
        rate = float(np.mean(group_preds == favorable_label))
        selection_rates.append(rate)

    max_rate = max(selection_rates)
    min_rate = min(selection_rates)

    # Edge case: max selection rate == 0 -> parity at zero
    if max_rate == 0.0:
        dp_diff = 0.0
        di_ratio = 1.0
    else:
        dp_diff = float(max_rate - min_rate)
        di_ratio = float(min_rate / max_rate)

    status = classify_disparate_impact(di_ratio)

    return {
        "protected_attribute": attr_name,
        "demographic_parity_diff": round(dp_diff, 4),
        "disparate_impact_ratio": round(di_ratio, 4),
        "status": status,
        "is_mock": False,
    }
