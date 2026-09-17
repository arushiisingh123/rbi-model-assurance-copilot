"""Synthetic bank customer population generator (owner: Manas).

Generates a deterministic synthetic credit-applicant population with a schema
distinct from the German Credit dataset (app/models/preprocessing.py), so the
synthetic bank genuinely stresses the model-adapter contract with a different
feature space rather than re-labeling the same columns.

Target label semantics (matches app/models/preprocessing.py's convention for
consistency across the platform):
    0 = GOOD (repaid), 1 = BAD (defaulted) -- the positive class.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES: List[str] = ["employment_type", "region", "loan_purpose"]

NUMERIC_FEATURES: List[str] = [
    "age",
    "annual_income",
    "employment_years",
    "existing_loans",
    "credit_utilization_ratio",
    "late_payments_12m",
    "loan_amount",
]

FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_COLUMN = "default_flag"
LABEL_GOOD = 0
LABEL_BAD = 1
POSITIVE_CLASS = LABEL_BAD

EMPLOYMENT_TYPES = ["salaried", "self_employed", "unemployed", "retired"]
EMPLOYMENT_TYPE_PROBS = [0.50, 0.25, 0.10, 0.15]
# Additive effect on the default logit -- salaried is the baseline (0.0).
EMPLOYMENT_TYPE_EFFECT = {
    "salaried": 0.0,
    "self_employed": 0.15,
    "unemployed": 0.90,
    "retired": 0.10,
}

REGIONS = ["north", "south", "east", "west"]
REGION_PROBS = [0.25, 0.25, 0.25, 0.25]

LOAN_PURPOSES = ["auto", "home", "education", "personal", "business"]
LOAN_PURPOSE_PROBS = [0.25, 0.25, 0.15, 0.20, 0.15]

# Calibrated so generate_customers(random_state=42) lands the positive
# (default) rate near the middle of the required [0.10, 0.40] band.
_LOGIT_INTERCEPT = -1.9


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - x.mean()) / std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_customers(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Deterministic synthetic credit-applicant population.

    Same ``random_state`` always produces a byte-identical DataFrame -- all
    randomness is drawn from a single seeded ``numpy.random.RandomState``, no
    wall-clock time or other unseeded source is used anywhere in this
    function.

    ``default_flag`` carries real signal, not independent noise: higher
    ``credit_utilization_ratio``, more ``late_payments_12m``, more
    ``existing_loans``, and being ``unemployed`` all increase default
    probability; higher ``annual_income`` and more ``employment_years``
    decrease it. ``region`` and ``loan_purpose`` are real features with no
    effect on the label (analogous to real-world irrelevant features).

    Returns
    -------
    pd.DataFrame
        Columns: FEATURE_COLUMNS + [TARGET_COLUMN], ``n`` rows.
    """
    rng = np.random.RandomState(random_state)

    employment_type = rng.choice(EMPLOYMENT_TYPES, size=n, p=EMPLOYMENT_TYPE_PROBS)
    region = rng.choice(REGIONS, size=n, p=REGION_PROBS)
    loan_purpose = rng.choice(LOAN_PURPOSES, size=n, p=LOAN_PURPOSE_PROBS)

    age = np.clip(rng.normal(40, 12, size=n), 18, 75).round(0)
    annual_income = np.clip(rng.lognormal(mean=10.8, sigma=0.5, size=n), 5_000, None).round(2)
    employment_years = np.clip(rng.normal(8, 6, size=n), 0, 45).round(1)
    existing_loans = np.clip(rng.poisson(1.2, size=n), 0, 8)
    credit_utilization_ratio = np.clip(rng.beta(2, 5, size=n), 0.0, 1.0).round(4)
    late_payments_12m = np.clip(rng.poisson(0.5, size=n), 0, 12)
    loan_amount = np.clip(rng.lognormal(mean=9.5, sigma=0.6, size=n), 500, None).round(2)

    # Retired/unemployed applicants realistically have low-to-zero current
    # employment tenure; nudge those rows down deterministically.
    low_tenure_mask = np.isin(employment_type, ["retired", "unemployed"])
    employment_years = np.where(
        low_tenure_mask, np.clip(employment_years * 0.2, 0, None).round(1), employment_years
    )

    employment_effect = np.array([EMPLOYMENT_TYPE_EFFECT[e] for e in employment_type])

    z = (
        _LOGIT_INTERCEPT
        + 1.4 * _standardize(credit_utilization_ratio)
        + 1.1 * _standardize(late_payments_12m.astype(float))
        + 0.6 * _standardize(existing_loans.astype(float))
        - 0.9 * _standardize(annual_income)
        - 0.5 * _standardize(employment_years)
        + employment_effect
    )
    default_prob = _sigmoid(z)
    default_flag = (rng.uniform(size=n) < default_prob).astype(int)

    return pd.DataFrame(
        {
            "employment_type": employment_type,
            "region": region,
            "loan_purpose": loan_purpose,
            "age": age.astype(int),
            "annual_income": annual_income,
            "employment_years": employment_years,
            "existing_loans": existing_loans.astype(int),
            "credit_utilization_ratio": credit_utilization_ratio,
            "late_payments_12m": late_payments_12m.astype(int),
            "loan_amount": loan_amount,
            TARGET_COLUMN: default_flag,
        }
    )
