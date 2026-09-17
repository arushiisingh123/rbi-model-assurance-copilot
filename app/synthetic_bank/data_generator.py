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

from typing import Any, Dict, List, Optional

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

# Stable per-record identity. IDENTITY METADATA ONLY -- deliberately NOT in
# FEATURE_COLUMNS, so it can never reach the model as a feature. It exists so
# explainability evidence can be attributed to a specific applicant: explain()
# reports row_index, which is a POSITION inside whatever frame was explained
# and restarts at 0 on every call, so it cannot anchor evidence on its own.
# Mirrors the role app/models/preprocessing.py's INSTANCE_ID_COLUMN plays for
# German Credit.
CUSTOMER_ID_COLUMN = "customer_id"
CUSTOMER_ID_PREFIX = "SB"
LABEL_GOOD = 0
LABEL_BAD = 1
POSITIVE_CLASS = LABEL_BAD

# Additive metadata describing default_flag's semantics -- does not change
# LABEL_GOOD/LABEL_BAD/POSITIVE_CLASS above, just documents them in a
# structured form for callers building model_metadata (e.g. a future
# RESTAdapter.trained_on / label_semantics wiring, owned by Khushi).
LABEL_SEMANTICS: Dict[str, Any] = {
    "0": "GOOD - repaid / favorable",
    "1": "BAD - defaulted / unfavorable",
    "positive_class": POSITIVE_CLASS,
    "probabilities_represent": "P(class == 1) = P(BAD)",
    "favorable_outcome_label": LABEL_GOOD,
}

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


def make_customer_ids(n: int, random_state: int) -> List[str]:
    """Deterministic, unique customer identifiers for a generated population.

    Derived purely from ``(random_state, position)`` -- NO random draws are
    involved. That matters twice over: the same call always produces the same
    ids, and adding identity to a population cannot perturb the rng call
    order or any generated feature distribution.

    Row ``i`` of a population keeps its id across regenerations, and across
    population sizes, which is the durability that lets explainability
    evidence reference an applicant after the frame has been re-derived.

    The seed is part of the id because two different populations are two
    different sets of people; sharing ids between them would make records
    from distinct scenarios collide. That is why the baseline (42), the
    drift scenario (99), and the registry's background batch (123) each get
    a disjoint id space for free.
    """
    return [f"{CUSTOMER_ID_PREFIX}-{random_state:04d}-{i:06d}" for i in range(n)]


def _generate_population(
    n: int,
    random_state: int,
    *,
    employment_type_probs: Optional[List[float]] = None,
    utilization_alpha: float = 2.0,
    utilization_beta: float = 5.0,
    late_payments_lambda: float = 0.5,
) -> pd.DataFrame:
    """Shared population-generation core behind every ``generate_*`` function
    in this module.

    The keyword-only overrides let ``generate_drift_customers()`` shift
    specific input-feature distributions (employment mix, credit
    utilization, delinquency rate) while reusing the exact same generation
    logic, rng call order, and clipping/rounding as the baseline population
    -- so ``generate_customers()`` (all overrides at their defaults) remains
    byte-identical to its pre-refactor output for a given ``random_state``.

    Every population built here carries a durable ``customer_id`` (see
    ``make_customer_ids()``). Attaching it at this ONE shared core -- rather
    than in each ``generate_*`` wrapper -- is what keeps identity consistent
    across the baseline and drift scenarios without duplicating any
    generation logic. It is derived arithmetically from ``(random_state,
    position)``, draws no randomness, and is appended after every rng call
    below, so the feature values are bit-for-bit what they were before
    identity existed.
    """
    rng = np.random.RandomState(random_state)
    employment_type_probs = employment_type_probs or EMPLOYMENT_TYPE_PROBS

    employment_type = rng.choice(EMPLOYMENT_TYPES, size=n, p=employment_type_probs)
    region = rng.choice(REGIONS, size=n, p=REGION_PROBS)
    loan_purpose = rng.choice(LOAN_PURPOSES, size=n, p=LOAN_PURPOSE_PROBS)

    age = np.clip(rng.normal(40, 12, size=n), 18, 75).round(0)
    annual_income = np.clip(rng.lognormal(mean=10.8, sigma=0.5, size=n), 5_000, None).round(2)
    employment_years = np.clip(rng.normal(8, 6, size=n), 0, 45).round(1)
    existing_loans = np.clip(rng.poisson(1.2, size=n), 0, 8)
    credit_utilization_ratio = np.clip(
        rng.beta(utilization_alpha, utilization_beta, size=n), 0.0, 1.0
    ).round(4)
    late_payments_12m = np.clip(rng.poisson(late_payments_lambda, size=n), 0, 12)
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
            # Identity first, so a reader sees whose row this is before its
            # features. Never in FEATURE_COLUMNS, so it cannot reach the model.
            CUSTOMER_ID_COLUMN: make_customer_ids(n, random_state),
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


def generate_customers(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Deterministic synthetic credit-applicant population -- the
    **baseline/normal** scenario every other generator in this module is
    defined relative to (``generate_reference()`` and
    ``generate_current("normal")`` both delegate here).

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
        Columns: [CUSTOMER_ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN],
        ``n`` rows. ``customer_id`` is identity metadata, never a feature --
        consumers that feed this to a model select ``FEATURE_COLUMNS``.
    """
    return _generate_population(n, random_state)


# ---------------------------------------------------------------------------
# Controlled drift scenario (DATA ONLY)
# ---------------------------------------------------------------------------

# Employment mix shifted toward unemployed (0.10 -> 0.30) relative to
# EMPLOYMENT_TYPE_PROBS above; still sums to 1.0 over the same EMPLOYMENT_TYPES.
DRIFT_EMPLOYMENT_TYPE_PROBS: List[float] = [0.35, 0.20, 0.30, 0.15]
# Beta(4, 3) has mean ~0.571 vs. the baseline Beta(2, 5)'s ~0.286 -- still
# clipped to the same documented [0.0, 1.0] range as generate_customers().
DRIFT_UTILIZATION_ALPHA = 4.0
DRIFT_UTILIZATION_BETA = 3.0
# Higher Poisson rate than the baseline's 0.5 -- still clipped to the same
# documented [0, 12] range as generate_customers().
DRIFT_LATE_PAYMENTS_LAMBDA = 2.0


def generate_drift_customers(n: int = 1000, random_state: int = 99) -> pd.DataFrame:
    """Controlled, deterministic INPUT-FEATURE-distribution shift for
    exercising the assurance pipeline's drift detection.

    Shifts three specific inputs relative to ``generate_customers()``'s
    baseline distribution -- ``employment_type`` mix (more unemployed),
    ``credit_utilization_ratio`` (higher), and ``late_payments_12m``
    (higher) -- while every value stays within the same documented valid
    ranges as the baseline population (utilization in [0.0, 1.0],
    late payments in [0, 12], employment_type still drawn from
    ``EMPLOYMENT_TYPES``). ``region`` and ``loan_purpose`` are unaffected.

    This function computes NO drift metric, threshold, or status of any
    kind -- it only produces DataFrames. PSI/KS calculation and PASS/
    WARNING/FAIL classification belong exclusively to ``app/drift/`` and
    are unaffected by this function.

    **This is SYNTHETIC SCENARIO DATA, not observed real-world population
    drift.** It must never be presented as observed drift, matching the
    same rule already documented for ``app/drift/scenario.py``.

    Note: because ``default_flag`` is derived from these same shifted
    inputs (see ``generate_customers()``'s docstring), the resulting
    default rate will typically also shift -- that is an expected side
    effect of shifting real risk features, not a separate guarantee this
    function makes or requires.

    Returns
    -------
    pd.DataFrame
        Columns: [CUSTOMER_ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN],
        ``n`` rows. The drift seed differs from the baseline's, so drift
        customers occupy a disjoint ``customer_id`` space -- a drifted
        population is a different set of people, not the same people again.
    """
    return _generate_population(
        n,
        random_state,
        employment_type_probs=DRIFT_EMPLOYMENT_TYPE_PROBS,
        utilization_alpha=DRIFT_UTILIZATION_ALPHA,
        utilization_beta=DRIFT_UTILIZATION_BETA,
        late_payments_lambda=DRIFT_LATE_PAYMENTS_LAMBDA,
    )


# ---------------------------------------------------------------------------
# Edge-case input generation
# ---------------------------------------------------------------------------

# Hand-specified boundary/extreme-value rows, each within the ranges
# generate_customers() documents (age 18-75, annual_income floor 5_000,
# employment_years >= 0, existing_loans 0-8, credit_utilization_ratio
# 0.0-1.0, late_payments_12m 0-12). No target column: these are inputs for
# stress-testing scoring, not a labeled training population -- there is no
# meaningful way to compute a population-relative default_flag (the risk
# model standardizes each feature against a population's own mean/std,
# which a handful of hand-picked rows does not have).
_EDGE_CASE_ROWS: List[Dict[str, Any]] = [
    {  # minimum age, floor income, spotless history
        "employment_type": "salaried", "region": "north", "loan_purpose": "auto",
        "age": 18, "annual_income": 5_000.0, "employment_years": 0.0,
        "existing_loans": 0, "credit_utilization_ratio": 0.0,
        "late_payments_12m": 0, "loan_amount": 500.0,
    },
    {  # maximum age, retired, worst-case utilization/delinquency/loan count
        "employment_type": "retired", "region": "south", "loan_purpose": "business",
        "age": 75, "annual_income": 5_000.0, "employment_years": 0.0,
        "existing_loans": 8, "credit_utilization_ratio": 1.0,
        "late_payments_12m": 12, "loan_amount": 500.0,
    },
    {  # unemployed, worst-case utilization/delinquency/loan count
        "employment_type": "unemployed", "region": "east", "loan_purpose": "personal",
        "age": 40, "annual_income": 5_000.0, "employment_years": 0.0,
        "existing_loans": 8, "credit_utilization_ratio": 1.0,
        "late_payments_12m": 12, "loan_amount": 500.0,
    },
    {  # self-employed, rare loan purpose, high income/tenure, zero risk signals
        "employment_type": "self_employed", "region": "west", "loan_purpose": "education",
        "age": 45, "annual_income": 500_000.0, "employment_years": 40.0,
        "existing_loans": 0, "credit_utilization_ratio": 0.0,
        "late_payments_12m": 0, "loan_amount": 1_000_000.0,
    },
]


def generate_edge_case_customers() -> pd.DataFrame:
    """Deterministic, hand-specified boundary/extreme-value input rows.

    Not a random sample -- every call returns the exact same rows, each
    documented in ``_EDGE_CASE_ROWS`` above. Purpose is to stress-test the
    adapter/model pipeline against schema edges (min/max age, income floor,
    full utilization, maximum delinquency/loan count, rare categories);
    this function does not change model behavior or compute any assurance
    result.

    Carries NO ``customer_id``, deliberately. These are hand-specified
    schema probes rather than a generated population of people -- there is
    no seed or row provenance for an id to be durable against, and labelling
    a boundary probe as a customer would invite evidence being attributed to
    an applicant who does not exist. Its exact input-only schema is also a
    contract callers already rely on.

    Returns
    -------
    pd.DataFrame
        Columns: FEATURE_COLUMNS only (no TARGET_COLUMN, no
        CUSTOMER_ID_COLUMN).
    """
    return pd.DataFrame(_EDGE_CASE_ROWS)[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Missing-data input generation
# ---------------------------------------------------------------------------


def generate_missing_data_customers(
    n: int = 500,
    random_state: int = 11,
    missing_rate: float = 0.1,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Baseline population (``generate_customers()``) with deterministic
    NaN values injected into ``columns`` at ``missing_rate``.

    This function ONLY produces the missing-data scenario; it does not
    change ``app/synthetic_bank/model.py``'s pipeline to tolerate the
    missingness it introduces. Whether the current pipeline
    (``OneHotEncoder(handle_unknown="ignore")`` + XGBoost) accepts or
    raises on these rows is left exactly as-is and is not a contract this
    function establishes.

    Parameters
    ----------
    n : int
        Number of rows to generate (via ``generate_customers()``).
    random_state : int
        Seeds both the underlying population and the missingness mask,
        deterministically.
    missing_rate : float
        Fraction of values to null out per column, in ``[0.0, 1.0)``.
        ``0.0`` injects no missingness; ``1.0`` is rejected (a fully-null
        column is a different scenario, not "missing data").
    columns : Optional[List[str]]
        Columns to inject missingness into. Defaults to all of
        ``FEATURE_COLUMNS``. Must be a subset of ``FEATURE_COLUMNS``.

    Carries NO ``customer_id``: this is an input-only scenario whose exact
    FEATURE_COLUMNS schema callers already rely on. Note the useful
    corollary of ``columns`` having to be a subset of ``FEATURE_COLUMNS`` --
    identity could never be among the nulled columns even if it were
    present, because a missing identity is not a missing feature.

    Returns
    -------
    pd.DataFrame
        Columns: FEATURE_COLUMNS only (no TARGET_COLUMN, no
        CUSTOMER_ID_COLUMN), ``n`` rows, with NaN values injected per the
        parameters above.
    """
    if not (0.0 <= missing_rate < 1.0):
        raise ValueError(f"missing_rate must be in [0.0, 1.0), got {missing_rate}")

    columns = list(columns) if columns is not None else list(FEATURE_COLUMNS)
    unknown_columns = set(columns) - set(FEATURE_COLUMNS)
    if unknown_columns:
        raise ValueError(
            f"Unknown columns for missing-data injection: {sorted(unknown_columns)}; "
            f"expected a subset of {FEATURE_COLUMNS}"
        )

    df = generate_customers(n=n, random_state=random_state)[FEATURE_COLUMNS].copy()

    # Separate rng stream from generate_customers()'s own, seeded off the
    # same random_state for reproducibility without correlating the
    # missingness mask with the population's own draws.
    mask_rng = np.random.RandomState(random_state + 1)
    for col in columns:
        missing_mask = mask_rng.uniform(size=len(df)) < missing_rate
        df.loc[missing_mask, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Named reference/current datasets (DATA ONLY -- no drift calculation)
# ---------------------------------------------------------------------------

CURRENT_SCENARIOS: List[str] = ["normal", "drift", "edge", "missing"]


def generate_reference(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Deterministic baseline/reference population.

    A named alias over ``generate_customers()`` (the "normal" scenario) for
    callers -- e.g. Arushi's drift module -- that want an explicitly-named
    reference dataset rather than reaching into ``generate_customers()``
    directly. Computes no drift metric; returns a DataFrame only.

    Returns
    -------
    pd.DataFrame
        Columns: [CUSTOMER_ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN],
        ``n`` rows.
    """
    return generate_customers(n=n, random_state=random_state)


def generate_current(
    scenario: str = "normal",
    n: int = 1000,
    random_state: int = 43,
    **scenario_kwargs: Any,
) -> pd.DataFrame:
    """Deterministic "current" population for a named scenario.

    Purely a data-generation dispatcher over the other ``generate_*``
    functions in this module -- computes no drift/fairness metric,
    threshold, or status. Callers needing PSI/KS/status must still call
    ``app/drift/``'s own functions on the DataFrame this returns.

    Parameters
    ----------
    scenario : str
        One of ``CURRENT_SCENARIOS``: ``"normal"``, ``"drift"``, ``"edge"``,
        or ``"missing"``.
    n, random_state :
        Forwarded to the underlying generator for the ``"normal"``,
        ``"drift"``, and ``"missing"`` scenarios. Ignored for ``"edge"``,
        which is a fixed hand-specified row set with no size or seed.
    **scenario_kwargs :
        Forwarded to ``generate_missing_data_customers()`` (e.g.
        ``missing_rate``, ``columns``) when ``scenario == "missing"``.
        Unused for every other scenario.

    Returns
    -------
    pd.DataFrame
        ``"normal"``/``"drift"``: [CUSTOMER_ID_COLUMN] + FEATURE_COLUMNS +
        [TARGET_COLUMN].
        ``"edge"``/``"missing"``: FEATURE_COLUMNS only (no target column and
        no customer id -- see the respective generator's docstring for why).

    Raises
    ------
    ValueError
        If ``scenario`` is not one of ``CURRENT_SCENARIOS``.
    """
    if scenario == "normal":
        return generate_customers(n=n, random_state=random_state)
    if scenario == "drift":
        return generate_drift_customers(n=n, random_state=random_state)
    if scenario == "edge":
        return generate_edge_case_customers()
    if scenario == "missing":
        return generate_missing_data_customers(
            n=n, random_state=random_state, **scenario_kwargs
        )
    raise ValueError(
        f"Unknown scenario '{scenario}'; expected one of {CURRENT_SCENARIOS}"
    )
