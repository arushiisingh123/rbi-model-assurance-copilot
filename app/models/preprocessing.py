"""Dataset loading and preprocessing for credit risk modeling (owner: Namitha).

Implements Phase 1 data ingestion, schema validation, train/test splitting,
and target polarity mapping for the UCI Statlog (German Credit Data) dataset.

Authoritative RAW feature schema
--------------------------------
``FEATURE_COLUMNS`` is the single authoritative list of the 20 RAW input
features the credit model expects, in the exact order the model is trained
and scored on. ``CATEGORICAL_FEATURES`` / ``NUMERIC_FEATURES`` split those
20 columns by type. Downstream modules (explainability, fairness, drift,
API) read the model's expected raw inputs from
``model_metadata["feature_names"]`` (see ``app/models/model.py``), which is
derived from this list -- they should never need to guess column names or
inspect the internally one-hot-expanded pipeline columns.

Attribute 9 is named ``personal_status_and_sex`` here, matching CLAUDE.md
and the fairness module. It is a combined marital-status + sex categorical
field (categories ``A91``-``A95``); it is NOT a standalone gender/sex
column and is preserved raw for downstream fairness grouping.

Target label semantics (project-agreed, do not reverse)
------------------------------------------------------
- ``0`` = GOOD  (low credit risk)
- ``1`` = BAD   (high credit risk / likely default)  -- the positive class

Raw UCI ``credit_risk`` uses ``1`` = Good, ``2`` = Bad; ``preprocess()``
maps ``1 -> 0`` and ``2 -> 1``.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATASET_PATH = "data/german_credit/german_credit.csv"

# Project-agreed target label semantics for the credit-risk model.
# 0 = GOOD (low risk), 1 = BAD (high risk / default). 1 is the positive class,
# so P(class 1) = P(BAD). The favorable credit outcome is GOOD, i.e. label 0 --
# downstream fairness/compliance must not assume the favorable label is 1.
LABEL_GOOD = 0
LABEL_BAD = 1
POSITIVE_CLASS = LABEL_BAD
FAVORABLE_OUTCOME_LABEL = LABEL_GOOD

# 20 input features (13 categorical, 7 numeric) matching the approved UCI dataset
CATEGORICAL_FEATURES: List[str] = [
    "status_checking_account",
    "credit_history",
    "purpose",
    "savings_account",
    "present_employment",
    "personal_status_and_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

NUMERIC_FEATURES: List[str] = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "present_residence",
    "age",
    "existing_credits",
    "num_dependents",
]

FEATURE_COLUMNS: List[str] = [
    "status_checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "present_employment",
    "installment_rate",
    "personal_status_and_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
]

TARGET_COLUMN = "credit_risk"
EXPECTED_ROW_COUNT = 1000


def load_dataset(path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the credit dataset from CSV and validate its structure.

    Parameters
    ----------
    path : str
        Path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with validated columns and rows.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If required columns are missing, row count is invalid, or unexpected
        null values are found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Ensure '{DEFAULT_DATASET_PATH}' exists."
        )

    df = pd.read_csv(path)

    # Validate required feature columns
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Dataset at '{path}' is missing required feature columns: {missing_features}"
        )

    # Validate target column if this is the full training dataset
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset at '{path}' is missing target column '{TARGET_COLUMN}'."
        )

    # Validate expected row count for standard German credit dataset
    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} rows in dataset at '{path}', got {len(df)}."
        )

    # Validate missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"Dataset contains unexpected missing values in columns: {cols_with_nulls}"
        )

    return df


def preprocess(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str], List[str]]:
    """Preprocess the dataset by separating features, mapping target polarity.

    Target Mapping:
    - Original raw `1` (Good Credit) -> `0` (Good credit / low risk)
    - Original raw `2` (Bad Credit)  -> `1` (Bad credit / high risk)

    The positive class is `1` (BAD / high risk), so downstream probabilities
    `P(class 1)` mean `P(BAD)`. The favorable credit outcome is `0` (GOOD).

    Note: The raw `personal_status_and_sex` column is preserved unmodified in
    `X`. Fairness grouping belongs strictly to downstream fairness modules.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series], List[str], List[str]]
        (X, y, categorical_column_names, numeric_column_names)
    """
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Input DataFrame is missing required feature columns: {missing_features}"
        )

    if TARGET_COLUMN in df.columns:
        X = df[FEATURE_COLUMNS].copy()
        raw_y = df[TARGET_COLUMN].copy()

        # Map target: 1 -> 0 (good credit), 2 -> 1 (bad credit)
        # Also handles already-mapped {0, 1}
        if set(raw_y.unique()).issubset({1, 2}):
            y = raw_y.map({1: 0, 2: 1}).astype(int)
        elif set(raw_y.unique()).issubset({0, 1}):
            y = raw_y.astype(int)
        else:
            raise ValueError(
                f"Unexpected target values in '{TARGET_COLUMN}': {raw_y.unique()}. "
                "Expected original {1, 2} or mapped {0, 1}."
            )
    else:
        # Prediction mode: only features provided, no target column.
        X = df[FEATURE_COLUMNS].copy()
        y = None

    return X, y, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into stratified train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Seed for deterministic splitting.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
