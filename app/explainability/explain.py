"""Phase 1 Explainability module (owner: Manas).

Provides SHAP and LIME explainability for credit-scoring models.
Produces per-instance feature contributions and global feature importance
following the agreed interface schema (see docs/module-interfaces.md).

Phase 1 Note:
Uses a private helper dummy LogisticRegression model fitted on
data/sample/credit_sample.csv. In Phase 2, this will be wired to Namitha's
real credit model via `app.models.model.load()`.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

# Default features used for Phase 1 credit sample dataset
DEFAULT_FEATURES: List[str] = ["income", "age", "credit_history_len"]
DEFAULT_DATASET_PATH: str = "data/sample/credit_sample.csv"


def _resolve_dataset_path(dataset_path: str = DEFAULT_DATASET_PATH) -> Path:
    """Resolve the dataset file path relative to current working directory or repo root."""
    path = Path(dataset_path)
    if path.exists():
        return path
    # Fallback to path relative to repo root (3 levels up from this file)
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / dataset_path
    if candidate.exists():
        return candidate
    return path


def _fit_dummy_model(
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> Tuple[LogisticRegression, pd.DataFrame, List[str]]:
    """Fit a dummy LogisticRegression model on sample credit data for Phase 1.

    NOTE ON PHASE 2 WIRING:
    This dummy model is a Phase 1 placeholder to enable independent module
    development and testing while Namitha's real model (app/models/model.py)
    is still a Phase 0 stub.
    In Phase 2 Integration, this helper will be replaced by loading the
    trained model artifact via `app.models.model.load()`.

    Returns:
        Tuple containing:
            - fitted LogisticRegression model
            - feature DataFrame X
            - list of feature names
    """
    csv_path = _resolve_dataset_path(dataset_path)
    df = pd.read_csv(csv_path)

    features = [f for f in DEFAULT_FEATURES if f in df.columns]
    X = df[features].copy()
    y = df["default"].copy()

    # max_iter=1000 is headroom, not a fix: on this sample lbfgs already
    # converges on its own tolerance at 96 iterations and the coefficients are
    # byte-identical at max_iter=100 or 20000. The default cap of 100 leaves
    # almost no margin, so an unscaled larger dataset (e.g. the UCI German
    # Credit columns approved in docs/decisions.md) would start hitting the cap
    # and emit ConvergenceWarning. Raising the cap avoids that cliff without
    # adding a scaler or otherwise changing the Phase 1 model design.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    return model, X, features


def _validate_feature_compatibility(
    received_features: List[str],
    expected_features: List[str],
) -> None:
    """Raise ValueError unless the received feature set matches the dummy model's.

    The Phase 1 dummy LogisticRegression is fitted on a fixed feature space.
    Its coefficients are applied positionally, so explaining a different set
    of features would silently produce numbers that look real but describe
    nothing. We refuse that case loudly instead (CLAUDE.md section 6: mock
    results must never be presented as real results).
    """
    if set(received_features) != set(expected_features):
        missing = sorted(set(expected_features) - set(received_features))
        unexpected = sorted(set(received_features) - set(expected_features))
        raise ValueError(
            "Incompatible feature set for the Phase 1 dummy model.\n"
            f"  Expected features: {sorted(expected_features)}\n"
            f"  Received features: {sorted(received_features)}\n"
            f"  Missing from input: {missing}\n"
            f"  Unexpected in input: {unexpected}\n"
            "Phase 1 explainability explains a dummy LogisticRegression trained "
            "only on the expected features above. Explaining a real model or a "
            "different feature space is Phase 2 work (see docs/module-interfaces.md, "
            "'Model artifact access')."
        )


def _explain_shap(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Generate SHAP explanations for the given model and feature matrix.

    Computes:
        - per-instance feature contributions for each row
        - global feature importance (mean absolute SHAP value per feature)

    Returns:
        Agreed explainability output dict with method="shap".
    """
    # Ensure X has the expected columns in correct order
    X_subset = X[feature_names].copy()

    explainer = shap.LinearExplainer(model, X_subset)
    shap_values = explainer(X_subset)

    per_instance = []
    num_rows = len(X_subset)
    for i in range(num_rows):
        contributions = {
            feat: float(shap_values.values[i, j])
            for j, feat in enumerate(feature_names)
        }
        per_instance.append({
            "row_index": int(i),
            "contributions": contributions,
        })

    global_importance = {
        feat: float(np.mean(np.abs(shap_values.values[:, j])))
        for j, feat in enumerate(feature_names)
    }

    return {
        "method": "shap",
        "per_instance": per_instance,
        "global_importance": global_importance,
        "is_mock": False,
    }


def _explain_lime(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Generate LIME explanations for the given model and feature matrix.

    Computes:
        - per-instance feature contributions for each row using LimeTabularExplainer
        - global feature importance (mean absolute LIME weight per feature across instances)

    HOW TO READ THESE NUMBERS (important — LIME and SHAP are NOT the same unit):

    LIME is run with `discretize_continuous=True`. That means LIME does not fit
    a slope against the raw feature. It bins each continuous feature and fits a
    local surrogate against BIN MEMBERSHIP. Internally the explained term for
    row 0 is a condition such as "33750.00 < income <= 47000.00", and we report
    its weight under the bare key "income".

    So a contribution here answers:

        "How much does THIS applicant's income band push the predicted
         probability of default, compared with the other bands?"

    It does NOT answer "how much does one extra rupee of income change the
    prediction". It is a per-instance attribution, not a per-unit slope.

    Consequences the API/dashboard must respect:

    - Units differ from SHAP. LIME weights are on the PREDICTED PROBABILITY
      scale (roughly -1..1 here); `_explain_shap` returns LOG-ODDS. On this
      sample LIME values run ~8-10x smaller. Never plot both on one shared
      axis and never subtract or average them together.
    - Bin edges come from the training sample passed in, so contributions are
      comparable across rows of one call, not across calls on different data.
    - Sign is meaningful and agrees with SHAP on 35 of 36 sample contributions,
      and the global_importance ranking matches SHAP exactly.

    Discretization is passed explicitly rather than relying on the LIME default
    so this choice stays visible in the code and survives a library default
    change. It is deliberate: with `discretize_continuous=False` the surrogate
    of this globally-linear dummy model returns nearly the SAME weight for
    every row (income ~= -0.25 everywhere), which is a global slope and would
    make `per_instance` contributions meaningless.

    Returns:
        Agreed explainability output dict with method="lime".
    """
    X_subset = X[feature_names].copy()
    X_numpy = X_subset.to_numpy()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_numpy,
        feature_names=feature_names,
        class_names=["no_default", "default"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    def predict_fn(x_arr: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x_arr, columns=feature_names)
        return model.predict_proba(x_df)

    per_instance = []
    num_rows = len(X_subset)
    for i in range(num_rows):
        exp = explainer.explain_instance(
            X_numpy[i],
            predict_fn,
            labels=(1,),
            num_features=len(feature_names),
        )
        # exp.as_map()[1] returns list of (feature_index, weight) for label 1
        weights_dict = dict(exp.as_map().get(1, []))
        contributions = {
            feat: float(weights_dict.get(j, 0.0))
            for j, feat in enumerate(feature_names)
        }
        per_instance.append({
            "row_index": int(i),
            "contributions": contributions,
        })

    global_importance = {
        feat: float(np.mean([abs(row["contributions"][feat]) for row in per_instance]))
        for feat in feature_names
    }

    return {
        "method": "lime",
        "per_instance": per_instance,
        "global_importance": global_importance,
        "is_mock": False,
    }


def explain(model_output: Optional[Dict[str, Any]] = None, method: str = "shap") -> Dict[str, Any]:
    """Generate model explanations using SHAP or LIME.

    Public entry point for the explainability module (owner: Manas).

    PHASE 1 SCOPE — IMPORTANT:
    This function ALWAYS explains the Phase 1 dummy LogisticRegression fitted
    on data/sample/credit_sample.csv. It never loads or explains another
    module's trained model. Loading Namitha's real model via
    `app.models.model.load()` is Phase 2 work (see docs/module-interfaces.md,
    "Model artifact access").

    Consequences of that scope:

    - `model_output` may supply feature DATA to explain, but only if its
      feature space exactly matches the dummy model's
      (income, age, credit_history_len). Any other feature set raises
      ValueError rather than returning numbers the dummy model cannot
      legitimately produce.
    - `model_output["model_metadata"]["model_type"]` is IGNORED in Phase 1.
      Passing e.g. "xgboost" does not cause an xgboost model to be explained.
    - `is_mock` is False because the SHAP/LIME computation is genuinely
      performed (not hardcoded) — it does not assert that a production model
      was used.
    - SHAP and LIME contributions are NOT interchangeable. SHAP values are on
      the log-odds scale; LIME values are on the predicted-probability scale
      and describe the effect of the instance's feature BIN, not a per-unit
      slope. Compare rankings between the two, never raw magnitudes, and never
      plot them on a shared axis. See `_explain_lime` for the full explanation.

    Args:
        model_output: Optional dict shaped like Namitha's model output. Only
            'feature_matrix' and 'model_metadata.feature_names' are read, and
            only to select which rows/columns to explain. If None or if
            feature_matrix is absent, the dummy model's own training sample is
            explained instead.
        method: Explainability method to use. Supported: "shap" (default) or "lime".

    Returns:
        dict: Standardized explainability output matching the approved interface:
            {
                "method": "shap" | "lime",
                "per_instance": [
                    {
                        "row_index": 0,
                        "contributions": {"income": 0.31, ...}
                    },
                    ...
                ],
                "global_importance": {"income": 0.42, ...},
                "is_mock": False
            }

    Raises:
        ValueError: If `method` is not 'shap' or 'lime', or if `model_output`
            supplies a feature set the Phase 1 dummy model was not trained on.
    """
    method_lower = method.lower()
    if method_lower not in ("shap", "lime"):
        raise ValueError(
            f"Unsupported explainability method: '{method}'. Supported methods are 'shap' and 'lime'."
        )

    # Phase 1 always explains the dummy model. model_output may only supply
    # feature DATA, and only within the dummy model's own feature space.
    # Phase 2 will load Namitha's real model via app.models.model.load().
    dummy_model, default_X, default_features = _fit_dummy_model()

    if model_output is not None and "feature_matrix" in model_output and model_output["feature_matrix"] is not None:
        raw_matrix = model_output["feature_matrix"]
        if isinstance(raw_matrix, pd.DataFrame):
            X = raw_matrix
        else:
            X = pd.DataFrame(raw_matrix)

        received_features = (
            model_output.get("model_metadata", {}).get("feature_names")
            or list(X.columns)
        )
        _validate_feature_compatibility(received_features, default_features)

        missing_columns = [f for f in default_features if f not in X.columns]
        if missing_columns:
            raise ValueError(
                f"feature_matrix is missing required column(s): {missing_columns}. "
                f"Expected columns: {sorted(default_features)}; "
                f"received columns: {list(X.columns)}."
            )

        # Reorder to the dummy model's training order. The model's coefficients
        # are positional, so a set-equal but differently-ordered input would
        # otherwise attribute each coefficient to the wrong feature.
        X = X[default_features]
        feature_names = default_features
    else:
        X = default_X
        feature_names = default_features

    model = dummy_model

    if method_lower == "shap":
        return _explain_shap(model, X, feature_names)
    else:
        return _explain_lime(model, X, feature_names)
