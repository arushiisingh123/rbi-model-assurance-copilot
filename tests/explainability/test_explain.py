"""Unit and smoke tests for app.explainability (Phase 1).

Verifies SHAP and LIME explainability against the Phase 1 dummy model
and sample credit dataset.

Renamed from test_explain_skeleton.py: these no longer test a Phase 0
stub, they test the real Phase 1 SHAP/LIME implementation.
"""

import warnings

import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from app.explainability.explain import (
    DEFAULT_FEATURES,
    _fit_dummy_model,
    explain,
)


def test_explain_no_args_shap_default():
    """Verify explain() runs end-to-end with no arguments using SHAP."""
    result = explain()

    assert result["method"] == "shap"
    assert result["is_mock"] is False
    assert "per_instance" in result
    assert "global_importance" in result


def test_explain_per_instance_length_matches_rows():
    """Verify per_instance length matches the 12 sample rows in credit_sample.csv."""
    result = explain()
    per_instance = result["per_instance"]

    assert len(per_instance) == 12
    for i, item in enumerate(per_instance):
        assert item["row_index"] == i
        assert isinstance(item["contributions"], dict)
        assert set(item["contributions"].keys()) == set(DEFAULT_FEATURES)
        for feat, val in item["contributions"].items():
            assert isinstance(val, (int, float))
            assert not pd.isna(val)


def test_explain_global_importance_keys_and_values():
    """Verify global_importance keys match feature names and have non-negative numeric values."""
    result = explain()
    global_imp = result["global_importance"]

    assert set(global_imp.keys()) == set(DEFAULT_FEATURES)
    for feat, val in global_imp.items():
        assert isinstance(val, (int, float))
        assert not pd.isna(val)
        assert val >= 0.0


def test_explain_lime_method():
    """Verify explain(method='lime') runs end-to-end and returns correct structure."""
    result = explain(method="lime")

    assert result["method"] == "lime"
    assert result["is_mock"] is False
    assert len(result["per_instance"]) == 12

    for i, item in enumerate(result["per_instance"]):
        assert item["row_index"] == i
        assert isinstance(item["contributions"], dict)
        assert set(item["contributions"].keys()) == set(DEFAULT_FEATURES)
        for feat, val in item["contributions"].items():
            assert isinstance(val, (int, float))
            assert not pd.isna(val)

    global_imp = result["global_importance"]
    assert set(global_imp.keys()) == set(DEFAULT_FEATURES)
    for feat, val in global_imp.items():
        assert isinstance(val, (int, float))
        assert not pd.isna(val)
        assert val >= 0.0


def test_explain_with_custom_model_output():
    """Verify explain() handles custom feature_matrix provided via model_output dict."""
    custom_df = pd.DataFrame({
        "income": [30000, 60000, 90000],
        "age": [25, 40, 55],
        "credit_history_len": [2, 8, 15],
    })
    # model_type is "stub" on purpose: Phase 1 always explains the dummy
    # LogisticRegression, so claiming a real model here would be false.
    model_output = {
        "predictions": [1, 0, 0],
        "probabilities": [0.8, 0.2, 0.1],
        "feature_matrix": custom_df,
        "model_metadata": {
            "model_type": "stub",
            "version": "0.1.0",
            "feature_names": ["income", "age", "credit_history_len"],
        },
        "is_mock": False,
    }

    result = explain(model_output=model_output, method="shap")
    assert result["method"] == "shap"
    assert result["is_mock"] is False
    assert len(result["per_instance"]) == 3
    assert set(result["global_importance"].keys()) == {"income", "age", "credit_history_len"}


def test_explain_invalid_method_raises_error():
    """Verify passing an unsupported method raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported explainability method"):
        explain(method="unknown_method")


def test_explain_rejects_foreign_feature_names():
    """Foreign features must raise, not be silently scored by the dummy model.

    Regression guard: previously a German-Credit-style feature matrix was
    explained using the dummy model's income/age/credit_history_len
    coefficients applied positionally, and returned is_mock=False.
    """
    foreign_df = pd.DataFrame({
        "duration_months": [6, 48, 12],
        "credit_amount": [1169, 5951, 2096],
        "employment_yrs": [5, 2, 3],
    })
    model_output = {
        "feature_matrix": foreign_df,
        "model_metadata": {"feature_names": list(foreign_df.columns)},
    }

    with pytest.raises(ValueError, match="Incompatible feature set"):
        explain(model_output=model_output)


def test_explain_rejects_wrong_feature_count():
    """A feature matrix with the wrong number of features must raise clearly.

    Previously this surfaced as an opaque numpy error:
    "shapes (3,) and (20,) not aligned".
    """
    wide_df = pd.DataFrame({f"feature_{i}": [1.0, 2.0, 3.0] for i in range(20)})
    model_output = {
        "feature_matrix": wide_df,
        "model_metadata": {"feature_names": list(wide_df.columns)},
    }

    with pytest.raises(ValueError, match="Incompatible feature set"):
        explain(model_output=model_output)


def test_explain_error_message_names_expected_and_received():
    """The ValueError must state both expected and received features."""
    foreign_df = pd.DataFrame({"credit_amount": [1169], "duration_months": [6], "age": [30]})
    model_output = {
        "feature_matrix": foreign_df,
        "model_metadata": {"feature_names": list(foreign_df.columns)},
    }

    with pytest.raises(ValueError) as exc_info:
        explain(model_output=model_output)

    message = str(exc_info.value)
    assert "Expected features" in message
    assert "Received features" in message
    for feature in DEFAULT_FEATURES:
        assert feature in message
    assert "credit_amount" in message


def test_explain_handles_reordered_feature_names():
    """Set-equal but reordered features must be realigned, not misattributed.

    The dummy model's coefficients are positional, so column order matters.
    """
    ordered = pd.DataFrame({
        "income": [30000, 60000, 90000],
        "age": [25, 40, 55],
        "credit_history_len": [2, 8, 15],
    })
    reordered = ordered[["age", "credit_history_len", "income"]]

    baseline = explain(model_output={
        "feature_matrix": ordered,
        "model_metadata": {"feature_names": list(ordered.columns)},
    })
    shuffled = explain(model_output={
        "feature_matrix": reordered,
        "model_metadata": {"feature_names": list(reordered.columns)},
    })

    assert baseline["global_importance"] == pytest.approx(shuffled["global_importance"])
    for base_row, shuf_row in zip(baseline["per_instance"], shuffled["per_instance"]):
        assert base_row["contributions"] == pytest.approx(shuf_row["contributions"])


def test_lime_output_is_deterministic():
    """LIME evidence must be reproducible across runs.

    Compliance evidence that changes between runs cannot be cited in a
    report, so the explainer's random_state must actually pin the sampling.
    """
    first = explain(method="lime")
    second = explain(method="lime")

    assert first["global_importance"] == pytest.approx(second["global_importance"])
    for a, b in zip(first["per_instance"], second["per_instance"]):
        assert a["contributions"] == pytest.approx(b["contributions"])


def test_lime_contributions_vary_per_instance():
    """Pin the discretized LIME semantics documented in _explain_lime.

    With discretize_continuous=True, a contribution is the effect of THIS
    row's feature bin, so it differs row to row and takes both signs across
    the sample. If discretization were switched off, the surrogate of this
    globally-linear dummy model would return essentially the same weight for
    every row (income ~= -0.25 everywhere) and per_instance contributions
    would stop being per-instance. This test fails if that happens.
    """
    result = explain(method="lime")
    per_feature = {
        feature: [row["contributions"][feature] for row in result["per_instance"]]
        for feature in DEFAULT_FEATURES
    }

    signs_seen = {
        feature: {v > 0 for v in values if v != 0.0}
        for feature, values in per_feature.items()
    }
    assert any(len(s) == 2 for s in signs_seen.values()), (
        "No feature changed sign across instances - LIME contributions look "
        f"like a global slope rather than per-instance attributions: {per_feature}"
    )


def test_lime_and_shap_agree_on_ranking_but_not_scale():
    """Document the cross-method relationship the dashboard must respect.

    The two methods rank features the same way, but their magnitudes are on
    different scales (LIME: predicted probability, SHAP: log-odds). They must
    never be plotted on a shared axis or averaged together.
    """
    shap_result = explain(method="shap")
    lime_result = explain(method="lime")

    def ranking(result):
        importance = result["global_importance"]
        return sorted(importance, key=importance.get, reverse=True)

    assert ranking(shap_result) == ranking(lime_result)

    shap_scale = sum(shap_result["global_importance"].values())
    lime_scale = sum(lime_result["global_importance"].values())
    assert shap_scale > lime_scale, (
        "Expected SHAP log-odds magnitudes to exceed LIME probability-scale "
        "magnitudes; if this changed, the documented scale warning in "
        "_explain_lime needs revisiting."
    )


def test_dummy_model_converges_without_warning():
    """The dummy model must converge cleanly, not stop at the iteration cap."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, _, _ = _fit_dummy_model()

    convergence_warnings = [
        w for w in caught if issubclass(w.category, ConvergenceWarning)
    ]
    assert not convergence_warnings, f"Model did not converge: {convergence_warnings}"
    assert model.n_iter_[0] < model.max_iter
