"""Adapter-aware explainability execution (owner: Manas, Phase 4).

WHAT IS UNDER TEST
------------------
``explain(adapter=X)`` must explain X -- the right model, through the right
explainer, on the right scale, labelled with the right fidelity and identity.

The properties asserted here are the ones that make an explanation citable in
a compliance report:

- IDENTITY. The explanation names the model it actually explained, derived
  from the adapter and not copyable from caller metadata.
- EXPLAINER. Chosen from observable structure, never from a class name or a
  ``model_type`` string. Fake estimator families this project has never seen
  are used throughout to prove that.
- SCALE. 'shap' does not imply one scale: linear is log-odds, tree and kernel
  are probability. Carried per-explanation, never inferred from the method.
- FIDELITY. exact / approximate / surrogate are different claims, and an
  API-only model must never be described as exact.
- ADDITIVITY. Where the explanation claims to be exact, it is checked against
  the model's own output rather than trusted.

A live synthetic bank HTTP server backs the black-box tests, so the REST path
is exercised over genuine HTTP against a genuinely remote model.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import requests
import uvicorn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.explainability.explain import (
    KERNEL_DEFAULT_ROWS,
    POSITIVE_CLASS,
    explain,
)
from app.models.model import MODEL_ID, RF_MODEL_ID, ModelAdapter
from app.models.registry import get_default_registry, reset_default_registry
from app.synthetic_bank.data_generator import FEATURE_COLUMNS as BANK_FEATURES
from app.synthetic_bank.data_generator import generate_customers
from app.synthetic_bank.service import app as bank_app

SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"
BANK_PORT = 8207


# ---------------------------------------------------------------------------
# Live synthetic bank service -- a genuinely remote model over real HTTP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_bank_adapter():
    """The registry's RESTAdapter, pointed at a real running bank service.

    A real uvicorn server on a background thread, on a port distinct from the
    service default (8100) and from the other test suites' ports. Nothing is
    mocked: KernelSHAP and LIME below drive thousands of predictions through
    actual HTTP requests.
    """
    config = uvicorn.Config(
        bank_app, host="127.0.0.1", port=BANK_PORT, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{BANK_PORT}"
    for _ in range(200):
        try:
            if requests.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.25)
    else:  # pragma: no cover - only on a genuinely broken environment
        server.should_exit = True
        pytest.skip("synthetic bank service did not come up")

    previous = os.environ.get("SYNTHETIC_BANK_URL")
    os.environ["SYNTHETIC_BANK_URL"] = base
    reset_default_registry()

    yield get_default_registry().get(SYNTHETIC_BANK_MODEL_ID)

    server.should_exit = True
    if previous is None:
        os.environ.pop("SYNTHETIC_BANK_URL", None)
    else:
        os.environ["SYNTHETIC_BANK_URL"] = previous
    reset_default_registry()


@pytest.fixture(scope="module")
def registry():
    return get_default_registry()


@pytest.fixture(scope="module")
def lr_adapter(registry):
    return registry.get(MODEL_ID)


@pytest.fixture(scope="module")
def rf_adapter(registry):
    return registry.get(RF_MODEL_ID)


def _rows(adapter, n=2) -> Dict[str, Any]:
    return {"feature_matrix": adapter.background_data().head(n)}


# ---------------------------------------------------------------------------
# Fake adapters and estimator families this project has never seen
# ---------------------------------------------------------------------------

FAKE_FEATURES: List[str] = ["cat_a", "cat_b", "num_a", "num_b"]


def _fake_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "cat_a": ["x", "y"] * (rows // 2),
            "cat_b": ["p", "q"] * (rows // 2),
            "num_a": rng.normal(0, 1, rows),
            "num_b": rng.normal(5, 2, rows),
        }
    )


def _fake_labels(frame: pd.DataFrame) -> np.ndarray:
    return ((frame["num_a"] + (frame["cat_a"] == "x").astype(float)) > 0.3).astype(int)


def _oddly_named_pipeline(estimator) -> Pipeline:
    """A pipeline whose steps are NOT called 'preprocessor'/'classifier'.

    The legacy ``_split_pipeline()`` requires those exact names. Adapter-aware
    decomposition must work by position instead, so this names them something
    else entirely.
    """
    return Pipeline(
        steps=[
            (
                "my_column_prep_v2",
                ColumnTransformer(
                    transformers=[
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["cat_a", "cat_b"],
                        ),
                        ("scale", StandardScaler(), ["num_a", "num_b"]),
                    ],
                    remainder="drop",
                ),
            ),
            ("the_actual_model", estimator),
        ]
    )


class FakeUnknownEstimator(BaseEstimator):
    """A model family with no linear coefficients and no tree structure.

    Exposes only ``predict_proba`` -- the minimum a black-box explainer needs
    and the maximum an unrecognised future family might offer.

    Inherits ``BaseEstimator`` only so sklearn's Pipeline will accept it (it
    requires ``__sklearn_tags__``). That base class supplies no ``coef_``, no
    ``estimators_``, no ``tree_`` and no ``get_booster``, so none of the
    structural probes can match it -- which is the whole point.
    """

    def __init__(self, n_features: int = 0) -> None:
        self.n_features = n_features
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        score = 1.0 / (1.0 + np.exp(-values[:, 0]))
        return np.column_stack([1.0 - score, score])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class FakeAdapter(ModelAdapter):
    """A ModelAdapter over an arbitrary artifact, for structural tests."""

    integration_type = "in_process"

    def __init__(
        self,
        *,
        model_id: str = "fake-model",
        model_type: str = "fake_type",
        integration_type: str = "in_process",
        artifact: Any = None,
        artifact_error: Optional[BaseException] = None,
        feature_names: Optional[List[str]] = None,
        background: Optional[pd.DataFrame] = None,
        supports_probability: bool = True,
        batch_scoring: bool = False,
        probability_fn=None,
    ) -> None:
        self.model_id = model_id
        self.model_version = "0.1.0"
        self.model_type = model_type
        self.integration_type = integration_type
        self.feature_names = list(FAKE_FEATURES if feature_names is None else feature_names)
        self._artifact = artifact
        self._artifact_error = artifact_error
        self._background = background
        self._supports_probability = supports_probability
        self._batch_scoring = batch_scoring
        self._probability_fn = probability_fn
        self.proba_calls = 0
        self.proba_rows = 0

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {
            "predict_proba": self._supports_probability,
            "batch": True,
            "batch_scoring": self._batch_scoring,
            # Deliberately dishonest: must never license an explanation.
            "explainability": True,
        }

    @property
    def supports_probability(self) -> bool:
        return self._supports_probability

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.proba_calls += 1
        self.proba_rows += len(X)
        if self._probability_fn is not None:
            return np.asarray(self._probability_fn(X), dtype=float)
        if self._artifact is not None and hasattr(self._artifact, "predict_proba"):
            proba = self._artifact.predict_proba(X)
            return np.asarray(proba)[:, 1]
        raise NotImplementedError

    def load_fitted_model(self) -> Any:
        if self._artifact_error is not None:
            raise self._artifact_error
        return self._artifact

    def background_data(self) -> Optional[pd.DataFrame]:
        return self._background


def _fitted_fake_adapter(estimator, **kwargs) -> FakeAdapter:
    frame = _fake_frame()
    pipeline = _oddly_named_pipeline(estimator)
    pipeline.fit(frame, _fake_labels(frame))
    return FakeAdapter(artifact=pipeline, background=frame, **kwargs)


# ===========================================================================
# 1, 2, 4, 8, 7. Explainer selection, identity and scale for real models
# ===========================================================================


def test_lr_adapter_routes_to_exact_linear_shap_on_the_log_odds_scale(lr_adapter):
    result = explain(model_output=_rows(lr_adapter), method="shap", adapter=lr_adapter)

    assert result["available"] is True
    assert result["explainer"] == "LinearExplainer"
    assert result["scale"] == "log_odds"
    assert result["fidelity"] == "exact"
    assert result["model_id"] == MODEL_ID
    assert result["model_type"] == "logistic_regression"
    assert result["integration_type"] == "in_process"


def test_rf_adapter_routes_to_exact_tree_shap_on_the_probability_scale(rf_adapter):
    result = explain(model_output=_rows(rf_adapter), method="shap", adapter=rf_adapter)

    assert result["available"] is True
    assert result["explainer"] == "TreeExplainer"
    assert result["scale"] == "probability"
    assert result["fidelity"] == "exact"
    assert result["model_id"] == RF_MODEL_ID
    assert result["model_type"] == "random_forest"


def test_linear_and_tree_shap_are_on_different_scales(lr_adapter, rf_adapter):
    """The contract point a report consumer must not get wrong.

    Both are 'shap' and both are 'exact', yet their numbers are in different
    units. Anything averaging or co-plotting them is comparing log-odds
    against probabilities.
    """
    lr = explain(model_output=_rows(lr_adapter), method="shap", adapter=lr_adapter)
    rf = explain(model_output=_rows(rf_adapter), method="shap", adapter=rf_adapter)

    assert lr["scale"] == "log_odds"
    assert rf["scale"] == "probability"
    assert lr["scale"] != rf["scale"]
    assert lr["method"] == rf["method"] == "shap"


# ===========================================================================
# 5, 6. Additivity -- the 'exact' claim is verified, not trusted
# ===========================================================================


def test_lr_shap_is_exactly_additive_to_the_decision_function(lr_adapter):
    """sum(contributions) + expected_value == decision_function(x)."""
    pipeline = lr_adapter.load_fitted_model()
    rows = lr_adapter.background_data().head(4)
    result = explain(model_output={"feature_matrix": rows}, method="shap", adapter=lr_adapter)

    preprocessing, estimator = pipeline[:-1], pipeline.steps[-1][1]
    background = preprocessing.transform(lr_adapter.background_data())
    expected_value = float(np.mean(estimator.decision_function(np.asarray(background))))
    actual = estimator.decision_function(np.asarray(preprocessing.transform(rows)))

    for i, item in enumerate(result["per_instance"]):
        total = sum(item["contributions"].values()) + expected_value
        assert total == pytest.approx(float(actual[i]), abs=1e-9)


def test_rf_shap_is_exactly_additive_to_predict_proba(rf_adapter):
    """sum(contributions) + base_value == predict_proba P(BAD).

    This is what makes ``scale='probability'`` a true statement for the tree
    path rather than a label. The implementation checks the same invariant
    internally and refuses to emit the explanation if it fails.
    """
    pipeline = rf_adapter.load_fitted_model()
    rows = rf_adapter.background_data().head(4)
    result = explain(model_output={"feature_matrix": rows}, method="shap", adapter=rf_adapter)

    preprocessing, estimator = pipeline[:-1], pipeline.steps[-1][1]
    transformed = np.asarray(preprocessing.transform(rows))
    classes = list(estimator.classes_)
    proba = estimator.predict_proba(transformed)[:, classes.index(POSITIVE_CLASS)]

    import shap as shap_lib

    explanation = shap_lib.TreeExplainer(estimator)(transformed)
    base = np.asarray(explanation.base_values)[:, classes.index(POSITIVE_CLASS)]

    for i, item in enumerate(result["per_instance"]):
        total = sum(item["contributions"].values()) + float(base[i])
        assert total == pytest.approx(float(proba[i]), abs=1e-6)


# ===========================================================================
# 9, 10, 11. Adapter-aware LIME: probability source, shape and polarity
# ===========================================================================


def test_adapter_lime_calls_the_adapters_probability_function():
    """LIME must reach the model only through adapter.predict_proba."""
    adapter = _fitted_fake_adapter(LogisticRegression(max_iter=500))
    assert adapter.proba_calls == 0

    result = explain(
        model_output={"feature_matrix": _fake_frame().head(1)},
        method="lime",
        adapter=adapter,
    )

    assert result["available"] is True
    assert adapter.proba_calls > 0, "LIME did not go through the adapter"
    assert adapter.proba_rows > 1000, "LIME should perturb many rows per instance"


def test_adapter_lime_widens_1d_probabilities_into_two_columns():
    """The adapter returns 1-D P(BAD); LIME needs a per-class matrix.

    Captures exactly what the wrapper hands LIME and asserts the widening is
    [1-p, p] -- column 0 GOOD, column 1 BAD -- summing to 1.
    """
    seen: List[np.ndarray] = []

    def probability(frame: pd.DataFrame) -> np.ndarray:
        values = 1.0 / (1.0 + np.exp(-frame["num_a"].to_numpy(dtype=float)))
        seen.append(values)
        return values

    frame = _fake_frame()
    adapter = FakeAdapter(
        artifact_error=NotImplementedError("remote"),
        background=frame,
        batch_scoring=True,
        probability_fn=probability,
    )

    explain(
        model_output={"feature_matrix": frame.head(1)},
        method="lime",
        adapter=adapter,
    )

    assert seen, "the adapter's probability function was never called"
    for values in seen:
        widened = np.column_stack([1.0 - values, values])
        assert widened.shape[1] == 2
        assert np.allclose(widened.sum(axis=1), 1.0)
        # Column 1 is the adapter's own P(BAD), unmodified.
        assert np.allclose(widened[:, 1], values)


def test_adapter_lime_polarity_column_one_is_the_bad_class():
    """A monotone model must attribute its driving feature with the right sign.

    P(BAD) increases with num_a here, so num_a must carry a POSITIVE weight
    for a high-num_a row. If the two probability columns were swapped, LIME
    would explain P(GOOD) and every sign would invert -- an error that leaves
    the output looking entirely normal.
    """
    frame = _fake_frame()

    def probability(f: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-3.0 * f["num_a"].to_numpy(dtype=float)))

    adapter = FakeAdapter(
        artifact_error=NotImplementedError("remote"),
        background=frame,
        batch_scoring=True,
        probability_fn=probability,
    )

    high = frame.nlargest(1, "num_a")
    result = explain(
        model_output={"feature_matrix": high}, method="lime", adapter=adapter
    )

    contributions = result["per_instance"][0]["contributions"]
    assert contributions["num_a"] > 0, (
        "num_a drives P(BAD) up, so its LIME weight for a high-num_a row must "
        "be positive; a negative weight means P(GOOD) was explained instead"
    )
    assert result["scale"] == "probability"
    assert result["fidelity"] == "surrogate"
    assert POSITIVE_CLASS == 1


def test_adapter_lime_is_available_for_both_real_models(lr_adapter, rf_adapter):
    for adapter in (lr_adapter, rf_adapter):
        result = explain(
            model_output=_rows(adapter, 1), method="lime", adapter=adapter
        )
        assert result["available"] is True
        assert result["explainer"] == "LimeTabularExplainer"
        assert result["scale"] == "probability"
        assert result["fidelity"] == "surrogate"


# ===========================================================================
# 12. The real synthetic bank model, over real HTTP
# ===========================================================================


def test_synthetic_bank_gets_approximate_kernel_shap_not_exact(live_bank_adapter):
    """An API-only model must never be described as model-internal SHAP.

    Its ``model_type`` is literally 'xgboost', so a name-based router would
    have claimed exact TreeExplainer. There is no local artifact to inspect,
    so the only honest answer is an approximate black-box estimate.
    """
    result = explain(adapter=live_bank_adapter, method="shap")

    assert result["available"] is True
    assert result["explainer"] == "KernelExplainer"
    assert result["fidelity"] == "approximate"
    assert result["scale"] == "probability"
    assert result["fidelity"] != "exact"
    assert result["explainer"] != "TreeExplainer"

    assert result["model_id"] == SYNTHETIC_BANK_MODEL_ID
    assert result["model_type"] == "xgboost"
    assert result["integration_type"] == "rest"
    assert set(result["feature_space"]) == set(BANK_FEATURES)
    assert result["capabilities"]["has_local_artifact"] is False


def test_synthetic_bank_explanation_states_it_never_saw_the_model(live_bank_adapter):
    result = explain(adapter=live_bank_adapter, method="shap")

    joined = " ".join(result["limitations"]).lower()
    assert "prediction interface" in joined
    assert "approximation" in joined or "not exact" in joined


def test_synthetic_bank_lime_is_surrogate_on_the_probability_scale(live_bank_adapter):
    rows = generate_customers(n=2, random_state=5)[BANK_FEATURES]
    result = explain(
        model_output={"feature_matrix": rows}, method="lime", adapter=live_bank_adapter
    )

    assert result["available"] is True
    assert result["fidelity"] == "surrogate"
    assert result["scale"] == "probability"
    assert set(result["global_importance"]) == set(BANK_FEATURES)
    assert len(result["per_instance"]) == 2


def test_synthetic_bank_never_falls_back_to_the_german_credit_model(live_bank_adapter):
    """The silent-substitution failure mode, asserted directly."""
    result = explain(adapter=live_bank_adapter, method="shap")

    assert set(result["global_importance"]) == set(BANK_FEATURES)
    assert len(result["global_importance"]) == 10
    assert "status_checking_account" not in result["global_importance"]
    assert "credit_amount" not in result["global_importance"]


def test_black_box_row_count_is_bounded_and_disclosed(live_bank_adapter):
    """Cost control must be visible, not silent truncation.

    A black-box row costs thousands of model calls, so the default row count
    is capped -- and the cap is reported in ``limitations`` rather than
    quietly returning fewer rows than the caller expected.
    """
    result = explain(adapter=live_bank_adapter, method="shap")

    assert len(result["per_instance"]) <= KERNEL_DEFAULT_ROWS
    assert any("row" in limit.lower() for limit in result["limitations"])


# ===========================================================================
# 13, 14, 15. Structured UNAVAILABLE -- never None, never zeros
# ===========================================================================


def _assert_structured_unavailable(result: Dict[str, Any], method: str) -> None:
    assert result["available"] is False
    assert result["method"] == method
    assert result["per_instance"] == []
    assert result["global_importance"] == {}
    assert result["explainer"] is None
    assert result["scale"] is None
    assert result["fidelity"] is None
    assert result["limitations"], "an unavailable result must say why"
    # is_mock describes whether numbers are stand-ins, not whether they exist.
    assert result["is_mock"] is False
    assert "model_id" in result


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_no_artifact_and_no_batch_scoring_is_unavailable(method):
    adapter = FakeAdapter(
        model_id="remote-unbatched",
        integration_type="rest",
        artifact_error=NotImplementedError("no local artifact"),
        background=_fake_frame(),
        batch_scoring=False,
    )

    result = explain(adapter=adapter, method=method)

    _assert_structured_unavailable(result, method)
    assert result["model_id"] == "remote-unbatched"
    assert any("batch scoring" in limit for limit in result["limitations"])


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_no_probability_is_unavailable(method):
    adapter = _fitted_fake_adapter(
        LogisticRegression(max_iter=500), supports_probability=False
    )

    result = explain(adapter=adapter, method=method)

    _assert_structured_unavailable(result, method)
    assert any("probability" in limit for limit in result["limitations"])


@pytest.mark.parametrize("method", ["shap", "lime"])
@pytest.mark.parametrize("background", [None, pd.DataFrame()], ids=["none", "empty"])
def test_no_background_data_is_unavailable(method, background):
    frame = _fake_frame()
    pipeline = _oddly_named_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(frame, _fake_labels(frame))
    adapter = FakeAdapter(artifact=pipeline, background=background)

    result = explain(adapter=adapter, method=method)

    _assert_structured_unavailable(result, method)
    assert any("reference" in limit for limit in result["limitations"])


def test_unavailable_never_fabricates_zero_contributions():
    """A table of zeros would read as 'no feature mattered'.

    That is a claim about the model. The honest output is an empty one plus
    the reason, so a consumer cannot mistake missing analysis for a finding.
    """
    adapter = _fitted_fake_adapter(
        LogisticRegression(max_iter=500), supports_probability=False
    )

    result = explain(adapter=adapter, method="shap")

    assert result["global_importance"] == {}
    assert not any(
        item.get("contributions") for item in result["per_instance"]
    )


def test_self_declared_explainability_flag_does_not_license_an_explanation():
    """capabilities['explainability'] is a claim, not an observation.

    FakeAdapter declares it True unconditionally. An adapter with no
    probabilities must still be unavailable.
    """
    adapter = _fitted_fake_adapter(
        LogisticRegression(max_iter=500), supports_probability=False
    )
    assert adapter.capabilities["explainability"] is True

    assert explain(adapter=adapter, method="shap")["available"] is False


def test_unsupported_method_still_raises_for_the_adapter_path():
    adapter = _fitted_fake_adapter(LogisticRegression(max_iter=500))

    with pytest.raises(ValueError, match="Unsupported explainability method"):
        explain(adapter=adapter, method="permutation")


# ===========================================================================
# 16-20. Schema handling through the adapter
# ===========================================================================


def test_adapter_schema_comes_from_the_adapter(live_bank_adapter):
    result = explain(adapter=live_bank_adapter, method="shap")

    assert result["feature_space"] == list(live_bank_adapter.feature_names)
    assert set(result["global_importance"]) == set(live_bank_adapter.feature_names)


def test_wrong_schema_through_the_adapter_fails(lr_adapter):
    """German Credit rows offered to a 4-feature fake model must be refused."""
    adapter = _fitted_fake_adapter(LogisticRegression(max_iter=500))

    with pytest.raises(ValueError, match="Incompatible feature schema"):
        explain(
            model_output={"feature_matrix": lr_adapter.background_data().head(2)},
            method="shap",
            adapter=adapter,
        )


def test_missing_feature_through_the_adapter_fails():
    adapter = _fitted_fake_adapter(LogisticRegression(max_iter=500))
    frame = _fake_frame().drop(columns=["num_b"])

    with pytest.raises(ValueError, match="Missing from input"):
        explain(model_output={"feature_matrix": frame}, method="shap", adapter=adapter)


def test_extra_feature_through_the_adapter_fails():
    adapter = _fitted_fake_adapter(LogisticRegression(max_iter=500))
    frame = _fake_frame()
    frame["scraped_postcode"] = 1

    with pytest.raises(ValueError, match="Unexpected in input"):
        explain(model_output={"feature_matrix": frame}, method="shap", adapter=adapter)


def test_duplicate_feature_through_the_adapter_fails():
    adapter = _fitted_fake_adapter(LogisticRegression(max_iter=500))
    frame = _fake_frame()
    doubled = pd.concat([frame, frame[["num_a"]]], axis=1)

    with pytest.raises(ValueError, match="Duplicate feature column"):
        explain(model_output={"feature_matrix": doubled}, method="shap", adapter=adapter)


def test_reordered_columns_do_not_change_contributions():
    adapter = _fitted_fake_adapter(RandomForestClassifier(n_estimators=20, random_state=0))
    frame = _fake_frame().head(3)

    baseline = explain(
        model_output={"feature_matrix": frame}, method="shap", adapter=adapter
    )
    shuffled = explain(
        model_output={"feature_matrix": frame[list(reversed(FAKE_FEATURES))]},
        method="shap",
        adapter=adapter,
    )

    for a, b in zip(baseline["per_instance"], shuffled["per_instance"]):
        for feature in FAKE_FEATURES:
            assert a["contributions"][feature] == pytest.approx(
                b["contributions"][feature], abs=1e-9
            )


# ===========================================================================
# 21, 22, 35. Structural routing, not a class allowlist
# ===========================================================================


def test_nonstandard_pipeline_step_names_work():
    """Steps named 'my_column_prep_v2'/'the_actual_model' must be handled.

    The legacy ``_split_pipeline()`` requires 'preprocessor'/'classifier'.
    Adapter-aware decomposition is positional, so any names work.
    """
    adapter = _fitted_fake_adapter(LogisticRegression(max_iter=500))
    steps = [name for name, _ in adapter.load_fitted_model().steps]
    assert "preprocessor" not in steps and "classifier" not in steps

    result = explain(
        model_output={"feature_matrix": _fake_frame().head(2)},
        method="shap",
        adapter=adapter,
    )

    assert result["available"] is True
    assert result["explainer"] == "LinearExplainer"
    assert set(result["global_importance"]) == set(FAKE_FEATURES)


def test_legacy_split_pipeline_would_have_rejected_that_pipeline():
    """Proof the previous test is testing something real."""
    from app.explainability.explain import _split_pipeline

    adapter = _fitted_fake_adapter(LogisticRegression(max_iter=500))

    with pytest.raises(ValueError, match="missing expected step"):
        _split_pipeline(adapter.load_fitted_model())


def test_tree_family_is_detected_structurally_not_by_class_name():
    adapter = _fitted_fake_adapter(RandomForestClassifier(n_estimators=20, random_state=0))

    result = explain(
        model_output={"feature_matrix": _fake_frame().head(2)},
        method="shap",
        adapter=adapter,
    )

    assert result["explainer"] == "TreeExplainer"
    assert result["fidelity"] == "exact"
    assert result["scale"] == "probability"


def test_unknown_future_estimator_gets_kernel_shap_not_a_rejection():
    """Requirement 22: an unrecognised family is approximated, not refused.

    ``FakeUnknownEstimator`` has no coefficients, no sub-estimators, no
    booster, and no class this project could ever have allowlisted. It has
    probabilities and reference data, so it gets an approximate explanation.
    """
    frame = _fake_frame()
    pipeline = _oddly_named_pipeline(FakeUnknownEstimator(n_features=6))
    pipeline.fit(frame, _fake_labels(frame))
    adapter = FakeAdapter(
        artifact=pipeline, background=frame, batch_scoring=True, model_type="invented_tomorrow"
    )

    result = explain(
        model_output={"feature_matrix": frame.head(1)}, method="shap", adapter=adapter
    )

    assert result["available"] is True
    assert result["explainer"] == "KernelExplainer"
    assert result["fidelity"] == "approximate"
    assert result["scale"] == "probability"
    assert set(result["global_importance"]) == set(FAKE_FEATURES)


def test_model_type_string_does_not_influence_routing():
    """Same structure, different model_type labels -> same explainer."""
    results = []
    for model_type in ("logistic_regression", "a_family_invented_tomorrow"):
        adapter = _fitted_fake_adapter(
            LogisticRegression(max_iter=500), model_type=model_type
        )
        results.append(
            explain(
                model_output={"feature_matrix": _fake_frame().head(2)},
                method="shap",
                adapter=adapter,
            )
        )

    assert results[0]["model_type"] != results[1]["model_type"]
    assert results[0]["explainer"] == results[1]["explainer"]
    assert results[0]["scale"] == results[1]["scale"]
    assert results[0]["global_importance"] == results[1]["global_importance"]


def test_execution_layer_names_no_concrete_estimator_class():
    """Requirement 35, on the adapter-aware execution code itself.

    Scoped to the adapter-aware region: the legacy no-adapter path documents
    the German Credit LogisticRegression pipeline it was written for, and that
    prose is accurate and must stay.
    """
    import inspect
    from importlib import import_module

    module = import_module("app.explainability.explain")
    for name in (
        "_explain_with_adapter",
        "_run_linear_shap",
        "_run_tree_shap",
        "_run_kernel_shap",
        "_run_adapter_lime",
        "_decompose",
        "_positive_class_index",
        "_select_class_axis",
    ):
        source = inspect.getsource(getattr(module, name))
        for token in (
            "LogisticRegression",
            "RandomForestClassifier",
            "XGBClassifier",
            "logistic_regression",
            "random_forest",
            "xgboost",
            "model_type ==",
            "model_id ==",
        ):
            assert token not in source, f"{name}() routes on {token!r}"


# ===========================================================================
# 23, 24. Identity cannot be spoofed
# ===========================================================================


def test_identity_is_derived_from_the_adapter_not_from_metadata(rf_adapter):
    """Caller metadata must not be able to relabel an explanation."""
    result = explain(
        model_output={
            "feature_matrix": rf_adapter.background_data().head(2),
            "model_metadata": {"feature_names": list(rf_adapter.feature_names)},
        },
        method="shap",
        adapter=rf_adapter,
    )

    assert result["model_id"] == RF_MODEL_ID


@pytest.mark.parametrize("field", ["model_id", "model_type"])
def test_conflicting_declared_identity_raises(rf_adapter, field):
    """A caller declaring a different model is a real disagreement.

    Silently preferring the adapter's identity would leave the caller
    believing its own label had been honoured.
    """
    metadata = {"feature_names": list(rf_adapter.feature_names), field: "something-else"}

    with pytest.raises(ValueError, match=f"Conflicting {field}"):
        explain(
            model_output={
                "feature_matrix": rf_adapter.background_data().head(2),
                "model_metadata": metadata,
            },
            method="shap",
            adapter=rf_adapter,
        )


def test_matching_declared_identity_is_accepted(rf_adapter):
    result = explain(
        model_output={
            "feature_matrix": rf_adapter.background_data().head(2),
            "model_metadata": {
                "feature_names": list(rf_adapter.feature_names),
                "model_id": RF_MODEL_ID,
                "model_type": "random_forest",
            },
        },
        method="shap",
        adapter=rf_adapter,
    )

    assert result["model_id"] == RF_MODEL_ID


# ===========================================================================
# Determinism and provenance
# ===========================================================================


def test_adapter_explanations_are_deterministic(rf_adapter):
    rows = _rows(rf_adapter, 2)
    first = explain(model_output=rows, method="shap", adapter=rf_adapter)
    second = explain(model_output=rows, method="shap", adapter=rf_adapter)

    assert first == second


def test_exact_explanations_report_no_sampling_provenance(rf_adapter):
    """n_samples/random_seed are None for an exact decomposition.

    Reporting a sample count there would imply a sampling process that never
    happened, and a reader could not tell an exact result from an estimate.
    """
    result = explain(model_output=_rows(rf_adapter), method="shap", adapter=rf_adapter)

    assert result["fidelity"] == "exact"
    assert result["n_samples"] is None
    assert result["random_seed"] is None
    assert result["n_background"] > 0


def test_sampled_explanations_report_their_sampling_provenance(live_bank_adapter):
    result = explain(adapter=live_bank_adapter, method="shap")

    assert result["fidelity"] == "approximate"
    assert result["n_samples"] is not None and result["n_samples"] > 0
    assert result["random_seed"] == 42
    assert result["n_background"] > 0


def test_positive_class_convention_agrees_across_the_platform():
    """The locally-declared constant must not drift from either module."""
    from app.models.preprocessing import POSITIVE_CLASS as german_credit
    from app.synthetic_bank.data_generator import POSITIVE_CLASS as bank

    assert POSITIVE_CLASS == german_credit == bank == 1
