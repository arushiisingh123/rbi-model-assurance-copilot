"""Tests for the explainability capability/routing layer (owner: Manas, Step 2).

The property under test throughout is that routing follows *observable
structure*, never model identity. Several tests deliberately use fake
estimators and fake adapters that no part of this project has ever seen: if
the capability layer routes those correctly, it will route a bank's model
correctly too, and that is the entire point of the module.

Nothing here executes SHAP or LIME. The capability layer is a decision layer,
so its tests assert decisions.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from app.explainability import capability as capability_module
from app.explainability.capability import (
    EXPLAINER_KERNEL,
    EXPLAINER_LIME,
    EXPLAINER_LINEAR,
    EXPLAINER_TREE,
    FIDELITY_APPROXIMATE,
    FIDELITY_EXACT,
    FIDELITY_SURROGATE,
    METHOD_LIME,
    METHOD_SHAP,
    SCALE_LOG_ODDS,
    SCALE_PROBABILITY,
    ExplainerCapability,
    detect,
    select_explainer,
)
from app.models.model import MODEL_ID, RF_MODEL_ID, ModelAdapter
from app.models.registry import get_default_registry

SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"


# ---------------------------------------------------------------------------
# Fakes -- estimator families and adapters this project has never seen.
#
# These exist to prove the routing is structural. None of them is a scikit-learn
# estimator, none inherits from anything the capability module imports, and
# none would match any conceivable class allowlist.
# ---------------------------------------------------------------------------


class FakeLinearEstimator:
    """Exposes fitted linear coefficients and nothing else."""

    def __init__(self) -> None:
        self.coef_ = np.array([[0.5, -0.25]])


class FakeTreeEstimator:
    """Exposes a fitted ensemble of sub-estimators and nothing else."""

    def __init__(self) -> None:
        self.estimators_ = [object(), object()]


class FakeBoosterEstimator:
    """Exposes a booster handle -- the other structural shape trees take."""

    def get_booster(self) -> object:
        return object()


class FakeUnknownEstimator:
    """A future model family: no linear coefficients, no tree structure."""


class FakeColumnTransformer:
    """Exposes the fitted attribute a column-wise transformer exposes."""

    def __init__(self) -> None:
        self.transformers_ = []


class FakePipeline:
    """A chained estimator whose steps are named nothing in particular.

    The step names are deliberately NOT 'preprocessor'/'classifier' -- the
    capability layer must locate the transformer and the final estimator
    structurally, not by the names this project happens to use.
    """

    def __init__(self, steps: List[tuple]) -> None:
        self.steps = steps

    @property
    def named_steps(self) -> Dict[str, Any]:
        return dict(self.steps)


class FakeAdapter(ModelAdapter):
    """A ModelAdapter with every capability-relevant input set explicitly."""

    def __init__(
        self,
        *,
        model_id: str = "fake-model",
        model_type: str = "fake_type",
        integration_type: str = "in_process",
        artifact: Any = None,
        artifact_error: Optional[BaseException] = None,
        supports_probability: bool = True,
        background: Optional[pd.DataFrame] = None,
        batch_scoring: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_type = model_type
        self.integration_type = integration_type
        self.feature_names = ["a", "b"]
        self._artifact = artifact
        self._artifact_error = artifact_error
        self._supports_probability = supports_probability
        self._background = background
        self._batch_scoring = batch_scoring

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {
            "predict_proba": self._supports_probability,
            "batch": True,
            "batch_scoring": self._batch_scoring,
            # Deliberately dishonest: the capability layer must ignore this.
            "explainability": True,
        }

    @property
    def supports_probability(self) -> bool:
        return self._supports_probability

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def load_fitted_model(self) -> Any:
        if self._artifact_error is not None:
            raise self._artifact_error
        return self._artifact

    def background_data(self) -> Optional[pd.DataFrame]:
        return self._background


def _background(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"a": range(rows), "b": range(rows)})


def _linear_pipeline() -> FakePipeline:
    return FakePipeline(
        [("step_one", FakeColumnTransformer()), ("step_two", FakeLinearEstimator())]
    )


def _tree_pipeline() -> FakePipeline:
    return FakePipeline(
        [("step_one", FakeColumnTransformer()), ("step_two", FakeTreeEstimator())]
    )


# ---------------------------------------------------------------------------
# Real registry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry():
    return get_default_registry()


@pytest.fixture(scope="module")
def lr_capability(registry) -> ExplainerCapability:
    return detect(registry.get(MODEL_ID))


@pytest.fixture(scope="module")
def rf_capability(registry) -> ExplainerCapability:
    return detect(registry.get(RF_MODEL_ID))


@pytest.fixture(scope="module")
def rest_capability(registry) -> ExplainerCapability:
    return detect(registry.get(SYNTHETIC_BANK_MODEL_ID))


# ===========================================================================
# 1. Real Logistic Regression adapter
# ===========================================================================


def test_lr_adapter_capabilities_are_detected(lr_capability):
    assert lr_capability.has_local_artifact is True
    assert lr_capability.is_pipeline is True
    assert lr_capability.has_linear_structure is True
    assert lr_capability.has_tree_structure is False
    assert lr_capability.supports_probability is True
    assert lr_capability.has_background_data is True
    assert lr_capability.has_raw_transformed_mapping is True


def test_lr_shap_selects_exact_linear_explainer(lr_capability):
    decision = select_explainer(lr_capability, METHOD_SHAP)

    assert decision.available is True
    assert decision.explainer == EXPLAINER_LINEAR
    assert decision.scale == SCALE_LOG_ODDS
    assert decision.fidelity == FIDELITY_EXACT


def test_lr_lime_is_available(lr_capability):
    decision = select_explainer(lr_capability, METHOD_LIME)

    assert decision.available is True
    assert decision.explainer == EXPLAINER_LIME
    assert decision.scale == SCALE_PROBABILITY
    assert decision.fidelity == FIDELITY_SURROGATE


# ===========================================================================
# 2. Real Random Forest adapter
# ===========================================================================


def test_rf_adapter_capabilities_are_detected(rf_capability):
    assert rf_capability.has_local_artifact is True
    assert rf_capability.is_pipeline is True
    assert rf_capability.has_tree_structure is True
    assert rf_capability.has_linear_structure is False
    assert rf_capability.supports_probability is True
    assert rf_capability.has_background_data is True


def test_rf_shap_selects_exact_tree_explainer(rf_capability):
    decision = select_explainer(rf_capability, METHOD_SHAP)

    assert decision.available is True
    assert decision.explainer == EXPLAINER_TREE
    assert decision.scale == SCALE_PROBABILITY
    assert decision.fidelity == FIDELITY_EXACT


def test_rf_shap_scale_differs_from_lr_shap_scale(lr_capability, rf_capability):
    """The contract point Step 5 depends on: 'shap' does not imply one scale.

    Tree contributions are additive to predict_proba; linear contributions are
    additive to the decision function. A single scale label keyed on the
    method name would be a false statement about one of them.
    """
    lr_scale = select_explainer(lr_capability, METHOD_SHAP).scale
    rf_scale = select_explainer(rf_capability, METHOD_SHAP).scale

    assert lr_scale == SCALE_LOG_ODDS
    assert rf_scale == SCALE_PROBABILITY
    assert lr_scale != rf_scale


def test_rf_tree_decision_warns_about_scale_incomparability(rf_capability):
    decision = select_explainer(rf_capability, METHOD_SHAP)

    assert any("probability scale" in limit for limit in decision.limitations)


def test_rf_lime_is_available(rf_capability):
    assert select_explainer(rf_capability, METHOD_LIME).available is True


# ===========================================================================
# 3 & 4. Real REST adapter (synthetic bank), as currently configured
# ===========================================================================


def test_rest_adapter_has_no_local_artifact(rest_capability):
    """No artifact, established by asking -- not inferred from integration_type."""
    assert rest_capability.has_local_artifact is False
    assert rest_capability.is_pipeline is False
    assert rest_capability.has_linear_structure is False
    assert rest_capability.has_tree_structure is False
    assert rest_capability.integration_type == "rest"
    assert any("no local model artifact" in note for note in rest_capability.notes)


def test_rest_adapter_is_never_classified_as_exact_shap(rest_capability):
    """Requirement 4: never claim model-internal SHAP without model internals."""
    decision = select_explainer(rest_capability, METHOD_SHAP)

    assert decision.fidelity != FIDELITY_EXACT
    assert decision.explainer != EXPLAINER_LINEAR
    assert decision.explainer != EXPLAINER_TREE


def test_rest_adapter_with_batch_scoring_is_approximately_explainable(rest_capability):
    """The synthetic bank now implements POST /score-batch, so it qualifies.

    UPDATED (Phase 6): this adapter previously declared no batch scoring and
    both methods were correctly UNAVAILABLE. The endpoint now genuinely
    exists, so the capability layer lets black-box explanation through -- but
    only as approximate/surrogate, never as exact, because nothing about the
    model's internals became inspectable. ``test_rest_adapter_is_never_``
    ``classified_as_exact_shap`` above still guards that.

    ``test_no_artifact_and_no_batch_scoring_is_unavailable`` below keeps the
    without-batching case covered, using a fake adapter so it no longer
    depends on how the real synthetic bank happens to be configured.
    """
    assert rest_capability.supports_batch_scoring is True

    shap_decision = select_explainer(rest_capability, METHOD_SHAP)
    assert shap_decision.available is True
    assert shap_decision.explainer == EXPLAINER_KERNEL
    assert shap_decision.fidelity == FIDELITY_APPROXIMATE
    assert shap_decision.scale == SCALE_PROBABILITY

    lime_decision = select_explainer(rest_capability, METHOD_LIME)
    assert lime_decision.available is True
    assert lime_decision.fidelity == FIDELITY_SURROGATE


def test_no_artifact_and_no_batch_scoring_is_unavailable():
    """Requirement 3, now pinned on a fake so it cannot drift.

    An adapter reachable only one row at a time cannot support perturbation
    methods: thousands of predictions per explained row against a single-row
    interface. The limitation must say so explicitly.
    """
    cap = detect(
        FakeAdapter(
            integration_type="rest",
            artifact_error=NotImplementedError("no local artifact"),
            supports_probability=True,
            background=_background(),
            batch_scoring=False,
        )
    )

    assert cap.has_local_artifact is False
    assert cap.supports_batch_scoring is False
    assert cap.available_methods == ()

    for method in (METHOD_SHAP, METHOD_LIME):
        decision = select_explainer(cap, method)
        assert decision.available is False
        assert decision.explainer is None
        assert decision.scale is None
        assert decision.fidelity is None
        assert any("batch scoring" in limit for limit in decision.limitations)


def test_rest_adapter_reflects_its_actual_contract(rest_capability):
    """Its real contract: probabilities yes, reference data yes, artifact no.

    Both methods are now available, and BOTH are inexact -- the honest
    outcome for a model whose internals were never inspected.
    """
    assert rest_capability.supports_probability is True
    assert rest_capability.has_background_data is True
    assert rest_capability.has_local_artifact is False
    assert set(rest_capability.available_methods) == {METHOD_SHAP, METHOD_LIME}

    for method in rest_capability.available_methods:
        assert select_explainer(rest_capability, method).fidelity != FIDELITY_EXACT


def test_rest_adapter_becomes_approximately_explainable_with_batch_scoring():
    """The one change that unlocks it -- nothing about the model itself."""
    adapter = FakeAdapter(
        integration_type="rest",
        artifact_error=NotImplementedError("no local artifact"),
        supports_probability=True,
        background=_background(),
        batch_scoring=True,
    )

    cap = detect(adapter)
    shap_decision = select_explainer(cap, METHOD_SHAP)
    lime_decision = select_explainer(cap, METHOD_LIME)

    assert shap_decision.available is True
    assert shap_decision.explainer == EXPLAINER_KERNEL
    assert shap_decision.fidelity == FIDELITY_APPROXIMATE
    assert shap_decision.scale == SCALE_PROBABILITY
    assert lime_decision.available is True

    # And it still says plainly that it never saw the model's internals.
    assert any("prediction interface" in limit for limit in shap_decision.limitations)
    assert any("not exact" in limit or "approximation" in limit
               for limit in shap_decision.limitations)


# ===========================================================================
# 5 & 6. Missing fundamental requirements
# ===========================================================================


def test_no_probability_makes_both_methods_unavailable():
    cap = detect(
        FakeAdapter(
            artifact=_linear_pipeline(),
            supports_probability=False,
            background=_background(),
        )
    )

    assert cap.supports_probability is False
    for method in (METHOD_SHAP, METHOD_LIME):
        decision = select_explainer(cap, method)
        assert decision.available is False
        assert any("probability" in limit for limit in decision.limitations)


@pytest.mark.parametrize(
    "background", [None, pd.DataFrame()], ids=["none", "empty"]
)
def test_no_background_data_makes_both_methods_unavailable(background):
    """An empty frame is not a reference distribution and must not count as one."""
    cap = detect(FakeAdapter(artifact=_linear_pipeline(), background=background))

    assert cap.has_background_data is False
    for method in (METHOD_SHAP, METHOD_LIME):
        decision = select_explainer(cap, method)
        assert decision.available is False
        assert any("reference" in limit for limit in decision.limitations)


# ===========================================================================
# 7, 8, 9. Structural detection on estimator families this project never saw
# ===========================================================================


def test_generic_linear_like_estimator_routes_to_exact_linear_shap():
    """Detection requires fitted coefficients, not a particular estimator class."""
    cap = detect(FakeAdapter(artifact=_linear_pipeline(), background=_background()))

    assert cap.has_linear_structure is True
    decision = select_explainer(cap, METHOD_SHAP)
    assert decision.explainer == EXPLAINER_LINEAR
    assert decision.fidelity == FIDELITY_EXACT
    assert decision.scale == SCALE_LOG_ODDS


@pytest.mark.parametrize(
    "estimator",
    [FakeTreeEstimator(), FakeBoosterEstimator()],
    ids=["ensemble_of_sub_estimators", "booster_handle"],
)
def test_generic_tree_like_estimator_routes_to_exact_tree_shap(estimator):
    """Both structural shapes a tree model takes are detected."""
    pipeline = FakePipeline([("t", FakeColumnTransformer()), ("m", estimator)])
    cap = detect(FakeAdapter(artifact=pipeline, background=_background()))

    assert cap.has_tree_structure is True
    decision = select_explainer(cap, METHOD_SHAP)
    assert decision.explainer == EXPLAINER_TREE
    assert decision.fidelity == FIDELITY_EXACT
    assert decision.scale == SCALE_PROBABILITY


def test_unknown_future_estimator_is_approximated_not_rejected():
    """The central requirement: an unrecognised model is still explainable.

    No linear coefficients, no tree structure, no class this module has ever
    heard of -- but it has probabilities and reference data, so it gets an
    approximate explanation rather than a refusal.
    """
    pipeline = FakePipeline(
        [("t", FakeColumnTransformer()), ("m", FakeUnknownEstimator())]
    )
    cap = detect(FakeAdapter(artifact=pipeline, background=_background()))

    assert cap.has_linear_structure is False
    assert cap.has_tree_structure is False

    decision = select_explainer(cap, METHOD_SHAP)
    assert decision.available is True
    assert decision.explainer == EXPLAINER_KERNEL
    assert decision.fidelity == FIDELITY_APPROXIMATE
    assert any("not exact" in limit or "approximation" in limit
               for limit in decision.limitations)

    assert select_explainer(cap, METHOD_LIME).available is True


def test_bare_estimator_without_a_pipeline_is_supported():
    """An artifact need not be a chain; raw space is then the model's space."""
    cap = detect(FakeAdapter(artifact=FakeLinearEstimator(), background=_background()))

    assert cap.is_pipeline is False
    assert cap.has_raw_transformed_mapping is True
    assert select_explainer(cap, METHOD_SHAP).explainer == EXPLAINER_LINEAR


def test_chain_without_a_transformer_falls_back_to_approximate():
    """Conservative: no recoverable feature mapping means no exact claim."""
    pipeline = FakePipeline([("only_step", FakeLinearEstimator())])
    cap = detect(FakeAdapter(artifact=pipeline, background=_background()))

    assert cap.is_pipeline is True
    assert cap.has_raw_transformed_mapping is False
    decision = select_explainer(cap, METHOD_SHAP)
    assert decision.explainer == EXPLAINER_KERNEL
    assert decision.fidelity == FIDELITY_APPROXIMATE


# ===========================================================================
# 10. Batch scoring detection
# ===========================================================================


@pytest.mark.parametrize("declared", [True, False])
def test_batch_scoring_is_read_from_adapter_capabilities(declared):
    cap = detect(
        FakeAdapter(background=_background(), artifact=None, batch_scoring=declared)
    )
    assert cap.supports_batch_scoring is declared


def test_batch_scoring_defaults_to_false_when_undeclared():
    """Absence of a declaration must never be read as support."""

    class NoCapabilitiesAdapter(FakeAdapter):
        @property
        def capabilities(self) -> Dict[str, bool]:
            return {"predict_proba": True, "batch": True}

    cap = detect(
        NoCapabilitiesAdapter(
            artifact_error=NotImplementedError("remote"), background=_background()
        )
    )

    assert cap.supports_batch_scoring is False
    assert select_explainer(cap, METHOD_SHAP).available is False


# ===========================================================================
# 11. No concrete model-class checks anywhere in the implementation
# ===========================================================================


FORBIDDEN_TOKENS = (
    "LogisticRegression",
    "RandomForestClassifier",
    "XGBClassifier",
    "GradientBoosting",
    "DecisionTree",
    "logistic_regression",
    "random_forest",
    "xgboost",
)


def test_capability_module_names_no_concrete_estimator_class():
    """The allowlist this module exists to avoid must not exist in its source."""
    source = inspect.getsource(capability_module)

    for token in FORBIDDEN_TOKENS:
        assert token not in source, (
            f"capability.py references {token!r}. Routing must depend on "
            "observable structure, not on a known model family."
        )


def test_capability_module_imports_no_estimator_classes():
    """Structural probing needs no estimator imports at all."""
    source = inspect.getsource(capability_module)

    assert "from sklearn" not in source
    assert "import sklearn" not in source


def test_routing_ignores_model_type_entirely():
    """Same structure, different model_type strings -> identical decision.

    A direct test that identity is not a routing input: only the declared
    model_type changes between these two adapters.
    """
    kwargs = dict(artifact=_tree_pipeline(), background=_background())

    one = detect(FakeAdapter(model_type="something_familiar", **kwargs))
    two = detect(FakeAdapter(model_type="a_family_invented_tomorrow", **kwargs))

    assert one.model_type != two.model_type
    assert (
        select_explainer(one, METHOD_SHAP).to_dict()
        == select_explainer(two, METHOD_SHAP).to_dict()
    )


def test_self_declared_explainability_flag_is_ignored():
    """Requirement 4: a claim is not an observation.

    FakeAdapter declares capabilities['explainability'] = True unconditionally.
    An adapter with no probabilities must still be unavailable.
    """
    adapter = FakeAdapter(supports_probability=False, background=_background())
    assert adapter.capabilities["explainability"] is True

    assert detect(adapter).available_methods == ()


# ===========================================================================
# 12. Deterministic, structured output
# ===========================================================================


def test_detection_is_deterministic(registry):
    """Same adapter, repeated detection, identical result."""
    for model_id in (MODEL_ID, RF_MODEL_ID, SYNTHETIC_BANK_MODEL_ID):
        adapter = registry.get(model_id)
        assert detect(adapter).to_dict() == detect(adapter).to_dict()


def test_selection_is_deterministic(lr_capability, rf_capability, rest_capability):
    for cap in (lr_capability, rf_capability, rest_capability):
        for method in (METHOD_SHAP, METHOD_LIME):
            assert (
                select_explainer(cap, method).to_dict()
                == select_explainer(cap, method).to_dict()
            )


def test_unavailable_decisions_are_structured_not_none_or_raised():
    """Callers must never have to interpret an exception or a null."""
    cap = detect(FakeAdapter(supports_probability=False, background=_background()))
    decision = select_explainer(cap, METHOD_SHAP)

    assert isinstance(decision.to_dict(), dict)
    assert decision.to_dict() == {
        "method": METHOD_SHAP,
        "available": False,
        "explainer": None,
        "scale": None,
        "fidelity": None,
        "limitations": [decision.limitations[0]],
    }
    assert decision.limitations[0]


def test_unsupported_method_returns_a_decision_rather_than_raising(lr_capability):
    decision = select_explainer(lr_capability, "permutation")

    assert decision.available is False
    assert any("Supported methods" in limit for limit in decision.limitations)


def test_available_methods_agrees_with_select_explainer(
    lr_capability, rf_capability, rest_capability
):
    """The convenience view can never drift from the routing decision."""
    for cap in (lr_capability, rf_capability, rest_capability):
        expected = tuple(
            method
            for method in (METHOD_SHAP, METHOD_LIME)
            if select_explainer(cap, method).available
        )
        assert cap.available_methods == expected


def test_capability_dict_is_json_serializable(lr_capability):
    import json

    json.dumps(lr_capability.to_dict())
    json.dumps(select_explainer(lr_capability, METHOD_SHAP).to_dict())


def test_artifact_load_failure_is_recorded_not_swallowed():
    """A missing artifact must be visible, not silently indistinguishable."""
    cap = detect(
        FakeAdapter(
            artifact_error=FileNotFoundError("artifact.joblib missing"),
            background=_background(),
        )
    )

    assert cap.has_local_artifact is False
    assert any("FileNotFoundError" in note for note in cap.notes)
