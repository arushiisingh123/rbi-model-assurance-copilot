"""Model registry for credit-scoring models and adapters (Owner: Manas).

Provides a centralized registry for looking up, listing, and health-checking
registered ModelAdapter instances by model_id.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from app.models.model import (
    LogisticRegressionAdapter,
    ModelAdapter,
    RandomForestAdapter,
)
from app.models.rest_adapter import RESTAdapter


class ModelNotFoundError(Exception):
    """Raised when a requested model_id is not registered in the ModelRegistry."""


class ModelRegistry:
    """Registry holding ModelAdapter instances keyed by model_id."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelAdapter] = {}
        self._unavailable: Dict[str, str] = {}

    def register(self, adapter: ModelAdapter) -> None:
        """Register a ModelAdapter instance in the registry."""
        if not isinstance(adapter, ModelAdapter):
            raise TypeError(f"Expected ModelAdapter instance, got {type(adapter)}")
        self._models[adapter.model_id] = adapter
        self._unavailable.pop(adapter.model_id, None)

    def register_guarded(self, model_id: str, factory: Callable[[], ModelAdapter]) -> bool:
        """Register the adapter ``factory`` builds, tolerating its failure.

        One model whose optional dependency is missing must not make the whole
        registry unusable. On failure the model is recorded as unavailable --
        with the original exception text, so the cause is inspectable rather
        than hidden -- and every other model stays registered and usable.

        Returns True when registration succeeded.
        """
        try:
            self.register(factory())
            return True
        except Exception as exc:  # noqa: BLE001 - recorded verbatim, never swallowed
            self._unavailable[model_id] = f"{type(exc).__name__}: {exc}"
            return False

    def get(self, model_id: str) -> ModelAdapter:
        """Retrieve a registered ModelAdapter by model_id.

        Raises:
            ModelNotFoundError: If model_id is not found in the registry. When
                the model is known but could not be constructed, the message
                carries the underlying reason instead of a bare "not found".
        """
        if model_id not in self._models:
            if model_id in self._unavailable:
                raise ModelNotFoundError(
                    f"Model '{model_id}' is registered but unavailable: "
                    f"{self._unavailable[model_id]}"
                )
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        return self._models[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        """Return serialized metadata for all registered models.

        Only successfully-constructed models appear here, so an existing
        consumer never receives a half-built adapter. Models that failed to
        construct are reported by ``unavailable_models()``.
        """
        return [adapter.metadata() for adapter in self._models.values()]

    def unavailable_models(self) -> Dict[str, str]:
        """{model_id: reason} for models that could not be constructed."""
        return dict(self._unavailable)


_default_registry: Optional[ModelRegistry] = None


def _build_synthetic_bank_adapter() -> RESTAdapter:
    """Build the RESTAdapter entry for the synthetic bank reference model.

    Reads SYNTHETIC_BANK_URL fresh on every call (default
    "http://127.0.0.1:8100", matching app.synthetic_bank.service.DEFAULT_PORT)
    so tests can retarget it at a live test instance via
    reset_default_registry(). Constructing a RESTAdapter makes no network
    call -- this is safe to do even if the synthetic bank service isn't
    running; only predict()/predict_proba()/health() touch the network.
    """
    from app.synthetic_bank.data_generator import (
        CATEGORICAL_FEATURES,
        FEATURE_COLUMNS,
        LABEL_SEMANTICS,
        generate_customers,
    )
    from app.synthetic_bank.model import (
        MODEL_ID,
        MODEL_TYPE,
        MODEL_VERSION,
        TRAINED_ON,
    )

    endpoint_url = os.environ.get("SYNTHETIC_BANK_URL", "http://127.0.0.1:8100")
    input_schema = {
        col: {"type": "categorical" if col in CATEGORICAL_FEATURES else "numeric"}
        for col in FEATURE_COLUMNS
    }
    # A modest, deterministic demo/background batch (distinct seed from
    # training) -- this is what predict_batch() falls back to as /model's
    # default demo input for any adapter whose schema differs from German
    # Credit's, since German Credit's own test split cannot serve that role
    # for a model with a completely different feature space.
    background = generate_customers(n=50, random_state=123)[FEATURE_COLUMNS].reset_index(drop=True)
    return RESTAdapter(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        endpoint_url=endpoint_url,
        feature_names=FEATURE_COLUMNS,
        input_schema=input_schema,
        # batch_scoring is True because app/synthetic_bank/service.py genuinely
        # implements POST /score-batch and RESTAdapter genuinely uses it -- NOT
        # because the adapter can accept a multi-row DataFrame (that is the
        # weaker, pre-existing "batch" flag, which is satisfied by a per-row
        # loop). app/explainability/capability.py gates black-box explanation
        # on batch_scoring, so declaring it without the endpoint would license
        # thousands of HTTP requests per explained row.
        capabilities={
            "predict_proba": True,
            "batch": True,
            "batch_scoring": True,
            "explainability": False,
        },
        background=background,
        # The bank model's OWN provenance and label semantics, declared by the
        # module that trained it rather than inherited from the in-process
        # German Credit default. Without these the platform reports an XGBoost
        # model trained on generated bank data as though it were trained on the
        # German Credit CSV -- a false statement about model provenance, and
        # exactly the kind of metadata a model-risk reviewer relies on.
        trained_on=TRAINED_ON,
        label_semantics=LABEL_SEMANTICS,
    )


def get_default_registry() -> ModelRegistry:
    """Return the lazily-populated default ModelRegistry singleton.

    Pre-populated with default LogisticRegressionAdapter and
    RandomForestAdapter instances, plus a RESTAdapter for the synthetic bank
    reference model (app.synthetic_bank).

    DEGRADED REGISTRATION (not silent): the synthetic bank entry needs
    ``app.synthetic_bank``, which imports xgboost. That is an optional
    dependency for the rest of the platform, and it used to be imported
    eagerly here -- so a missing xgboost made the WHOLE registry
    unconstructible, taking the German Credit models down with it even though
    they have no such dependency. A construction failure for one model now
    leaves every other model usable and is recorded in
    ``registry.unavailable_models()`` with the original error preserved, so
    the cause stays visible instead of being swallowed.
    """
    global _default_registry
    if _default_registry is None:
        registry = ModelRegistry()
        registry.register(LogisticRegressionAdapter.load_default())
        registry.register(RandomForestAdapter.load_default())
        registry.register_guarded(
            "synthetic-bank-credit-v1", _build_synthetic_bank_adapter
        )
        _default_registry = registry
    return _default_registry


def reset_default_registry() -> None:
    """Test-only helper: clears the cached default registry singleton.

    The next get_default_registry() call rebuilds it from current state
    (e.g. after changing the SYNTHETIC_BANK_URL environment variable in a
    test). Rebuilding is cheap: LogisticRegressionAdapter/RandomForestAdapter
    load from an already-trained on-disk joblib artifact rather than
    retraining, and RESTAdapter construction makes no network call.
    """
    global _default_registry
    _default_registry = None
