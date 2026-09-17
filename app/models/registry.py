"""Model registry for credit-scoring models and adapters (Owner: Manas).

Provides a centralized registry for looking up, listing, and health-checking
registered ModelAdapter instances by model_id.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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

    def register(self, adapter: ModelAdapter) -> None:
        """Register a ModelAdapter instance in the registry."""
        if not isinstance(adapter, ModelAdapter):
            raise TypeError(f"Expected ModelAdapter instance, got {type(adapter)}")
        self._models[adapter.model_id] = adapter

    def get(self, model_id: str) -> ModelAdapter:
        """Retrieve a registered ModelAdapter by model_id.

        Raises:
            ModelNotFoundError: If model_id is not found in the registry.
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        return self._models[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        """Return serialized metadata for all registered models."""
        return [adapter.metadata() for adapter in self._models.values()]


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
        generate_customers,
    )
    from app.synthetic_bank.model import MODEL_ID, MODEL_TYPE, MODEL_VERSION

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
        capabilities={"predict_proba": True, "batch": True, "explainability": False},
        background=background,
    )


def get_default_registry() -> ModelRegistry:
    """Return the lazily-populated default ModelRegistry singleton.

    Pre-populated with default LogisticRegressionAdapter and
    RandomForestAdapter instances, plus a RESTAdapter for the synthetic bank
    reference model (app.synthetic_bank).
    """
    global _default_registry
    if _default_registry is None:
        registry = ModelRegistry()
        registry.register(LogisticRegressionAdapter.load_default())
        registry.register(RandomForestAdapter.load_default())
        registry.register(_build_synthetic_bank_adapter())
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
