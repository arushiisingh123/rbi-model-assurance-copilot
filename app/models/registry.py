"""Model registry for credit-scoring models and adapters (Owner: Manas).

Provides a centralized registry for looking up, listing, and health-checking
registered ModelAdapter instances by model_id.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.model import (
    LogisticRegressionAdapter,
    ModelAdapter,
    RandomForestAdapter,
)


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


def get_default_registry() -> ModelRegistry:
    """Return the lazily-populated default ModelRegistry singleton.

    Pre-populated with default LogisticRegressionAdapter and RandomForestAdapter
    instances.
    """
    global _default_registry
    if _default_registry is None:
        registry = ModelRegistry()
        registry.register(LogisticRegressionAdapter.load_default())
        registry.register(RandomForestAdapter.load_default())
        _default_registry = registry
    return _default_registry
