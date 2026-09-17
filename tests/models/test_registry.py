"""Unit tests for Model Registry and generalized ModelAdapter contract (owner: Manas).

Verifies:
- ModelAdapter's generalized contract (integration_type, input_schema, capabilities, health(), metadata())
- ModelRegistry registration, lookup, error handling, and listing
- Default registry lazy singleton initialization with LR and RF adapters
"""
import pytest

from app.models import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_ID,
    MODEL_TYPE,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    RF_MODEL_ID,
    RF_MODEL_TYPE,
    RF_MODEL_VERSION,
    LogisticRegressionAdapter,
    ModelAdapter,
    ModelNotFoundError,
    ModelRegistry,
    RandomForestAdapter,
    get_default_registry,
)


# =====================================================================
# 1. ModelAdapter Contract Generalization Tests
# =====================================================================


@pytest.fixture(scope="module")
def lr_adapter():
    return LogisticRegressionAdapter.load_default()


@pytest.fixture(scope="module")
def rf_adapter():
    return RandomForestAdapter.load_default()


def test_lr_adapter_generalized_contract(lr_adapter):
    # Base attributes
    assert lr_adapter.model_id == MODEL_ID
    assert lr_adapter.model_version == MODEL_VERSION
    assert lr_adapter.model_type == MODEL_TYPE
    assert lr_adapter.integration_type == "in_process"

    # Input schema
    schema = lr_adapter.input_schema
    assert isinstance(schema, dict)
    assert len(schema) == len(FEATURE_COLUMNS) == 20
    for col in CATEGORICAL_FEATURES:
        assert schema[col] == {"type": "categorical"}
    for col in NUMERIC_FEATURES:
        assert schema[col] == {"type": "numeric"}

    # Capabilities
    caps = lr_adapter.capabilities
    assert caps == {
        "predict_proba": True,
        "batch": True,
        "explainability": True,
    }

    # Health
    health = lr_adapter.health()
    assert health == {"status": "ok"}

    # Metadata
    meta = lr_adapter.metadata()
    assert meta == {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "integration_type": "in_process",
        "capabilities": {
            "predict_proba": True,
            "batch": True,
            "explainability": True,
        },
    }


def test_rf_adapter_generalized_contract(rf_adapter):
    # Base attributes
    assert rf_adapter.model_id == RF_MODEL_ID
    assert rf_adapter.model_version == RF_MODEL_VERSION
    assert rf_adapter.model_type == RF_MODEL_TYPE
    assert rf_adapter.integration_type == "in_process"

    # Input schema
    schema = rf_adapter.input_schema
    assert isinstance(schema, dict)
    assert len(schema) == len(FEATURE_COLUMNS) == 20
    for col in CATEGORICAL_FEATURES:
        assert schema[col] == {"type": "categorical"}
    for col in NUMERIC_FEATURES:
        assert schema[col] == {"type": "numeric"}

    # Capabilities
    caps = rf_adapter.capabilities
    assert caps == {
        "predict_proba": True,
        "batch": True,
        "explainability": True,
    }

    # Health
    health = rf_adapter.health()
    assert health == {"status": "ok"}

    # Metadata
    meta = rf_adapter.metadata()
    assert meta == {
        "model_id": RF_MODEL_ID,
        "model_version": RF_MODEL_VERSION,
        "model_type": RF_MODEL_TYPE,
        "integration_type": "in_process",
        "capabilities": {
            "predict_proba": True,
            "batch": True,
            "explainability": True,
        },
    }


# =====================================================================
# 2. ModelRegistry Unit Tests
# =====================================================================


def test_registry_register_and_get_roundtrip(lr_adapter, rf_adapter):
    registry = ModelRegistry()
    registry.register(lr_adapter)
    registry.register(rf_adapter)

    assert registry.get(MODEL_ID) is lr_adapter
    assert registry.get(RF_MODEL_ID) is rf_adapter


def test_registry_get_unknown_model_raises_model_not_found():
    registry = ModelRegistry()
    with pytest.raises(ModelNotFoundError) as exc_info:
        registry.get("non-existent-model")
    assert "non-existent-model" in str(exc_info.value)


def test_registry_register_invalid_type_raises_type_error():
    registry = ModelRegistry()
    with pytest.raises(TypeError):
        registry.register("not-an-adapter")  # type: ignore


def test_registry_list_models(lr_adapter, rf_adapter):
    registry = ModelRegistry()
    assert registry.list_models() == []

    registry.register(lr_adapter)
    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["model_id"] == MODEL_ID

    registry.register(rf_adapter)
    models = registry.list_models()
    assert len(models) == 2
    model_ids = {m["model_id"] for m in models}
    assert model_ids == {MODEL_ID, RF_MODEL_ID}


# =====================================================================
# 3. Default Registry Tests
# =====================================================================


def test_default_registry_contains_both_default_models():
    default_reg = get_default_registry()
    models = default_reg.list_models()

    assert len(models) == 2
    registered_ids = {m["model_id"] for m in models}
    assert registered_ids == {
        "german-credit-logistic-regression",
        "german-credit-random-forest",
    }

    lr = default_reg.get("german-credit-logistic-regression")
    assert isinstance(lr, LogisticRegressionAdapter)

    rf = default_reg.get("german-credit-random-forest")
    assert isinstance(rf, RandomForestAdapter)


def test_default_registry_is_singleton():
    reg1 = get_default_registry()
    reg2 = get_default_registry()
    assert reg1 is reg2
