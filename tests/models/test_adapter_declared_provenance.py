"""Adapter-declared training provenance and label semantics.

An externally served model is trained on its own data and has its own label
polarity. Before these declarations existed, ``predict_batch()`` stamped the
in-process German Credit defaults onto EVERY model's metadata -- so the
synthetic bank's XGBoost model reported that it had been trained on
``data/german_credit/german_credit.csv``.

Two distinct harms, which is why both fields are covered here:

``trained_on``      a false statement about model provenance. Model-risk
                    review turns on knowing what a model was trained on;
                    reporting another dataset is worse than reporting nothing.

``label_semantics`` fairness reads ``favorable_outcome_label`` from it. A model
                    whose favourable class is the opposite of the default would
                    have every fairness verdict silently inverted.

Both are declarations, not inferences: an adapter that declares nothing keeps
the in-process default exactly as before.
"""
import pandas as pd
import pytest

from app.models.model import (
    MODEL_ID,
    RF_MODEL_ID,
    LABEL_SEMANTICS,
    LogisticRegressionAdapter,
    RandomForestAdapter,
    predict_batch,
)
from app.models.preprocessing import DEFAULT_DATASET_PATH
from app.models.rest_adapter import RESTAdapter

SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"


@pytest.fixture(scope="module")
def bank_adapter():
    """The registered synthetic-bank adapter.

    Taken from the registry rather than constructed here: the point is what the
    PLATFORM declares for that model, not what a test can build.
    """
    from app.models.registry import get_default_registry

    return get_default_registry().get(SYNTHETIC_BANK_MODEL_ID)


# ---------------------------------------------------------------------------
# The defect this closes
# ---------------------------------------------------------------------------


def test_the_bank_adapter_does_not_claim_german_credit_as_its_training_source(
    bank_adapter,
):
    """The regression. An XGBoost model trained on generated bank data must
    never report the German Credit CSV as its provenance."""
    assert bank_adapter.trained_on is not None
    assert "german_credit" not in bank_adapter.trained_on
    assert bank_adapter.trained_on != DEFAULT_DATASET_PATH


def test_the_bank_adapter_declares_the_provenance_its_own_module_defines(
    bank_adapter,
):
    """Not an invented string: the value comes from the module that trained it."""
    from app.synthetic_bank.model import TRAINED_ON

    assert bank_adapter.trained_on == TRAINED_ON


def test_the_bank_adapter_declares_its_own_label_semantics(bank_adapter):
    from app.synthetic_bank.data_generator import LABEL_SEMANTICS as BANK_SEMANTICS

    assert bank_adapter.label_semantics == BANK_SEMANTICS
    # Its own statement, not the German Credit object.
    assert bank_adapter.label_semantics is not LABEL_SEMANTICS


# ---------------------------------------------------------------------------
# German Credit models are unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter_factory, expected_id",
    [
        (LogisticRegressionAdapter.load_default, MODEL_ID),
        (RandomForestAdapter.load_default, RF_MODEL_ID),
    ],
)
def test_german_credit_models_keep_their_existing_provenance(
    adapter_factory, expected_id
):
    adapter = adapter_factory()
    output = predict_batch(adapter=adapter)

    assert adapter.model_id == expected_id
    assert output["model_metadata"]["trained_on"] == DEFAULT_DATASET_PATH
    assert output["model_metadata"]["label_semantics"] == LABEL_SEMANTICS


def test_the_default_no_adapter_path_is_unchanged():
    """The pre-existing default path must be byte-identical to before."""
    metadata = predict_batch()["model_metadata"]

    assert metadata["trained_on"] == DEFAULT_DATASET_PATH
    assert metadata["label_semantics"] == LABEL_SEMANTICS


# ---------------------------------------------------------------------------
# Declaration, never inference
# ---------------------------------------------------------------------------


def _bare_rest_adapter(**kwargs) -> RESTAdapter:
    return RESTAdapter(
        model_id="undeclared-model",
        model_version="1.0.0",
        model_type="whatever",
        endpoint_url="http://127.0.0.1:1",
        feature_names=["a"],
        input_schema={"a": {"type": "numeric"}},
        capabilities={"predict_proba": True, "batch": True},
        **kwargs,
    )


def test_an_adapter_that_declares_nothing_keeps_the_in_process_defaults():
    """Absence of a declaration is not a licence to invent one, but it must
    also not break the existing default behaviour."""
    adapter = _bare_rest_adapter()

    assert adapter.trained_on is None
    assert adapter.label_semantics is None


def test_a_declared_none_is_treated_as_not_declared_not_as_a_value(monkeypatch):
    """An adapter defining the attribute as None must fall back, not report None.

    Otherwise ``model_metadata.trained_on`` becomes null and a consumer reading
    it sees an empty provenance rather than the real default.
    """
    adapter = _bare_rest_adapter()
    frame = pd.DataFrame({"a": [1.0, 2.0]})

    monkeypatch.setattr(adapter, "predict", lambda X: [0] * len(X))
    monkeypatch.setattr(adapter, "predict_proba", lambda X: [0.1] * len(X))

    metadata = predict_batch(feature_matrix=frame, adapter=adapter)["model_metadata"]

    assert metadata["trained_on"] == DEFAULT_DATASET_PATH
    assert metadata["label_semantics"] == LABEL_SEMANTICS


def test_a_declared_value_reaches_model_metadata(monkeypatch):
    declared_semantics = {
        "0": "DECLINED",
        "1": "APPROVED",
        "positive_class": 1,
        "favorable_outcome_label": 1,
    }
    adapter = _bare_rest_adapter(
        trained_on="vendor-supplied-training-extract",
        label_semantics=declared_semantics,
    )
    frame = pd.DataFrame({"a": [1.0, 2.0]})

    monkeypatch.setattr(adapter, "predict", lambda X: [0] * len(X))
    monkeypatch.setattr(adapter, "predict_proba", lambda X: [0.1] * len(X))

    metadata = predict_batch(feature_matrix=frame, adapter=adapter)["model_metadata"]

    assert metadata["trained_on"] == "vendor-supplied-training-extract"
    # The inverted polarity survives -- this is the case that would silently
    # flip every fairness verdict if the default were used instead.
    assert metadata["label_semantics"]["favorable_outcome_label"] == 1
