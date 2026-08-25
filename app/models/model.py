"""Phase 0 stub for the credit-scoring model (owner: Namitha).

Real training/evaluation logic is Phase 1 work. These functions only
return correctly-shaped fake data so other modules can be built and
tested against the agreed interface (see docs/module-interfaces.md).
"""

MODEL_METADATA = {
    "model_type": "stub",
    "version": "0.0.0-phase0",
    "trained_on": "data/sample/credit_sample.csv",
    "feature_names": ["income", "age", "credit_history_len"],
}


def train(dataset_path: str = "data/sample/credit_sample.csv") -> dict:
    """Stub: pretend to train on the given dataset path.

    Phase 1 will replace this with real preprocessing + training.
    """
    return {"status": "stub_trained", "dataset_path": dataset_path}


def predict_batch(feature_matrix=None) -> dict:
    """Stub: return fake predictions in the agreed output shape.

    Real prediction logic (Phase 1) will use a trained model instead
    of hardcoded values.
    """
    return {
        "predictions": [0, 1, 0],
        "probabilities": [0.12, 0.81, 0.33],
        "feature_matrix": feature_matrix,
        "model_metadata": MODEL_METADATA,
        "is_mock": True,
    }


def save(path: str = "data/sample/model_stub.bin") -> dict:
    """Stub: pretend to save the model to disk."""
    return {"status": "stub_saved", "path": path}


def load(path: str = "data/sample/model_stub.bin") -> dict:
    """Stub: pretend to load a model from disk."""
    return {"status": "stub_loaded", "path": path}
