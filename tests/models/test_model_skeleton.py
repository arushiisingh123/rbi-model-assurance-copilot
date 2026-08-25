"""Phase 0 smoke tests for app.models: import works, stubs return the
agreed shape. No real ML logic exists yet, so we only check structure.
"""
from app.models.model import train, predict_batch, save, load


def test_train_returns_status():
    result = train()
    assert "status" in result


def test_predict_batch_shape():
    result = predict_batch()
    assert "predictions" in result
    assert "probabilities" in result
    assert "model_metadata" in result
    assert result["is_mock"] is True


def test_save_and_load_stubs():
    saved = save()
    loaded = load()
    assert saved["status"] == "stub_saved"
    assert loaded["status"] == "stub_loaded"
