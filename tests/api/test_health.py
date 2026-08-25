"""Phase 0 smoke tests for the FastAPI skeleton (owner: Khushi)."""
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_mock_assurance_result_endpoint():
    response = client.get("/mock-assurance-result")
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["is_mock"] is True
    assert "note" in body
