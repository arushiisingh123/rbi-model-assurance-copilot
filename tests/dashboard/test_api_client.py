"""Unit tests for dashboard api_client helper (Owner: Khushi)."""
import requests

from app.api.mock_data import (
    MOCK_ASSURANCE_RESULT,
    MOCK_COMPLIANCE_RESULT,
    MOCK_DRIFT_RESULT,
    MOCK_EXPLAINABILITY_RESULT_LIME,
    MOCK_EXPLAINABILITY_RESULT_SHAP,
    MOCK_FAIRNESS_RESULT,
    MOCK_MODEL_RESULT,
    MOCK_REPORT_RESULT,
)
from dashboard.api_client import (
    get_assurance_result,
    get_compliance,
    get_drift_comparison,
    get_explainability,
    get_fairness_drift,
    get_health,
    get_model,
    get_report,
    render_status,
)


class DummyResponse:
    """Mock requests response for unit testing."""

    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def test_get_health_success(monkeypatch):
    test_data = {"status": "ok"}
    monkeypatch.setattr(requests, "get", lambda url, timeout=2: DummyResponse(test_data))
    data, source = get_health()
    assert source == "api"
    assert data == test_data


def test_get_health_fallback(monkeypatch):
    def mock_get(url, timeout=2):
        raise requests.exceptions.ConnectionError("Backend unreachable")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_health()
    assert source == "fallback"
    assert data == {"status": "unreachable"}


def test_get_model_success(monkeypatch):
    test_data = {"test": "model_data"}
    monkeypatch.setattr(requests, "get", lambda url, timeout=2: DummyResponse(test_data))
    data, source = get_model()
    assert source == "api"
    assert data == test_data


def test_get_model_fallback(monkeypatch):
    def mock_get(url, timeout=2):
        raise requests.exceptions.ConnectionError("Backend unreachable")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_model()
    assert source == "fallback"
    assert data == MOCK_MODEL_RESULT
    assert bool(data)


def test_get_explainability_shap_success(monkeypatch):
    captured_urls = []

    def mock_get(url, timeout=2):
        captured_urls.append(url)
        return DummyResponse({"method": "shap"})

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_explainability(method="shap")
    assert source == "api"
    assert data["method"] == "shap"
    assert "method=shap" in captured_urls[0]


def test_get_explainability_lime_success(monkeypatch):
    captured_urls = []

    def mock_get(url, timeout=2):
        captured_urls.append(url)
        return DummyResponse({"method": "lime"})

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_explainability(method="lime")
    assert source == "api"
    assert data["method"] == "lime"
    assert "method=lime" in captured_urls[0]


def test_get_explainability_lime_fallback(monkeypatch):
    def mock_get(url, timeout=2):
        raise requests.exceptions.Timeout("Timeout")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_explainability(method="lime")
    assert source == "fallback"
    assert data == MOCK_EXPLAINABILITY_RESULT_LIME
    assert data["method"] == "lime"


def test_get_explainability_shap_fallback(monkeypatch):
    def mock_get(url, timeout=2):
        raise requests.exceptions.ConnectionError("Connection error")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_explainability(method="shap")
    assert source == "fallback"
    assert data == MOCK_EXPLAINABILITY_RESULT_SHAP
    assert data["method"] == "shap"


def test_get_fairness_drift_success(monkeypatch):
    test_data = {"fairness": {}, "drift": {}}
    monkeypatch.setattr(requests, "get", lambda url, timeout=2: DummyResponse(test_data))
    data, source = get_fairness_drift()
    assert source == "api"
    assert data == test_data


def test_get_fairness_drift_fallback(monkeypatch):
    def mock_get(url, timeout=2):
        raise requests.exceptions.RequestException("Request error")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_fairness_drift()
    assert source == "fallback"
    assert data["fairness"] == MOCK_FAIRNESS_RESULT
    assert data["drift"] == MOCK_DRIFT_RESULT
    assert bool(data)


def test_get_compliance_success(monkeypatch):
    test_data = {"findings": []}
    monkeypatch.setattr(requests, "get", lambda url, timeout=2: DummyResponse(test_data))
    data, source = get_compliance()
    assert source == "api"
    assert data == test_data


def test_get_compliance_fallback(monkeypatch):
    def mock_get(url, timeout=2):
        raise requests.exceptions.ConnectionError("Failed")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_compliance()
    assert source == "fallback"
    assert data == MOCK_COMPLIANCE_RESULT
    assert bool(data)


def test_get_assurance_result_success(monkeypatch):
    test_data = {"assurance": "ok"}
    monkeypatch.setattr(requests, "get", lambda url, timeout=2: DummyResponse(test_data))
    data, source = get_assurance_result()
    assert source == "api"
    assert data == test_data


def test_get_assurance_result_fallback(monkeypatch):
    def mock_get(url, timeout=2):
        raise requests.exceptions.ConnectionError("Failed")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_assurance_result()
    assert source == "fallback"
    assert data == MOCK_ASSURANCE_RESULT
    assert bool(data)


def test_get_report_success(monkeypatch):
    test_data = {"test": "report_data"}
    monkeypatch.setattr(requests, "get", lambda url, timeout=2: DummyResponse(test_data))
    data, source = get_report()
    assert source == "api"
    assert data == test_data


def test_get_report_fallback(monkeypatch):
    def mock_get(url, timeout=2):
        raise requests.exceptions.ConnectionError("Failed")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_report()
    assert source == "fallback"
    assert data == MOCK_REPORT_RESULT
    assert bool(data)


def test_render_status():
    # Standard statuses return non-empty strings with labels and emojis
    assert "PASS" in render_status("PASS")
    assert "WARNING" in render_status("WARNING")
    assert "FAIL" in render_status("FAIL")
    assert "PENDING" in render_status("PENDING")

    # Case insensitivity
    assert "PASS" in render_status("pass")
    assert "WARNING" in render_status("warning")

    # Unknown value returns without raising
    res = render_status("UNKNOWN_STATUS")
    assert res == "❔ UNKNOWN_STATUS"

    # None and non-string inputs handled safely without raising
    none_res = render_status(None)
    assert none_res == "❔ None"

    int_res = render_status(123)
    assert int_res == "❔ 123"


def test_get_drift_comparison_success(monkeypatch):
    test_data = {"comparability": "COMPARABLE", "drift_a": {}, "drift_b": {}}
    monkeypatch.setattr(requests, "get", lambda url, timeout=5: DummyResponse(test_data))
    data, source = get_drift_comparison()
    assert source == "api"
    assert data == test_data


def test_get_drift_comparison_fallback_unavailable(monkeypatch):
    def mock_get(url, timeout=5):
        raise requests.exceptions.ConnectionError("Backend unreachable")

    monkeypatch.setattr(requests, "get", mock_get)
    data, source = get_drift_comparison()
    assert source == "unavailable"
    assert data is None

