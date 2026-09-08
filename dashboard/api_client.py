"""API Client helper for the Streamlit dashboard (Owner: Khushi).

Handles HTTP requests to the FastAPI backend with automatic graceful fallbacks
to synthetic mock fixtures when the backend service is unreachable.
"""
import os
from typing import Any
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

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def render_status(status: Any) -> str:
    """Map status string to a short labeled string with an emoji.

    Safely handles None, non-string, or unknown values without raising exceptions.
    """
    if status is None:
        return "❔ None"
    status_str = str(status).strip()
    status_upper = status_str.upper()
    if status_upper == "PASS":
        return "✅ PASS"
    elif status_upper == "WARNING":
        return "⚠️ WARNING"
    elif status_upper == "FAIL":
        return "❌ FAIL"
    elif status_upper == "PENDING":
        return "⏳ PENDING"
    else:
        return f"❔ {status_str}"


def get_model() -> tuple[dict, str]:
    """Fetch credit model data from API or fallback."""
    url = f"{API_BASE_URL}/model"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return response.json(), "api"
    except requests.exceptions.RequestException:
        return MOCK_MODEL_RESULT, "fallback"


def get_explainability(method: str = "shap") -> tuple[dict, str]:
    """Fetch explainability findings (SHAP or LIME) from API or fallback."""
    url = f"{API_BASE_URL}/explainability?method={method}"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return response.json(), "api"
    except requests.exceptions.RequestException:
        if str(method).lower() == "lime":
            return MOCK_EXPLAINABILITY_RESULT_LIME, "fallback"
        return MOCK_EXPLAINABILITY_RESULT_SHAP, "fallback"


def get_fairness_drift() -> tuple[dict, str]:
    """Fetch fairness and drift evaluation results from API or fallback."""
    url = f"{API_BASE_URL}/fairness-drift"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return response.json(), "api"
    except requests.exceptions.RequestException:
        fallback = {
            "fairness": MOCK_FAIRNESS_RESULT,
            "drift": MOCK_DRIFT_RESULT,
            "note": (
                "SYNTHETIC DRIFT SCENARIO: Fallback mock scenario for interface validation. "
                "Production assurance evaluates dev training split vs held-out test split."
            ),
        }
        return fallback, "fallback"


def get_compliance() -> tuple[dict, str]:
    """Fetch RBI compliance findings from API or fallback."""
    url = f"{API_BASE_URL}/compliance"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return response.json(), "api"
    except requests.exceptions.RequestException:
        return MOCK_COMPLIANCE_RESULT, "fallback"


def get_assurance_result() -> tuple[dict, str]:
    """Fetch aggregated assurance results across all modules from API or fallback."""
    url = f"{API_BASE_URL}/assurance-result"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return response.json(), "api"
    except requests.exceptions.RequestException:
        return MOCK_ASSURANCE_RESULT, "fallback"


def get_report() -> tuple[dict, str]:
    """Fetch compliance assurance report from API or fallback."""
    url = f"{API_BASE_URL}/report"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return response.json(), "api"
    except requests.exceptions.RequestException:
        return MOCK_REPORT_RESULT, "fallback"
