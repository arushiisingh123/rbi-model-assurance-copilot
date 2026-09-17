"""Monitoring API client for the dashboard (owner: Arushi).

A monitoring-specific counterpart to ``dashboard.api_client``, kept separate so
the monitoring lane adds no fixture or endpoint to the shared client module.

NO MOCK FALLBACK, DELIBERATELY
    The other panels fall back to a synthetic fixture when the API is
    unreachable. Monitoring does not, and must not: a fabricated PSI or a
    fabricated PASS is precisely the kind of number that must never appear on a
    risk dashboard. When the API cannot be reached this returns an EMPTY
    payload and a source label saying so, and the panel renders an explicit
    "no monitoring result available" state.

    That is the honest failure mode. An empty panel tells the truth; a mock
    monitoring verdict does not.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# Returned when the monitoring API cannot be reached. Deliberately carries no
# result: there is nothing true to show.
UNAVAILABLE_PAYLOAD: Dict[str, Any] = {
    "result": None,
    "evidence": [],
    "protected_attribute": None,
}

SOURCE_API = "API"
SOURCE_UNAVAILABLE = "Monitoring API unreachable (no data shown)"


def get_monitoring(
    model_id: Optional[str] = None,
    protected_attribute: Optional[str] = None,
    timeout: float = 30.0,
) -> Tuple[Dict[str, Any], str]:
    """Fetch a monitoring run from ``GET /monitoring``.

    Args:
        model_id: Registered model to monitor. None uses the API's default.
        protected_attribute: Explicit override. None lets the adapter's own
            declaration decide, which may legitimately be "none declared".
        timeout: Seconds to wait. Higher than the other clients' 2s because a
            monitoring run scores two windows, and a REST-backed model scores
            them over the network.

    Returns:
        ``(payload, source)``. On any failure, ``(UNAVAILABLE_PAYLOAD, ...)``
        with a source label saying the API was unreachable -- never fabricated
        monitoring numbers.
    """
    params: Dict[str, Any] = {}
    if model_id is not None:
        params["model_id"] = model_id
    if protected_attribute is not None:
        params["protected_attribute"] = protected_attribute

    try:
        response = requests.get(
            f"{API_BASE_URL}/monitoring", params=params, timeout=timeout
        )
        response.raise_for_status()
        return response.json(), SOURCE_API
    except (requests.RequestException, ValueError):
        return dict(UNAVAILABLE_PAYLOAD), SOURCE_UNAVAILABLE
