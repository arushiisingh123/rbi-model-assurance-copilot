"""REST Adapter for external credit-scoring models served over HTTP (Owner: Manas).

Provides a ModelAdapter implementation that delegates prediction and health
checks to a remote REST endpoint following the scoring contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from app.models.model import ModelAdapter, ProbabilityCapabilityUnavailable


class RESTAdapterError(Exception):
    """Raised when a REST-backed model's endpoint is unreachable, times out,
    or returns an unexpected/invalid response during predict/predict_proba.
    Health checks do NOT raise this -- see health() below."""


class RESTAdapter(ModelAdapter):
    """ModelAdapter communicating with an external model service via REST HTTP calls."""

    integration_type = "rest"

    def __init__(
        self,
        model_id: str,
        model_version: str,
        model_type: str,
        endpoint_url: str,
        feature_names: List[str],
        input_schema: Dict[str, Dict[str, str]],
        capabilities: Dict[str, bool],
        timeout: float = 10.0,
        background: Optional[pd.DataFrame] = None,
    ):
        self.model_id = model_id
        self.model_version = model_version
        self.model_type = model_type
        self.endpoint_url = endpoint_url.rstrip("/")
        self.feature_names = list(feature_names)
        self._input_schema = dict(input_schema)
        self._capabilities = dict(capabilities)
        self.timeout = timeout
        self._background = background

    @property
    def supports_probability(self) -> bool:
        """Whether the remote model supports probability predictions."""
        return self._capabilities.get("predict_proba", False)

    @property
    def input_schema(self) -> Dict[str, Dict[str, str]]:
        """Input feature schema provided explicitly at construction."""
        return self._input_schema

    @property
    def capabilities(self) -> Dict[str, bool]:
        """Capability flags provided explicitly at construction."""
        return self._capabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions (0 = GOOD, 1 = BAD) for X by scoring each row via HTTP."""
        records = X.to_dict("records")
        predictions: List[int] = []

        for idx, row in enumerate(records):
            try:
                resp = requests.post(
                    f"{self.endpoint_url}/score",
                    json=row,
                    timeout=self.timeout,
                )
                if not (200 <= resp.status_code < 300):
                    raise RESTAdapterError(
                        f"Scoring failed for row {idx} with HTTP status {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
            except requests.RequestException as exc:
                raise RESTAdapterError(
                    f"HTTP request error scoring row {idx} at {self.endpoint_url}/score: {exc}"
                ) from exc
            except (ValueError, TypeError) as exc:
                if isinstance(exc, RESTAdapterError):
                    raise
                raise RESTAdapterError(
                    f"Invalid JSON response scoring row {idx} at {self.endpoint_url}/score: {exc}"
                ) from exc

            if not isinstance(data, dict) or "prediction" not in data or data["prediction"] is None:
                raise RESTAdapterError(
                    f"Response for row {idx} missing 'prediction' field: {data}"
                )

            try:
                predictions.append(int(data["prediction"]))
            except (ValueError, TypeError) as exc:
                raise RESTAdapterError(
                    f"Invalid prediction value '{data.get('prediction')}' for row {idx}: {exc}"
                ) from exc

        return np.array(predictions, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(class == 1) == P(BAD) as a 1-D array, one value per row via HTTP."""
        if not self.supports_probability:
            raise ProbabilityCapabilityUnavailable(
                f"Model '{self.model_id}' does not support probability predictions."
            )

        records = X.to_dict("records")
        probabilities: List[float] = []

        for idx, row in enumerate(records):
            try:
                resp = requests.post(
                    f"{self.endpoint_url}/score",
                    json=row,
                    timeout=self.timeout,
                )
                if not (200 <= resp.status_code < 300):
                    raise RESTAdapterError(
                        f"Scoring failed for row {idx} with HTTP status {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
            except requests.RequestException as exc:
                raise RESTAdapterError(
                    f"HTTP request error scoring row {idx} at {self.endpoint_url}/score: {exc}"
                ) from exc
            except (ValueError, TypeError) as exc:
                if isinstance(exc, RESTAdapterError):
                    raise
                raise RESTAdapterError(
                    f"Invalid JSON response scoring row {idx} at {self.endpoint_url}/score: {exc}"
                ) from exc

            if not isinstance(data, dict) or "probability" not in data or data["probability"] is None:
                raise RESTAdapterError(
                    f"Response for row {idx} missing 'probability' field: {data}"
                )

            try:
                probabilities.append(float(data["probability"]))
            except (ValueError, TypeError) as exc:
                raise RESTAdapterError(
                    f"Invalid probability value '{data.get('probability')}' for row {idx}: {exc}"
                ) from exc

        return np.array(probabilities, dtype=float)

    def health(self) -> dict:
        """Query remote liveness/readiness endpoint. Never raises exceptions."""
        try:
            resp = requests.get(f"{self.endpoint_url}/health", timeout=self.timeout)
            if 200 <= resp.status_code < 300:
                data = resp.json()
                if isinstance(data, dict):
                    return data
                return {"status": "ok", "raw": data}
            return {
                "status": "unreachable",
                "error": f"HTTP status {resp.status_code}: {resp.text}",
            }
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)}

    def load_fitted_model(self) -> Any:
        """Raise NotImplementedError as REST models have no local model artifact."""
        raise NotImplementedError(
            "RESTAdapter has no local fitted model artifact; "
            "it delegates prediction to a remote HTTP endpoint."
        )

    def background_data(self) -> Optional[pd.DataFrame]:
        """Return reference/background data if provided at initialization."""
        return self._background
