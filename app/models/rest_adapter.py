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
        protected_attribute: Optional[str] = None,
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
        # None (the default) means "not declared" -- orchestration must treat
        # fairness as not applicable/PENDING for this adapter, never guess a
        # protected attribute or fall back to another model's.
        self.protected_attribute = protected_attribute

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

    @property
    def supports_batch_scoring(self) -> bool:
        """Whether the remote service exposes POST /score-batch.

        Read from the declared ``batch_scoring`` capability, defaulting to
        False. The default matters: this adapter's fallback path issues ONE
        HTTP request PER ROW, so an unbatched adapter scored by a
        perturbation explainer would emit thousands of requests per explained
        row. ``app/explainability/capability.py`` reads this same flag to
        refuse black-box explanation in that case.

        Note the deliberate distinction from the pre-existing ``batch`` flag,
        which every adapter declares True and which only means "accepts a
        multi-row DataFrame" -- true here even though it is served by a
        per-row loop. ``batch_scoring`` is the stronger claim: one CALL for
        the whole batch. Only declare it where /score-batch genuinely exists.
        """
        return bool(self._capabilities.get("batch_scoring", False))

    def _score_batch(self, X: pd.DataFrame) -> Dict[str, List[Any]]:
        """One POST to /score-batch for the whole frame, order preserved.

        Returns the parsed ``{"predictions": [...], "probabilities": [...]}``
        payload. Both lists are verified to have exactly one entry per input
        row before being returned -- a short or overlong response would
        silently shift every subsequent applicant's score by one position,
        so it is refused rather than truncated or padded.
        """
        records = X.to_dict("records")
        try:
            resp = requests.post(
                f"{self.endpoint_url}/score-batch",
                json={"instances": records},
                timeout=self.timeout,
            )
            if not (200 <= resp.status_code < 300):
                raise RESTAdapterError(
                    f"Batch scoring failed for {len(records)} row(s) with HTTP "
                    f"status {resp.status_code}: {resp.text}"
                )
            data = resp.json()
        except requests.RequestException as exc:
            raise RESTAdapterError(
                f"HTTP request error batch-scoring {len(records)} row(s) at "
                f"{self.endpoint_url}/score-batch: {exc}"
            ) from exc
        except (ValueError, TypeError) as exc:
            if isinstance(exc, RESTAdapterError):
                raise
            raise RESTAdapterError(
                f"Invalid JSON response batch-scoring at "
                f"{self.endpoint_url}/score-batch: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise RESTAdapterError(
                f"Batch response must be a JSON object, got {type(data).__name__}: {data}"
            )

        result: Dict[str, List[Any]] = {}
        for key in ("predictions", "probabilities"):
            values = data.get(key)
            if values is None:
                raise RESTAdapterError(
                    f"Batch response missing '{key}' field: {data}"
                )
            if not isinstance(values, list):
                raise RESTAdapterError(
                    f"Batch response '{key}' must be a list, got "
                    f"{type(values).__name__}."
                )
            if len(values) != len(records):
                raise RESTAdapterError(
                    f"Batch response '{key}' has {len(values)} entries for "
                    f"{len(records)} input row(s). Refusing a misaligned "
                    "batch: results are positional, so a length mismatch "
                    "would attribute one row's score to another."
                )
            result[key] = values
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions (0 = GOOD, 1 = BAD) for X.

        Uses one POST /score-batch call when the adapter declares
        ``batch_scoring``; otherwise falls back to the original one-HTTP-call-
        per-row loop. Both paths return the same values in the same order.
        """
        if self.supports_batch_scoring:
            values = self._score_batch(X)["predictions"]
            try:
                return np.array([int(v) for v in values], dtype=int)
            except (ValueError, TypeError) as exc:
                raise RESTAdapterError(
                    f"Invalid prediction value in batch response: {exc}"
                ) from exc

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
        """Return P(class == 1) == P(BAD) as a 1-D array, one value per row.

        Uses one POST /score-batch call when the adapter declares
        ``batch_scoring``; otherwise falls back to the original one-HTTP-call-
        per-row loop. This is the function perturbation-based explainers call
        thousands of times per explained row, so the batch path is what makes
        KernelSHAP/LIME feasible against a remote model at all.
        """
        if not self.supports_probability:
            raise ProbabilityCapabilityUnavailable(
                f"Model '{self.model_id}' does not support probability predictions."
            )

        if self.supports_batch_scoring:
            values = self._score_batch(X)["probabilities"]
            try:
                return np.array([float(v) for v in values], dtype=float)
            except (ValueError, TypeError) as exc:
                raise RESTAdapterError(
                    f"Invalid probability value in batch response: {exc}"
                ) from exc

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
