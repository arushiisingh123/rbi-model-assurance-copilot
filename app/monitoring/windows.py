"""Monitoring windows: the reference/current pair every monitor compares (owner: Arushi).

WHY A WINDOW TYPE EXISTS
    Every metric in this project already compares a REFERENCE population with a
    CURRENT one -- ``drift_report(reference_data, current_data)`` and
    ``prediction_drift_report(reference_predictions, current_predictions)`` both
    take the pair positionally. Until now each caller assembled that pair
    itself, which is how ``compute_real_drift()`` ended up hard-coding one
    dataset and one 80/20 split.

    A ``MonitoringWindow`` is the smallest object that names one side of the
    comparison and keeps its features, predictions, scores and record identities
    together. Continuous monitoring is then the same comparison over a different
    pair of windows, rather than a different code path.

WHAT THIS IS NOT
    - Not a scheduler, a store, or a telemetry collector. A window holds data it
      was handed; it does not fetch, persist, or poll. The live collector is
      Khushi's (see ``docs/module-interfaces.md``), and this type is the shape it
      can hand over.
    - Not a time series. ``window_id`` is a caller-supplied label, not a parsed
      timestamp. Nothing here interprets, orders, or compares window identifiers,
      because inventing a time semantics the platform does not yet have would be
      a guess about Khushi's collector rather than a contract with it.
    - Not a second model contract. A window is built FROM the standard
      ``predict_batch()`` output; it does not redefine it.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

__all__ = [
    "MonitoringWindow",
    "window_from_model_output",
]


@dataclass(frozen=True)
class MonitoringWindow:
    """One side of a monitoring comparison.

    Frozen: a window is the evidence of what was observed over a period. A
    monitor that could mutate its own inputs could report a measurement that no
    longer matches the data it was given.

    Attributes:
        window_id: Caller-supplied label for this window, e.g. ``"reference"``
            or ``"2026-Q3"``. Carried into results and evidence so a
            measurement can be traced back to the period it describes. Never
            parsed or interpreted.
        features: Raw feature frame for the window, or None. Required for
            feature/data drift; unnecessary for prediction drift.
        predictions: Predicted classes, or None. Required for prediction drift.
        scores: Continuous scores aligned 1:1 with ``predictions`` -- for the
            credit models ``P(class == 1) == P(BAD)`` -- or None when the model
            has no probability capability.
        instance_ids: Stable per-record identity aligned 1:1 with
            ``predictions``, or None. Carried for traceability; no monitoring
            metric is computed from it.
        favorable_label: The prediction value counting as the favourable
            outcome in this window, read from the model contract's
            ``label_semantics``. None when the contract did not state one.
    """

    window_id: str
    features: Optional[pd.DataFrame] = None
    predictions: Optional[List[Any]] = None
    scores: Optional[List[float]] = None
    instance_ids: Optional[List[str]] = None
    favorable_label: Optional[Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id.strip():
            raise ValueError(
                "window_id must be a non-empty string -- a monitoring result "
                "that cannot name the period it describes is not traceable."
            )

        if self.features is not None and not isinstance(self.features, pd.DataFrame):
            raise ValueError(
                f"features must be a pandas DataFrame or None, got "
                f"{type(self.features).__name__}. drift_report() compares "
                "frames, not serialized records."
            )

        # Predictions, scores and instance_ids describe the SAME records in the
        # same order. A length disagreement means the window was assembled
        # wrongly, and measuring different channels over different record sets
        # would report them side by side as one population's behaviour.
        self._require_aligned("scores", self.scores)
        self._require_aligned("instance_ids", self.instance_ids)

    def _require_aligned(self, name: str, values: Optional[List[Any]]) -> None:
        if values is None or self.predictions is None:
            return
        if len(values) != len(self.predictions):
            raise ValueError(
                f"Window '{self.window_id}': {name} has {len(values)} entries "
                f"but predictions has {len(self.predictions)}. They describe "
                "the same records in the same order and must be the same length."
            )

    @property
    def has_features(self) -> bool:
        """Whether feature/data drift can be evaluated from this window."""
        return self.features is not None and not self.features.empty

    @property
    def has_predictions(self) -> bool:
        """Whether prediction/output drift can be evaluated from this window."""
        return self.predictions is not None and len(self.predictions) > 0

    @property
    def has_scores(self) -> bool:
        """Whether the continuous score channel is available from this window."""
        return self.scores is not None and len(self.scores) > 0


def window_from_model_output(
    model_output: Dict[str, Any],
    *,
    window_id: str,
) -> MonitoringWindow:
    """Build a window from one standard ``predict_batch()`` output.

    This is the monitoring layer's single point of contact with the model/data
    contract. It READS the contract's existing keys and never re-derives,
    recalculates, or renames any of them:

        feature_matrix   -> features
        predictions      -> predictions
        probabilities    -> scores
        instance_ids     -> instance_ids
        model_metadata.label_semantics.favorable_outcome_label
                         -> favorable_label

    Model-agnostic by construction: every adapter in the registry returns this
    same dict shape through ``predict_batch(adapter=...)``, so a Logistic
    Regression, Random Forest, or REST-backed model produces a window the same
    way. Nothing here mentions a feature name, a model type, or a dataset.

    ``probabilities`` is absent for a model whose adapter has no probability
    capability; ``scores`` is then None and the score channel is reported as
    unavailable rather than as zero drift.

    Args:
        model_output: A dict shaped like ``app.models.model.predict_batch()``
            output. Only the keys listed above are read; any other key is
            ignored rather than reinterpreted.
        window_id: Label for the window being built.

    Returns:
        A ``MonitoringWindow``.

    Raises:
        ValueError: if ``model_output`` is not a dict, or if the channels it
            carries are not mutually aligned (enforced by ``MonitoringWindow``).
    """
    if not isinstance(model_output, dict):
        raise ValueError(
            "model_output must be a dict shaped like predict_batch() output, "
            f"got {type(model_output).__name__}."
        )

    features = model_output.get("feature_matrix")
    predictions = model_output.get("predictions")
    scores = model_output.get("probabilities")
    instance_ids = model_output.get("instance_ids")

    metadata = model_output.get("model_metadata") or {}
    label_semantics = metadata.get("label_semantics") or {}
    favorable_label = label_semantics.get("favorable_outcome_label")

    return MonitoringWindow(
        window_id=window_id,
        features=features,
        predictions=list(predictions) if predictions is not None else None,
        scores=list(scores) if scores is not None else None,
        instance_ids=list(instance_ids) if instance_ids is not None else None,
        favorable_label=favorable_label,
    )
