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
    - Not a time series. ``window_id`` is a caller-supplied label and is still
      never parsed, ordered, or compared. The optional ``window_start`` /
      ``window_end`` bounds below are the ONLY time semantics here, and they
      order nothing on their own -- ``monitor_run()`` does not sort, select, or
      schedule by them.
    - Not a second model contract. A window is built FROM the standard
      ``predict_batch()`` output; it does not redefine it.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

__all__ = [
    "WINDOW_PROVENANCE_VALUES",
    "MonitoringWindow",
    "window_from_model_output",
]

# The provenance vocabulary a monitoring window may declare.
#
# These are exactly the three values the project already uses for analytical
# provenance (``TechnicalFinding.provenance`` in ``app/api/schemas.py``); no
# fourth value and no monitoring-specific taxonomy is introduced. They are
# declared here rather than imported so this module stays free of ``app.api``
# -- the same rule, for the same reason, as ``REQUIRED_CONTEXT_FIELDS`` in
# ``monitor.py``. A test asserts this tuple against the real schema so the two
# cannot drift apart.
#
#   observed           real traffic / a real scored population
#   mock               fabricated placeholder data
#   synthetic_fixture  deterministically generated scenario data, e.g.
#                      ``app.drift.scenario`` or
#                      ``app.synthetic_bank.data_generator``
#
# Deliberately OPTIONAL and unvalidated-when-absent: ``None`` means "the caller
# did not state where this data came from", which is honest. Defaulting it to
# ``observed`` would let synthetic scenario data be reported as real observed
# drift -- exactly the claim every generator docstring in this project forbids.
WINDOW_PROVENANCE_VALUES = ("observed", "mock", "synthetic_fixture")


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
        provenance: Where this window's data came from -- one of
            ``WINDOW_PROVENANCE_VALUES`` -- or None when the caller did not
            state it. ``is_mock=False`` on a monitoring result says only that
            the arithmetic is real; this is the field that says whether the
            DATA was observed. None is not a synonym for ``"observed"``.
        window_start: Optional inclusive start of the period this window's
            records describe. See ``window_end`` for the exact semantics.
        window_end: Optional inclusive end of that period.

            SEMANTICS, stated once: these are **monitoring-window boundaries**
            -- the period the records in this window were drawn from. They are
            NOT collection time (when a collector happened to fetch the rows)
            and NOT scoring time (when the model was run). Those are different
            questions, and a window that conflated them would misdate its own
            finding.

            Both are optional and independent: a caller may supply neither,
            either, or both. When BOTH are supplied they must satisfy
            ``window_start <= window_end``, and must agree about timezone
            awareness -- comparing a naive datetime with an aware one is a
            ``TypeError`` in Python, so it is refused here with an explanation
            instead. Timezone-aware UTC is preferred, matching the only other
            timestamp in this project (``app/report/generate.py`` uses
            ``datetime.now(timezone.utc)``), but naive datetimes are accepted
            so a caller with naive local timestamps is not blocked.

            Nothing in this package orders, sorts, selects, or schedules by
            these values. They exist so a finding can state the period it
            describes; a scheduler is explicitly not in scope.
    """

    window_id: str
    features: Optional[pd.DataFrame] = None
    predictions: Optional[List[Any]] = None
    scores: Optional[List[float]] = None
    instance_ids: Optional[List[str]] = None
    favorable_label: Optional[Any] = None
    provenance: Optional[str] = None
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None

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

        # Optional metadata, validated last so the data-channel errors above
        # surface first -- a misassembled window is the more urgent problem.
        self._validate_provenance()
        self._validate_time_bounds()

    def _require_aligned(self, name: str, values: Optional[List[Any]]) -> None:
        if values is None or self.predictions is None:
            return
        if len(values) != len(self.predictions):
            raise ValueError(
                f"Window '{self.window_id}': {name} has {len(values)} entries "
                f"but predictions has {len(self.predictions)}. They describe "
                "the same records in the same order and must be the same length."
            )

    def _validate_provenance(self) -> None:
        """Refuse a provenance value outside the project's existing vocabulary.

        Validated rather than free-text because the set is small, finite, and
        already established elsewhere in the project. An unrecognised value
        would be carried into evidence and read as though it meant something.
        """
        if self.provenance is None:
            return
        if self.provenance not in WINDOW_PROVENANCE_VALUES:
            raise ValueError(
                f"Window '{self.window_id}': unknown provenance "
                f"{self.provenance!r}. Expected one of "
                f"{list(WINDOW_PROVENANCE_VALUES)}, or None to leave it "
                "unstated."
            )

    def _validate_time_bounds(self) -> None:
        """Check the optional window bounds, when supplied.

        Each bound is independently optional. Only when both are present is
        there anything to compare -- and then a reversed pair is refused,
        because a window whose end precedes its start does not describe a
        period at all.
        """
        for name, value in (
            ("window_start", self.window_start),
            ("window_end", self.window_end),
        ):
            if value is not None and not isinstance(value, datetime):
                raise ValueError(
                    f"Window '{self.window_id}': {name} must be a "
                    f"datetime or None, got {type(value).__name__}."
                )

        if self.window_start is None or self.window_end is None:
            return

        # Mixing naive and aware datetimes raises TypeError on comparison.
        # Refusing it here turns an opaque crash into a statement of the
        # actual problem.
        start_aware = self.window_start.tzinfo is not None
        end_aware = self.window_end.tzinfo is not None
        if start_aware != end_aware:
            raise ValueError(
                f"Window '{self.window_id}': window_start and window_end must "
                "both be timezone-aware or both be naive; got "
                f"start={'aware' if start_aware else 'naive'}, "
                f"end={'aware' if end_aware else 'naive'}. Timezone-aware UTC "
                "is preferred."
            )

        if self.window_start > self.window_end:
            raise ValueError(
                f"Window '{self.window_id}': window_start "
                f"({self.window_start.isoformat()}) is after window_end "
                f"({self.window_end.isoformat()}). A window's end cannot "
                "precede its start."
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
    provenance: Optional[str] = None,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
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
        provenance: Optional; passed straight through to the window. It is a
            PARAMETER rather than something read from ``model_output``
            deliberately -- the model contract carries no provenance field,
            and only the caller knows whether the frame it scored was real
            traffic or generated scenario data. Inferring it here would be a
            guess, and the guess that matters most (``"observed"``) is the one
            that must never be made.
        window_start: Optional; passed straight through.
        window_end: Optional; passed straight through. Same reasoning as
            ``provenance`` -- ``predict_batch()`` output carries no timestamps.

    Returns:
        A ``MonitoringWindow``.

    Raises:
        ValueError: if ``model_output`` is not a dict, if the channels it
            carries are not mutually aligned, if ``provenance`` is not in
            ``WINDOW_PROVENANCE_VALUES``, or if the window bounds are invalid
            (all enforced by ``MonitoringWindow``).
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
        provenance=provenance,
        window_start=window_start,
        window_end=window_end,
    )
