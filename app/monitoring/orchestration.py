"""Monitoring orchestration: model adapter -> windows -> result -> evidence (owner: Arushi).

WHAT THIS IS
    The single reachable entry point for the monitoring lane. It turns a
    registered ``ModelAdapter`` into the two windows ``monitor_run()`` compares,
    runs the three analytical channels, and returns the result together with its
    evidence records.

        adapter -> predict_batch() -> MonitoringWindow x2
                -> monitor_run() -> monitoring_evidence()

    Everything it reports is produced by a module that already owns that
    calculation. This layer computes no metric, defines no threshold, and
    introduces no second status vocabulary -- exactly the rule
    ``monitor.py`` already follows.

WHERE THE TWO WINDOWS COME FROM, AND WHY IT IS NOT A HIDDEN DATASET
    A monitor compares a REFERENCE population with a CURRENT one, and this
    module never invents either:

        reference   the caller's frame, or ``adapter.background_data()`` --
                    the reference population THAT MODEL declares as its own.
        current     the caller's frame, or the adapter's own default batch
                    from ``predict_batch(adapter=...)``.

    No dataset path, feature name, protected attribute, or label value is named
    anywhere in this module. A model whose adapter declares no background data
    is REFUSED rather than quietly compared against somebody else's dataset --
    the failure mode that made
    ``app.api.orchestration.compute_real_drift()`` silently evaluate German
    Credit's ``age`` against an unrelated model's ``age``.

    Note the honest consequence: for an adapter whose default batch IS its
    background data, the two windows are the same population and drift is
    legitimately zero. That is a true measurement of two identical windows, and
    it is reported as such rather than dressed up. Genuinely shifted windows
    come from the caller -- see ``tests/monitoring/test_monitoring_synthetic_bank.py``,
    which drives the bank's own scenario generator through this path.

PROVENANCE IS NEVER GUESSED
    ``provenance`` is whatever the caller states, and defaults to ``None`` --
    meaning "not stated". It is never defaulted to ``"observed"``: this module
    cannot know whether the frame it was handed is real traffic, and guessing
    the one value that matters is how synthetic scenario data ends up presented
    as observed drift.

WHAT THIS IS NOT
    Not a collector, a scheduler, or a store. It fetches nothing, persists
    nothing, and polls nothing. Not a report producer: it imports nothing from
    ``app.report``, ``app.rag``, or any LLM, reaches no compliance conclusion,
    and writes no narrative. Not an API layer -- it imports nothing from
    ``app.api``, so ``context`` arrives as a plain mapping built by the caller.
"""

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from app.models.model import ProbabilityCapabilityUnavailable, predict_batch
from app.monitoring.evidence import monitoring_evidence
from app.monitoring.monitor import monitor_run
from app.monitoring.windows import MonitoringWindow, window_from_model_output

__all__ = [
    "DEFAULT_CURRENT_WINDOW_ID",
    "DEFAULT_REFERENCE_WINDOW_ID",
    "build_monitoring_window",
    "resolve_protected_attribute",
    "run_monitoring",
]

# Neutral labels used when the caller does not name its windows. Deliberately
# generic: a window label is a caller's word for a period, and inventing a
# dataset-flavoured default here would name something this module cannot know.
DEFAULT_REFERENCE_WINDOW_ID = "reference"
DEFAULT_CURRENT_WINDOW_ID = "current"


def _score_frame(adapter: Any, frame: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Score ``frame`` through the adapter, returning predict_batch() output.

    ``frame=None`` delegates entirely to ``predict_batch``'s own default batch
    for that adapter, so this module never decides what a model's default input
    is.

    A label-only model is handled rather than refused. ``predict_batch()``
    calls ``predict_proba()`` unconditionally, so an adapter whose
    ``supports_probability`` is False raises out of it (a known upstream
    limitation -- see docs/monitoring-handoffs.md). Rather than let the whole
    monitoring run fail for a model that can still be monitored on its label
    channel, the probability-free path builds the same contract shape from
    ``adapter.predict()`` and simply omits ``probabilities``.

    Nothing is fabricated: the score channel is ABSENT, which
    ``prediction_drift_report()`` reports as ``unavailable_no_scores`` rather
    than as zero drift.
    """
    if getattr(adapter, "supports_probability", True):
        try:
            return predict_batch(feature_matrix=frame, adapter=adapter)
        except ProbabilityCapabilityUnavailable:
            # The adapter's declared capability disagreed with its behaviour;
            # fall through to the label-only path rather than failing the run.
            pass

    if frame is None:
        frame = adapter.background_data()
        if frame is None or len(frame) == 0:
            raise ValueError(
                f"Adapter '{getattr(adapter, 'model_id', '<unknown>')}' has no "
                "probability capability and no background data, so no default "
                "batch could be built. Supply the window's features explicitly."
            )

    scored = frame.reset_index(drop=True)[list(adapter.feature_names)]
    predictions = [int(p) for p in adapter.predict(scored)]

    # Shaped exactly like predict_batch() output MINUS "probabilities", which
    # window_from_model_output() already treats as an absent score channel.
    return {
        "predictions": predictions,
        "instance_ids": [f"row-{i:04d}" for i in range(len(scored))],
        "feature_matrix": scored,
        "model_metadata": {
            "model_type": getattr(adapter, "model_type", None),
            "version": getattr(adapter, "model_version", None),
            "feature_names": list(scored.columns),
        },
        "is_mock": False,
    }


def build_monitoring_window(
    adapter: Any,
    *,
    window_id: str,
    features: Optional[pd.DataFrame] = None,
    provenance: Optional[str] = None,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
) -> MonitoringWindow:
    """Score ``features`` through ``adapter`` and wrap the result as a window.

    Args:
        adapter: A registered ``ModelAdapter``. Its own ``feature_names``,
            ``predict``/``predict_proba`` and ``background_data`` are the only
            model-specific things consulted.
        window_id: Caller's label for the period. Never parsed.
        features: The window's raw feature frame. ``None`` uses the adapter's
            own default batch.
        provenance: One of ``WINDOW_PROVENANCE_VALUES``, or None for
            "not stated". Never inferred.
        window_start / window_end: Optional monitoring-window boundaries.

    Returns:
        A ``MonitoringWindow``.

    Raises:
        ValueError: if the window's channels are misaligned, or the provenance
            or bounds are invalid (all enforced by ``MonitoringWindow``).
    """
    model_output = _score_frame(adapter, features)
    return window_from_model_output(
        model_output,
        window_id=window_id,
        provenance=provenance,
        window_start=window_start,
        window_end=window_end,
    )


def resolve_protected_attribute(
    adapter: Any,
    override: Optional[str] = None,
) -> Optional[str]:
    """The attribute fairness should be evaluated over, or None.

    Resolution order, and there is no third step:

        1. ``override`` -- an explicit caller instruction. Which attribute is
           protected is a regulatory and governance decision, so a caller that
           states one is obeyed.
        2. ``adapter.protected_attribute`` -- the model's own DECLARATION.
           ``None`` means "not declared", which is a real answer, not a gap.

    There is deliberately no inference step: no scanning of column names, no
    dtype heuristic, no falling back to another model's attribute. An
    undeclared attribute yields ``None``, and ``monitor_run()`` then reports
    the fairness channel as PENDING -- the same outcome
    ``app.api.orchestration.compute_real_fairness()`` produces through its
    ``none_declared`` sentinel.
    """
    if override is not None:
        return override
    return getattr(adapter, "protected_attribute", None)


def run_monitoring(
    adapter: Any,
    *,
    context: Mapping[str, Any],
    reference_features: Optional[pd.DataFrame] = None,
    current_features: Optional[pd.DataFrame] = None,
    reference_window_id: str = DEFAULT_REFERENCE_WINDOW_ID,
    current_window_id: str = DEFAULT_CURRENT_WINDOW_ID,
    protected_attribute: Optional[str] = None,
    reference_provenance: Optional[str] = None,
    current_provenance: Optional[str] = None,
    reference_window_start: Optional[datetime] = None,
    reference_window_end: Optional[datetime] = None,
    current_window_start: Optional[datetime] = None,
    current_window_end: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Run every monitoring channel for one model and return result + evidence.

    Args:
        adapter: The ``ModelAdapter`` being monitored.
        context: ``AssuranceRunContext``-shaped mapping carrying non-empty
            ``model_id``, ``model_version`` and ``assurance_run_id``. Threaded
            through unchanged; this module mints no identity of its own.
        reference_features: Baseline frame. ``None`` uses
            ``adapter.background_data()`` -- the model's OWN declared reference
            population, never another model's dataset.
        current_features: Monitored frame. ``None`` uses the adapter's default
            batch.
        reference_window_id / current_window_id: Labels for the two periods.
        protected_attribute: Explicit override. Omitted, the adapter's own
            declaration is used; see ``resolve_protected_attribute``.
        reference_provenance / current_provenance: Stated by the caller, or
            None. Never guessed.
        reference_window_start / ...end, current_window_start / ...end:
            Optional monitoring-window boundaries.

    Returns:
        Dictionary with keys:
            result (dict): the full ``monitor_run()`` output, unmodified.
            evidence (list of dict): ``monitoring_evidence(result)`` records,
                one per measured channel plus a ``monitoring_summary``.
            protected_attribute (str or None): the attribute actually used, so
                a reader can see whether fairness was evaluated and on what.
                None means none was declared or supplied -- and the fairness
                channel is correspondingly PENDING.

    Raises:
        ValueError: if the adapter declares no background data and no
            ``reference_features`` were supplied, if a window's channels are
            misaligned, or if ``context`` lacks required identity fields.
    """
    if reference_features is None:
        reference_features = adapter.background_data()
        if reference_features is None or len(reference_features) == 0:
            raise ValueError(
                f"Adapter '{getattr(adapter, 'model_id', '<unknown>')}' "
                "declares no background data, so no reference window could be "
                "built. Supply reference_features explicitly -- monitoring "
                "will not substitute another model's dataset as a baseline."
            )

    reference = build_monitoring_window(
        adapter,
        window_id=reference_window_id,
        features=reference_features,
        provenance=reference_provenance,
        window_start=reference_window_start,
        window_end=reference_window_end,
    )
    current = build_monitoring_window(
        adapter,
        window_id=current_window_id,
        features=current_features,
        provenance=current_provenance,
        window_start=current_window_start,
        window_end=current_window_end,
    )

    resolved_attribute = resolve_protected_attribute(adapter, protected_attribute)

    result = monitor_run(
        reference,
        current,
        context=context,
        protected_attribute=resolved_attribute,
    )
    evidence: List[Dict[str, Any]] = monitoring_evidence(result)

    return {
        "result": result,
        "evidence": evidence,
        "protected_attribute": resolved_attribute,
    }
