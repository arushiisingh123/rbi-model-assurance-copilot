"""Prediction / output drift (owner: Arushi).

WHAT THIS IS, AND HOW IT DIFFERS FROM FEATURE DRIFT
    ``app.drift.drift.drift_report()`` answers "has the INPUT population
    changed?". It compares two feature frames, takes no model, and is
    model-independent by construction: two models handed the same rows produce
    the same feature drift, and that identical result is correct rather than a
    defect.

    This module answers a different question: "has what the MODEL OUTPUTS
    changed?". Its inputs are the model's own predictions and scores, so its
    result IS model-specific -- two models scoring one identical feature matrix
    can and do report different prediction drift. The two questions are
    reported through two separate functions and are never merged into one
    number.

TWO OUTPUT CHANNELS, TWO DIFFERENT METRICS -- AND WHY
    A model emits two kinds of output, and the SAME PSI formula is not valid
    for both.

    1. LABEL CHANNEL -- discrete predicted classes (0 = GOOD, 1 = BAD).
       Measured with a CATEGORICAL PSI: the class frequencies are compared
       directly, one term per observed class. No binning is involved.

    2. SCORE CHANNEL -- the continuous probability ``P(class == 1) == P(BAD)``.
       Measured with the EXISTING quantile PSI and KS statistic imported
       unchanged from ``app.drift.drift``. Those helpers are correct for
       continuous data and are reused rather than reimplemented.

    WHY THE LABEL CHANNEL MAY NOT USE THE QUANTILE PSI
        ``_compute_feature_psi`` derives 10 quantile edges from the reference,
        collapses duplicates with ``np.unique``, then widens the outer edges to
        -inf/+inf. On two-valued data the collapsed edges are frequently just
        ``[0, 1]``, and after widening that is ``[-inf, +inf]`` -- a SINGLE bin
        holding all of the mass. PSI is then identically 0.0 no matter how far
        the base rate moved.

        Measured on the real models, reference = train split, current = test
        split:

            LR  ref BAD rate 0.2137 -> cur 0.2400 : quantile PSI 0.000000
            RF  ref BAD rate 0.2350 -> cur 0.1450 : quantile PSI 0.000000

        The Random Forest's BAD-flag rate fell by 38% relative and the quantile
        PSI reported 0.0, which ``classify_psi`` reads as PASS. The categorical
        PSI reports 0.0039 and 0.0535 for the same two cases.

        The collapse is not a smooth loss of sensitivity that could be tolerated
        with a caveat -- it is arbitrary. Holding a large shift fixed and moving
        only the reference base rate, a 0.15 reference rate collapses to 0.0
        while 0.10 and 0.20 do not. Whether a real shift is seen at all depends
        on where the base rate happens to fall relative to the decile grid.

        Where the binning does happen to survive, the quantile PSI equals the
        categorical PSI exactly. So the categorical form is not a competing
        definition of the metric; it is the same measurement computed in a way
        that cannot silently degenerate.

        ``tests/drift/test_prediction_drift.py`` pins this as a regression.

    KS IS REPORTED FOR THE SCORE CHANNEL ONLY
        KS is a statistic on a cumulative distribution. On a two-point support
        it reduces to the absolute difference between the two base rates, which
        ``per_class`` already reports exactly and far more legibly. Reporting it
        as a "KS statistic" on labels would dress a base-rate difference up as a
        distributional test, so the label channel reports no KS.

THRESHOLDS
    None are defined here. Status comes from ``classify_psi`` in
    ``app.config.thresholds``, the same single authoritative source feature
    drift uses. Applying it to a score distribution is in fact PSI's original
    credit-scoring use -- population stability of a score -- and applying it to
    class frequencies is standard categorical PSI usage. Reusing one convention
    is deliberate: a second set of drift bands would be a second source of
    truth. As with every threshold in this project, these are engineering
    conventions and NOT RBI regulatory requirements (docs/thresholds.md).

API-LAYER INDEPENDENCE
    This module imports nothing from ``app.api``, matching
    ``app.drift.comparability``. It takes plain sequences and returns a plain
    dict, so ``app/drift/`` acquires no dependency on the API layer.

Note on is_mock:
    ``is_mock=False`` means the arithmetic is real. It says nothing about where
    the predictions came from. Predictions scored over a dataset built by
    ``app.drift.scenario.build_drift_scenario`` are SYNTHETIC and must never be
    presented as observed drift in a real lending population.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.thresholds import STATUS_PENDING, classify_psi, worst_status

# Reused from the feature-drift module rather than reimplemented, so the two
# drift surfaces can never disagree about how a continuous distribution is
# compared, how non-finite values are dropped, or how many decimals are
# reported. The score channel is continuous data, which is exactly what these
# helpers are correct for.
from app.drift.drift import (
    EPSILON,
    _ROUNDING_DP,
    _compute_feature_ks,
    _compute_feature_psi,
    _finite_values,
)

__all__ = [
    "SCORE_AVAILABILITY_COMPUTED",
    "SCORE_AVAILABILITY_UNAVAILABLE",
    "prediction_drift_report",
]

# Whether a score-channel measurement exists at all. Mirrors the
# ``roc_auc`` / ``roc_auc_status`` pairing already used for
# ``ModelMetrics`` in app/api/schemas.py: the value is Optional and a
# sibling field says why it is absent, so a missing measurement is never
# confused with a measured zero.
SCORE_AVAILABILITY_COMPUTED = "computed"
SCORE_AVAILABILITY_UNAVAILABLE = "unavailable_no_scores"


def _as_series(values: Any, argument_name: str) -> pd.Series:
    """Return ``values`` as a 1-D Series, positionally indexed, nulls intact.

    Nulls are deliberately KEPT here. Alignment between a window's predictions
    and its scores is checked on the raw record count, before either channel is
    cleaned: dropping nulls first would shorten one channel and make a
    correctly-aligned pair look mismatched.

    Accepts a list, numpy array, or pandas Series. Unlike the fairness module
    there is no positional-pairing hazard between the reference and current
    windows -- they are two independent populations and may legitimately differ
    in length -- so a pandas index is simply discarded here rather than
    rejected.
    """
    if values is None:
        raise ValueError(f"{argument_name} must not be None.")

    if isinstance(values, pd.DataFrame):
        raise ValueError(
            f"{argument_name} must be a 1-D sequence of model outputs, not a "
            "DataFrame. Prediction drift compares one output column per window; "
            "pass e.g. model_output['predictions']."
        )

    if isinstance(values, pd.Series):
        series = values.reset_index(drop=True)
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim > 1:
            raise ValueError(
                f"{argument_name} must be 1-dimensional, got shape {array.shape}."
            )
        series = pd.Series(array)

    return series


def _to_native(value: Any) -> Any:
    """Unwrap a numpy scalar to its plain Python equivalent.

    ``np.int64(1)`` compares equal to ``1`` but is not JSON-serializable and
    renders as ``np.int64(1)``. Class labels travel into evidence records and
    API responses, so they are reported as native Python values -- the same
    reason ``drift_report()`` casts every metric through ``float()``.
    """
    return value.item() if isinstance(value, np.generic) else value


def _observed_classes(reference: np.ndarray, current: np.ndarray) -> List[Any]:
    """Classes observed in either window, in first-observed order.

    Reference order first, then any class current introduces. Deliberately not
    sorted: label values need not be numeric or mutually comparable for a
    model-agnostic output channel, and first-observed order is the convention
    ``app.fairness`` already uses for its group ordering.
    """
    classes: List[Any] = []
    for value in list(reference) + list(current):
        native = _to_native(value)
        if native not in classes:
            classes.append(native)
    return classes


def _compute_label_psi(
    reference: np.ndarray,
    current: np.ndarray,
    classes: List[Any],
) -> float:
    """Categorical PSI over ``classes`` -- class frequencies, no binning.

    ``sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct))`` with one term per
    observed class. This is the standard PSI definition for a discrete
    characteristic; the only adaptation is the same ``EPSILON`` substitution for
    zero-frequency classes that the feature-drift PSI already applies, so a
    class absent from one window cannot produce ``log(0)``.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    ref_pct = np.array([np.mean(reference == c) for c in classes], dtype=float)
    cur_pct = np.array([np.mean(current == c) for c in classes], dtype=float)

    ref_pct = np.where(ref_pct == 0, EPSILON, ref_pct)
    cur_pct = np.where(cur_pct == 0, EPSILON, cur_pct)

    psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi_val)


def _per_class_rates(
    reference: np.ndarray,
    current: np.ndarray,
    classes: List[Any],
) -> List[Dict[str, Any]]:
    """Reference and current frequency of each observed class.

    Informational detail behind ``label_psi``, aligned index-for-index with
    ``classes_evaluated`` and rounded to the same precision. It carries no
    status and no threshold of its own -- exactly the role ``per_feature``
    plays for feature drift.
    """
    return [
        {
            "class": cls,
            "reference_rate": round(float(np.mean(reference == cls)), _ROUNDING_DP),
            "current_rate": round(float(np.mean(current == cls)), _ROUNDING_DP),
        }
        for cls in classes
    ]


def _unavailable_score_channel() -> Dict[str, Any]:
    """Score-channel fields when no usable score distribution exists.

    ``None`` rather than ``0.0``: a model with no probability capability has not
    been measured as stable, it has not been measured at all. A zero here would
    read as a PASS-worthy result for a measurement that never happened.
    """
    return {
        "score_psi": None,
        "score_ks_statistic": None,
        "score_status": STATUS_PENDING,
        "score_availability": SCORE_AVAILABILITY_UNAVAILABLE,
    }


def _score_channel(
    reference_scores: Any,
    current_scores: Any,
    *,
    reference_n: int,
    current_n: int,
) -> Dict[str, Any]:
    """Continuous PSI and KS over the model's score distributions.

    Returns the unavailable channel when either side is absent or leaves no
    finite values after cleaning.
    """
    if reference_scores is None or current_scores is None:
        return _unavailable_score_channel()

    ref_raw = _as_series(reference_scores, "reference_scores")
    cur_raw = _as_series(current_scores, "current_scores")

    # Scores and predictions describe the same records in the same window, so a
    # length disagreement means the caller assembled the window wrongly. Better
    # to refuse than to measure two channels over different record sets and
    # report them side by side as one model's behaviour.
    #
    # Compared on RAW record counts, before either channel drops nulls: a NaN
    # score for a real record is a missing measurement, not a missing record,
    # and cleaning first would make an aligned pair look mismatched.
    if len(ref_raw) != reference_n:
        raise ValueError(
            f"reference_scores has {len(ref_raw)} value(s) but "
            f"reference_predictions has {reference_n}. Scores and predictions "
            "describe the same records in the same monitoring window and must "
            "be the same length."
        )
    if len(cur_raw) != current_n:
        raise ValueError(
            f"current_scores has {len(cur_raw)} value(s) but "
            f"current_predictions has {current_n}. Scores and predictions "
            "describe the same records in the same monitoring window and must "
            "be the same length."
        )

    ref_vals = _finite_values(ref_raw)
    cur_vals = _finite_values(cur_raw)
    if len(ref_vals) == 0 or len(cur_vals) == 0:
        return _unavailable_score_channel()

    score_psi = round(_compute_feature_psi(ref_vals, cur_vals), _ROUNDING_DP)
    score_ks = round(_compute_feature_ks(ref_vals, cur_vals), _ROUNDING_DP)

    return {
        "score_psi": score_psi,
        "score_ks_statistic": score_ks,
        # Classify the REPORTED (rounded) value so the number shown and the
        # status attached to it can never disagree -- the same rule
        # drift_report() follows at the 0.10 boundary.
        "score_status": classify_psi(score_psi),
        "score_availability": SCORE_AVAILABILITY_COMPUTED,
    }


def _pending_result() -> Dict[str, Any]:
    """Neutral result when no prediction distribution could be compared.

    Absent or unusable output is not evidence of drift, so the status is PENDING
    rather than FAIL -- the same rule ``drift_report()`` applies to absent
    feature data.
    """
    return {
        "label_psi": 0.0,
        "label_status": STATUS_PENDING,
        "classes_evaluated": [],
        "per_class": [],
        **_unavailable_score_channel(),
        "status": STATUS_PENDING,
        "is_mock": False,
    }


def prediction_drift_report(
    reference_predictions: Any = None,
    current_predictions: Any = None,
    *,
    reference_scores: Optional[Any] = None,
    current_scores: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate drift in a model's OWN OUTPUT between two monitoring windows.

    This is prediction/output drift, not feature/data drift. See the module
    docstring for why the two are separate and why the label and score channels
    use different PSI formulations.

    Args:
        reference_predictions: Predicted classes over the reference window.
            List, numpy array, or pandas Series. Nulls are dropped. The two
            windows are independent populations and need not be the same length.
        current_predictions: Predicted classes over the current window.
        reference_scores: Optional continuous scores for the reference window --
            for the credit models, ``P(class == 1) == P(BAD)``. Must be the same
            length as ``reference_predictions``. Omit for a model with no
            probability capability; the score channel is then reported as
            unavailable rather than as zero drift.
        current_scores: Optional continuous scores for the current window.

    Returns:
        Dictionary with keys:
            label_psi (float): categorical PSI over predicted class
                frequencies, rounded to 4dp.
            label_status (str): PASS, WARNING, FAIL, or PENDING, from
                ``classify_psi(label_psi)``.
            classes_evaluated (list): classes observed in either window, in
                first-observed order.
            per_class (list of dict): one entry per observed class, in the same
                order, each with ``class``, ``reference_rate`` and
                ``current_rate``. Informational only -- no status, no threshold.
            score_psi (float or None): quantile PSI over the score
                distributions, or None when no score channel exists.
            score_ks_statistic (float or None): KS statistic over the score
                distributions, or None when no score channel exists. Reported
                as a metric only -- no KS threshold is defined
                (docs/thresholds.md).
            score_status (str): status from ``classify_psi(score_psi)``, or
                PENDING when the channel is unavailable.
            score_availability (str): ``"computed"`` or
                ``"unavailable_no_scores"``.
            status (str): the more severe of ``label_status`` and
                ``score_status`` via ``worst_status``, which skips PENDING -- so
                a model without probabilities is judged on its label channel
                alone rather than being dragged to PENDING overall.
            is_mock (bool): False -- the calculation is real.

    Raises:
        ValueError: if either prediction sequence is None, if a prediction or
            score input is not a 1-D sequence, or if a score sequence's length
            does not match its own window's prediction sequence.

    Notes:
        PENDING is returned for both channels when either window has no usable
        predictions. A model that emits a single class in both windows is NOT
        pending -- that is a real, stable distribution and correctly scores
        ``label_psi = 0.0`` (PASS).
    """
    reference_raw = _as_series(reference_predictions, "reference_predictions")
    current_raw = _as_series(current_predictions, "current_predictions")

    reference = reference_raw.dropna().to_numpy()
    current = current_raw.dropna().to_numpy()

    if len(reference) == 0 or len(current) == 0:
        return _pending_result()

    classes = _observed_classes(reference, current)
    label_psi = round(_compute_label_psi(reference, current, classes), _ROUNDING_DP)
    # Classify the reported value, as above.
    label_status = classify_psi(label_psi)

    # Raw counts, not the null-dropped ones -- see _score_channel.
    score = _score_channel(
        reference_scores,
        current_scores,
        reference_n=len(reference_raw),
        current_n=len(current_raw),
    )

    return {
        "label_psi": label_psi,
        "label_status": label_status,
        "classes_evaluated": classes,
        "per_class": _per_class_rates(reference, current, classes),
        **score,
        "status": worst_status(label_status, score["score_status"]),
        "is_mock": False,
    }
