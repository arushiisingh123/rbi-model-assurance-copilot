"""Deterministic management interpretation and investigation guidance.

WHAT THIS IS
    Static, hand-written text selected by a lookup on values the analytical
    modules already produced -- a channel name and a status. It turns
    "prediction_drift: FAIL" into a sentence a manager can act on.

WHAT THIS IS NOT
    - **Not a diagnosis.** Nothing here identifies a root cause. A FAIL on any
      channel is consistent with several very different underlying causes, and
      the platform cannot distinguish them from the measurement alone. Every
      string below therefore proposes what to INVESTIGATE, never what is wrong.
    - **Not LLM output.** There is no model call anywhere in this module. The
      wording is fixed, reviewable, and identical on every run, which is what
      makes it safe to put in front of management.
    - **Not a compliance determination.** A red instance or a failing channel
      is not a regulatory breach, and nothing here says or implies it is.
    - **Not analytics.** No metric, threshold or status is computed or
      re-derived. A status arrives already assigned by
      ``app/config/thresholds.py`` via the owning module.

LANGUAGE RULES, DELIBERATELY ENFORCED
    Verbs are limited to *review, validate, investigate, compare, assess,
    confirm*. The module never says a model is defective, never attributes a
    finding to an employee, and never recommends retraining -- retraining is a
    decision that needs evidence this platform does not have, and recommending
    it automatically would turn a measurement into a mandate.

    ``tests/report/test_guidance.py`` pins these rules, so the constraint is
    enforced rather than merely documented.
"""

from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "CHANNEL_FAIRNESS",
    "CHANNEL_EXPLAINABILITY",
    "CHANNEL_FEATURE_DRIFT",
    "CHANNEL_PREDICTION_DRIFT",
    "INVESTIGATION_CATEGORIES",
    "instance_risk_interpretation",
    "investigation_guidance",
    "channel_guidance",
]

CHANNEL_FEATURE_DRIFT = "feature_drift"
CHANNEL_PREDICTION_DRIFT = "prediction_drift"
CHANNEL_FAIRNESS = "fairness"
CHANNEL_EXPLAINABILITY = "explainability"

# The categories an investigation can fall into. Presented as a checklist to
# work through, NOT as a ranked set of likely causes -- the platform has no
# basis for ranking them, and presenting one first would imply it did.
INVESTIGATION_CATEGORIES = (
    "input/data quality",
    "model behaviour",
    "population/data drift",
    "threshold/policy alignment",
    "operational/process",
)

# Statuses that warrant management attention. PASS needs none, and PENDING is
# an absent measurement rather than a finding.
_ACTIONABLE = ("WARNING", "FAIL")


def instance_risk_interpretation(
    prediction: Optional[Any] = None,
    probability: Optional[float] = None,
    favorable_label: Optional[Any] = None,
) -> Dict[str, Any]:
    """What a high-risk ("red") per-instance attribution actually means.

    Answers the question a reviewer asks first when they see a red row, and --
    just as importantly -- states the three things it does NOT establish.

    Args:
        prediction: The model's predicted class for this record, if known.
        probability: ``P(class == 1)`` for this record, if known.
        favorable_label: The prediction value the model's own
            ``label_semantics`` calls favourable, if declared.

    Returns:
        ``meaning`` (str), ``does_not_establish`` (list[str]),
        ``investigate`` (list[str]), and an echo of whatever identity values
        were supplied. No value is computed or inferred here.
    """
    meaning = (
        "This record received a high predicted probability of the model's "
        "positive class. In the attribution chart, the highlighted features "
        "are those pushing the model's output toward that outcome. It is the "
        "model's estimate for this record -- not a decision, and not a finding."
    )

    does_not_establish = [
        "It does not establish that the customer has done anything wrong.",
        "It does not establish that an employee acted improperly.",
        "It does not establish a breach of any RBI requirement.",
        "It does not establish that the model is defective.",
    ]

    investigate = [
        "Review the contributing feature values against the source record.",
        "Validate that the input data for this record is complete and correct.",
        "Compare the record against the reference population to see whether it "
        "is unusual or out of distribution.",
        "Assess whether the score sits close to the decision threshold, where "
        "small input differences change the outcome.",
        "Confirm that the model's behaviour on this record is consistent with "
        "current credit policy.",
    ]

    return {
        "meaning": meaning,
        "does_not_establish": does_not_establish,
        "investigate": investigate,
        "investigation_categories": list(INVESTIGATION_CATEGORIES),
        "prediction": prediction,
        "probability": probability,
        "favorable_label": favorable_label,
    }


# What to investigate when a given channel is not passing. Keyed by channel;
# the wording is deliberately channel-specific, because "review the data" is
# useless guidance and a drift finding and a fairness finding lead an
# investigator to genuinely different places.
_CHANNEL_GUIDANCE: Dict[str, Dict[str, Any]] = {
    CHANNEL_FEATURE_DRIFT: {
        "means": (
            "The distribution of the model's INPUT features has moved relative "
            "to the reference population. This is a property of the data, not "
            "of the model -- any model scoring these rows would see the same "
            "shift."
        ),
        "investigate": [
            "Review recent changes in the applicant or customer population.",
            "Validate upstream data pipelines and feed definitions for changes.",
            "Investigate new, missing or out-of-distribution values in the "
            "features showing the largest movement.",
            "Confirm the reference window is still an appropriate baseline for "
            "the period being monitored.",
        ],
    },
    CHANNEL_PREDICTION_DRIFT: {
        "means": (
            "The distribution of the MODEL'S OWN OUTPUT has moved relative to "
            "the reference window. Unlike feature drift this is model-specific: "
            "two models over the same rows can differ here."
        ),
        "investigate": [
            "Compare the current output distribution against the reference "
            "population.",
            "Review whether input drift on the same windows accounts for the "
            "movement.",
            "Assess whether the decision threshold still reflects current "
            "policy for this portfolio.",
            "Validate that scoring configuration and model version are "
            "unchanged over the period.",
        ],
    },
    CHANNEL_FAIRNESS: {
        "means": (
            "Selection rates differ across groups of the declared protected "
            "attribute by more than the configured threshold. This is a "
            "measurement of outcome distribution, not a determination of "
            "discrimination."
        ),
        "investigate": [
            "Review the affected group-level metrics and their sample sizes.",
            "Validate data quality and representation for the smaller groups.",
            "Assess whether the difference persists across monitoring windows "
            "or is specific to this one.",
            "Review model, data and process contributors together before "
            "attributing the difference to any one of them.",
        ],
    },
    CHANNEL_EXPLAINABILITY: {
        "means": (
            "Feature attributions describe how the model reached its output "
            "for these records. They explain the model's behaviour; they do "
            "not validate whether that behaviour is correct."
        ),
        "investigate": [
            "Review the contributing feature values for the flagged records.",
            "Validate the source data behind the largest contributors.",
            "Assess whether the features driving the model are consistent with "
            "documented credit policy.",
        ],
    },
}


def channel_guidance(channel: str) -> Optional[Dict[str, Any]]:
    """The static guidance entry for ``channel``, or None if not covered.

    Returns None rather than a generic fallback: inventing guidance for a
    channel nobody wrote guidance for would produce plausible-sounding advice
    with no thought behind it.
    """
    entry = _CHANNEL_GUIDANCE.get(channel)
    if entry is None:
        return None
    return {
        "channel": channel,
        "means": entry["means"],
        "investigate": list(entry["investigate"]),
    }


def investigation_guidance(
    channel_status: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """Guidance for every channel currently at WARNING or FAIL.

    Args:
        channel_status: ``{channel: status}`` as the analytical modules
            reported it -- for example a monitoring result's
            ``channel_status``. Statuses are read, never re-derived.

    Returns:
        One entry per actionable channel, each with ``channel``, ``status``,
        ``means`` and ``investigate``. Channels at PASS need no action and
        channels at PENDING were not measured, so neither appears. A channel
        with no written guidance is skipped rather than given generic advice.

        An empty list means nothing measured requires attention -- which is
        NOT the same as everything having been measured. Read it alongside the
        PENDING channels in the result it came from.
    """
    guidance: List[Dict[str, Any]] = []
    for channel, status in channel_status.items():
        if status not in _ACTIONABLE:
            continue
        entry = channel_guidance(channel)
        if entry is None:
            continue
        guidance.append({**entry, "status": status})
    return guidance
