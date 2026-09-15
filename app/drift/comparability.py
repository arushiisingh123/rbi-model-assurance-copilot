"""Cross-model drift comparability enforcement (owner: Arushi — Phase 5).

Decides whether two drift assurance envelopes may be compared at all, and
refuses when they may not. The refusal is the product: a caller cannot obtain a
normal comparison result for an incompatible pair, so an invalid side-by-side
cannot be produced by ignoring an advisory flag.

WHY THIS EXISTS — THE MAX AGGREGATION TRAP
    ``drift_report()`` reports ``psi`` and ``ks_statistic`` as the MAXIMUM
    across the features it actually evaluated, and ``features_evaluated`` is the
    intersection of usable numeric columns in the two frames it was given. The
    aggregate is therefore a maximum over a *set*, and that set is part of the
    operand.

    Comparing ``max(S_a)`` with ``max(S_b)`` when ``S_a != S_b`` compares two
    different functions, not two measurements of one thing -- even when every
    feature the two sets share drifted identically. A model evaluated on one
    extra volatile feature can report a far higher PSI, and because
    ``status = classify_psi(psi)`` the distortion propagates into the
    PASS/WARNING/FAIL vocabulary. The result is a confident statement about
    relative drift that no real difference in drift produced.

    ``tests/drift/test_comparability.py`` pins a worked example of exactly this.

WHAT THIS MODULE IS NOT
    - It computes nothing. No PSI, no KS, no rounding, no aggregation, no
      re-derivation of any reported value. It reads identity and feature
      metadata and compares them.
    - It defines no threshold and classifies no severity. The four analytical
      statuses keep their existing meanings, produced where they always were
      (``app/config/thresholds.py`` remains the single source of truth).
    - It is not a per-feature comparison. Comparing the shared features of two
      otherwise-incompatible results is a legitimate future extension, but it
      is not what ``COMPARABLE`` licenses here and it is deliberately not
      implemented.

API-LAYER INDEPENDENCE
    This module imports nothing from ``app.api``. It operates on
    envelope-shaped plain dictionaries, mirroring ``drift_report()``'s own
    dict-returning contract, so ``app/drift/`` acquires no dependency on the
    API layer. Callers holding a pydantic ``DriftAssuranceEnvelope`` pass
    ``envelope.model_dump()``; the returned dict is shaped for
    ``DriftComparisonResult`` but is constructed without importing it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "COMPARABLE",
    "NOT_COMPARABLE",
    "REASON_PREFIXES",
    "compare_drift_envelopes",
]

# Comparability outcomes. These mirror the two literals the approved
# DriftComparisonResult schema accepts; they are declared here rather than
# imported so this module stays free of app.api.
COMPARABLE = "COMPARABLE"
NOT_COMPARABLE = "NOT_COMPARABLE"

# The status value drift_report() reports when nothing could be evaluated.
# Declared locally for the same reason -- importing the status vocabulary from
# the threshold config is unnecessary here, since this module never classifies.
_PENDING_STATUS = "PENDING"

# Stable machine-readable prefixes for NOT_COMPARABLE reasons.
#
# The approved schema gives ``reason`` as a free-text Optional[str], so a
# caller that needs to branch on *why* a pair was refused has only the string
# to work with. These prefixes make that possible without adding a field to a
# frozen shared contract: the prefix is the contract, the remainder is prose.
REASON_PENDING = "pending_result:"
REASON_DATASET_ID_MISMATCH = "dataset_id_mismatch:"
REASON_DATASET_VERSION_MISMATCH = "dataset_version_mismatch:"
REASON_FEATURE_SET_MISMATCH = "feature_set_mismatch:"
REASON_FEATURE_ORDER_MISMATCH = "feature_order_mismatch:"
REASON_FEATURE_SPACE_MISMATCH = "feature_space_mismatch:"

REASON_PREFIXES: Tuple[str, ...] = (
    REASON_PENDING,
    REASON_DATASET_ID_MISMATCH,
    REASON_DATASET_VERSION_MISMATCH,
    REASON_FEATURE_SET_MISMATCH,
    REASON_FEATURE_ORDER_MISMATCH,
    REASON_FEATURE_SPACE_MISMATCH,
)

_LABEL_A = "drift_a"
_LABEL_B = "drift_b"


def compare_drift_envelopes(
    envelope_a: Mapping[str, Any],
    envelope_b: Mapping[str, Any],
) -> Dict[str, Any]:
    """Decide whether two drift assurance envelopes may be compared.

    Parameters
    ----------
    envelope_a, envelope_b
        Drift assurance envelopes as plain dictionaries, shaped like the
        approved ``DriftAssuranceEnvelope``: ``context`` (with ``model_id``,
        ``model_version``, ``assurance_run_id``, optional ``adapter_id``),
        ``dataset_id``, optional ``dataset_version``, ``feature_space``, and
        ``result`` (a ``drift_report()`` output).

    Returns
    -------
    dict
        Shaped for ``DriftComparisonResult``::

            {"comparability": "COMPARABLE" | "NOT_COMPARABLE",
             "reason": str | None,
             "drift_a": <envelope_a, unchanged>,
             "drift_b": <envelope_b, unchanged>}

        Both envelopes are returned by reference, unmodified, in every
        outcome -- a caller inspecting a refusal needs to see what was
        refused.

    Notes
    -----
    Conditions are evaluated in a fixed order and the first failure decides
    the outcome, so a reason always names the *first* reason the pair was
    refused rather than an arbitrary one.

    Three things deliberately do NOT affect comparability:

    - ``model_id`` differing. Comparing two models is the reason this function
      exists.
    - ``model_version`` differing. Comparing versions of one model is a
      legitimate assurance question.
    - ``assurance_run_id`` differing. The approved design mints a run id per
      envelope, so two envelopes from one logical run already differ; making
      it a condition would refuse every real pair.
    """
    # Envelope A is checked first, then B, so the reason always names the
    # first envelope that lacks a measurement. Checked explicitly rather than
    # with ``or`` so the control flow does not depend on the reason string
    # being truthy.
    pending = _pending_reason(_LABEL_A, envelope_a)
    if pending is None:
        pending = _pending_reason(_LABEL_B, envelope_b)
    if pending is not None:
        return _refuse(envelope_a, envelope_b, f"{REASON_PENDING} {pending}")

    dataset_id_a = envelope_a.get("dataset_id")
    dataset_id_b = envelope_b.get("dataset_id")
    if dataset_id_a != dataset_id_b:
        return _refuse(
            envelope_a,
            envelope_b,
            f"{REASON_DATASET_ID_MISMATCH} {_LABEL_A} dataset_id="
            f"{dataset_id_a!r} vs {_LABEL_B} dataset_id={dataset_id_b!r}. "
            "Drift describes a change between two datasets, so results "
            "measured on different datasets are not comparable.",
        )

    # ``dataset_version`` is None on every envelope orchestration currently
    # builds, so None == None is intentionally accepted: refusing an
    # unknown-vs-unknown pair would make every real comparison
    # NOT_COMPARABLE. ``dataset_id`` equality already constrains the pair.
    # Revisit when orchestration emits real versions -- two *unknown* versions
    # are not provably the same version.
    version_a = envelope_a.get("dataset_version")
    version_b = envelope_b.get("dataset_version")
    if version_a != version_b:
        return _refuse(
            envelope_a,
            envelope_b,
            f"{REASON_DATASET_VERSION_MISMATCH} {_LABEL_A} dataset_version="
            f"{version_a!r} vs {_LABEL_B} dataset_version={version_b!r}. "
            "Different versions of a dataset are different reference "
            "populations.",
        )

    features_a = _features(envelope_a)
    features_b = _features(envelope_b)

    if set(features_a) != set(features_b):
        only_a = sorted(set(features_a) - set(features_b))
        only_b = sorted(set(features_b) - set(features_a))
        return _refuse(
            envelope_a,
            envelope_b,
            f"{REASON_FEATURE_SET_MISMATCH} evaluated feature sets differ "
            f"(only in {_LABEL_A}: {only_a}; only in {_LABEL_B}: {only_b}). "
            "Aggregate psi and ks_statistic are each a MAX over the features "
            "actually evaluated, so maxima taken over different feature sets "
            "are not comparable -- even where the shared features agree "
            "exactly.",
        )

    if features_a != features_b:
        # Same members (the set check above passed) in a different order.
        #
        # This project's feature schema (app.models.preprocessing's
        # FEATURE_COLUMNS) carries no duplicate labels, so in practice a
        # multiplicity difference cannot reach this branch. That is a property
        # of the dataset schema, not a guarantee from pandas (which permits
        # duplicate column labels) or from drift_report(). Should duplicates
        # ever occur, the pair is still correctly refused here; only the
        # wording would understate the cause.
        return _refuse(
            envelope_a,
            envelope_b,
            f"{REASON_FEATURE_ORDER_MISMATCH} the same features were "
            f"evaluated in a different order ({_LABEL_A}: {features_a}; "
            f"{_LABEL_B}: {features_b}). feature_space is derived from the "
            "ordered list, so the two envelopes carry different fingerprints "
            "for the same members; reconcile the ordering before comparing.",
        )

    feature_space_a = envelope_a.get("feature_space")
    feature_space_b = envelope_b.get("feature_space")
    if feature_space_a != feature_space_b:
        # Backstop. With identical ordered feature lists a correctly derived
        # feature_space matches, so reaching here means at least one envelope
        # carries a fingerprint that does not describe its own
        # features_evaluated. Refuse rather than trust the mismatch away.
        return _refuse(
            envelope_a,
            envelope_b,
            f"{REASON_FEATURE_SPACE_MISMATCH} {_LABEL_A} feature_space="
            f"{feature_space_a!r} vs {_LABEL_B} feature_space="
            f"{feature_space_b!r}, although both evaluated {features_a}. "
            "At least one fingerprint does not describe its own "
            "features_evaluated.",
        )

    return _permit(envelope_a, envelope_b)


def _permit(
    envelope_a: Mapping[str, Any], envelope_b: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build a COMPARABLE outcome, noting self-comparison where it applies."""
    context_a = envelope_a.get("context") or {}
    context_b = envelope_b.get("context") or {}
    same_identity = (
        context_a.get("model_id") == context_b.get("model_id")
        and context_a.get("model_version") == context_b.get("model_version")
    )

    if same_identity:
        reason = (
            "Comparable; both envelopes refer to the same model identity and "
            "version, so this is effectively a self-comparison."
        )
    else:
        reason = (
            "Comparable; the envelopes satisfy the required compatibility "
            "conditions (same dataset and version, same evaluated features in "
            "the same order, same feature_space)."
        )

    return {
        "comparability": COMPARABLE,
        "reason": reason,
        "drift_a": envelope_a,
        "drift_b": envelope_b,
    }


def _refuse(
    envelope_a: Mapping[str, Any],
    envelope_b: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    """Build a NOT_COMPARABLE outcome, preserving both envelopes by reference."""
    return {
        "comparability": NOT_COMPARABLE,
        "reason": reason,
        "drift_a": envelope_a,
        "drift_b": envelope_b,
    }


def _features(envelope: Mapping[str, Any]) -> List[str]:
    """The ordered feature list an envelope's result actually evaluated."""
    result = envelope.get("result") or {}
    return list(result.get("features_evaluated") or [])


def _pending_reason(label: str, envelope: Mapping[str, Any]) -> Optional[str]:
    """Describe why an envelope holds no comparable measurement, or None.

    Tolerates a hand-built envelope with a missing or empty ``result``: an
    absent measurement is treated as PENDING and refused, rather than raising.
    A pair where nothing was evaluated has nothing to compare, and reporting
    that plainly is more useful than a KeyError.
    """
    result = envelope.get("result") or {}

    if result.get("status") == _PENDING_STATUS:
        return (
            f"{label} has status {_PENDING_STATUS}; no feature could be "
            "evaluated, so it holds no drift measurement to compare."
        )

    if not result.get("features_evaluated"):
        return (
            f"{label} evaluated no features; there is no drift measurement "
            "to compare."
        )

    return None
