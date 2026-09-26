"""Authoritative analytical thresholds for fairness and drift evaluation.

These thresholds are project and industry analytical conventions (engineering
heuristics). None of the thresholds defined here is an RBI regulatory
requirement.

Authoritative source: docs/thresholds.md
"""

import os

# Technical status values
STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FAIL = "FAIL"
STATUS_PENDING = "PENDING"

VALID_STATUSES = (STATUS_PASS, STATUS_WARNING, STATUS_FAIL, STATUS_PENDING)

# Disparate Impact ratio thresholds (four-fifths rule convention)
# r >= 0.80 -> PASS | 0.70 <= r < 0.80 -> WARNING | r < 0.70 -> FAIL
DI_PASS_THRESHOLD = 0.80
DI_WARNING_THRESHOLD = 0.70

# Small-group reporting threshold -- A PROJECT CONVENTION WITH NO SOURCE.
#
# WHAT THIS IS NOT
#   This default is NOT established by any project document, any RBI
#   instrument, or any statistical method this project has adopted. Unlike
#   DI_PASS_THRESHOLD and the PSI thresholds above -- which docs/thresholds.md
#   records as recognised industry conventions -- nothing in this repository
#   validates this number. It was chosen only so that the warning has some
#   default, and the team is expected to set it deliberately.
#
#   Because it is unvalidated it must never be described as a statistical
#   test, a confidence bound, or a minimum sample size "requirement". It
#   supports one descriptive statement only: the group is small, so the
#   reported ratio may be sensitive to individual records. It does NOT support
#   a claim that any result is "statistically unreliable" -- that would be a
#   statistical conclusion, and this project defines no method for drawing one.
#
# WHAT IT DOES
#   It marks which observed groups fall below it, so a reader can see that a
#   headline ratio rests on few records. That is all.
#
# WHAT IT NEVER DOES
#   * It never excludes a group from any calculation. Dropping small groups
#     would hide exactly the minorities fairness analysis exists to examine,
#     and would silently change the reported metrics.
#   * It never changes a PASS/WARNING/FAIL status. Status comes from
#     classify_disparate_impact() alone, using the unchanged thresholds above.
#     The warning and the status are independent outputs.
#
# CONFIGURING IT
#   Set the RBI_MIN_GROUP_SIZE environment variable, or call
#   set_min_group_size() at runtime. A non-positive or unparseable value is
#   rejected rather than silently defaulted, so a misconfiguration is visible.
_DEFAULT_MIN_GROUP_SIZE = 30

# True when the active value is just the unconfigured default. Surfaced to the
# UI so the screen can say the threshold has not been set by the team.
MIN_GROUP_SIZE_IS_PROJECT_DEFAULT = "RBI_MIN_GROUP_SIZE" not in os.environ


def _load_min_group_size() -> int:
    raw = os.environ.get("RBI_MIN_GROUP_SIZE")
    if raw is None:
        return _DEFAULT_MIN_GROUP_SIZE
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"RBI_MIN_GROUP_SIZE must be an integer, got {raw!r}."
        ) from None
    if value <= 0:
        raise ValueError(
            f"RBI_MIN_GROUP_SIZE must be a positive integer, got {value}."
        )
    return value


MIN_GROUP_SIZE_FOR_STABLE_RATE = _load_min_group_size()


def set_min_group_size(value: int) -> None:
    """Set the small-group reporting threshold at runtime.

    Provided so the team can configure this deliberately rather than inherit
    an unvalidated default. Changing it alters only which groups are FLAGGED;
    it cannot change a metric or a PASS/WARNING/FAIL status.
    """
    global MIN_GROUP_SIZE_FOR_STABLE_RATE, MIN_GROUP_SIZE_IS_PROJECT_DEFAULT
    value = int(value)
    if value <= 0:
        raise ValueError(f"min group size must be positive, got {value}.")
    MIN_GROUP_SIZE_FOR_STABLE_RATE = value
    MIN_GROUP_SIZE_IS_PROJECT_DEFAULT = False

# Population Stability Index (PSI) thresholds (credit-industry convention)
# p < 0.10 -> PASS | 0.10 <= p <= 0.25 -> WARNING | p > 0.25 -> FAIL
PSI_PASS_THRESHOLD = 0.10
PSI_WARNING_THRESHOLD = 0.25


def classify_disparate_impact(ratio: float) -> str:
    """Classify a disparate impact ratio into PASS, WARNING, or FAIL.

    Boundary rule:
      ratio >= 0.80           -> PASS
      0.70 <= ratio < 0.80    -> WARNING (half-open interval)
      ratio < 0.70            -> FAIL
    """
    if ratio >= DI_PASS_THRESHOLD:
        return STATUS_PASS
    if ratio >= DI_WARNING_THRESHOLD:
        return STATUS_WARNING
    return STATUS_FAIL


def classify_psi(psi: float) -> str:
    """Classify a Population Stability Index value into PASS, WARNING, or FAIL.

    Boundary rule:
      psi < 0.10              -> PASS
      0.10 <= psi <= 0.25     -> WARNING (closed interval: 0.25 is WARNING)
      psi > 0.25              -> FAIL (e.g. 0.2500001 is FAIL)
    """
    if psi < PSI_PASS_THRESHOLD:
        return STATUS_PASS
    if psi <= PSI_WARNING_THRESHOLD:
        return STATUS_WARNING
    return STATUS_FAIL


# Severity ordering over the status vocabulary, used to combine several
# independently-classified results into one overall status. This is NOT a
# threshold: it defines no new boundary and classifies no metric. It only
# says which of two ALREADY-CLASSIFIED statuses is the more severe.
#
# PENDING is deliberately absent. PENDING means "not measured", which is not a
# severity at all -- it is neither better nor worse than PASS. Ranking it would
# force a false choice: treating it as benign would hide an unmeasured channel
# behind a PASS, and treating it as severe would report absent data as a
# finding (docs/thresholds.md: missing data is never reported as FAIL).
_SEVERITY_RANK = {
    STATUS_PASS: 0,
    STATUS_WARNING: 1,
    STATUS_FAIL: 2,
}


def worst_status(*statuses: str) -> str:
    """Return the most severe of the MEASURED statuses given.

    FAIL > WARNING > PASS. ``PENDING`` inputs are skipped rather than ranked
    (see ``_SEVERITY_RANK``), so a channel that could not be measured never
    upgrades or downgrades the channels that could.

    Args:
        *statuses: Status values already produced by a classifier. Every value
            must be in ``VALID_STATUSES``.

    Returns:
        The most severe status among the non-PENDING inputs. Returns
        ``STATUS_PENDING`` when no argument is given, or when every argument is
        ``PENDING`` -- if nothing was measured, the overall result is
        "not measured", not PASS.

    Raises:
        ValueError: if any argument is not a recognised status. An unknown
            status is a contract violation, and silently ignoring it could
            drop a real FAIL from the aggregate.
    """
    unknown = [s for s in statuses if s not in VALID_STATUSES]
    if unknown:
        raise ValueError(
            f"Unknown status value(s): {unknown}. "
            f"Valid statuses are {list(VALID_STATUSES)}."
        )

    measured = [s for s in statuses if s in _SEVERITY_RANK]
    if not measured:
        return STATUS_PENDING

    return max(measured, key=lambda s: _SEVERITY_RANK[s])


def is_small_group(group_count: int) -> bool:
    """Whether a group falls below the small-group reporting threshold.

    Descriptive only: it reports that a group is small, not that any result is
    unreliable. It never removes a group from a calculation and never changes
    a status -- see ``MIN_GROUP_SIZE_FOR_STABLE_RATE``.

    Reads the module attribute at call time so ``set_min_group_size()`` takes
    effect without callers re-importing.
    """
    return int(group_count) < MIN_GROUP_SIZE_FOR_STABLE_RATE
