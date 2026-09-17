"""Authoritative analytical thresholds for fairness and drift evaluation.

These thresholds are project and industry analytical conventions (engineering
heuristics). None of the thresholds defined here is an RBI regulatory
requirement.

Authoritative source: docs/thresholds.md
"""

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
