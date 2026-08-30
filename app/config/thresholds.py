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
