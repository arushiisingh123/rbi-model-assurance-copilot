"""Tests for app.config.thresholds.

Verifies boundary precision, doc-parity with docs/thresholds.md, absence of
invented KS / demographic parity thresholds, and valid status definitions.
"""
from pathlib import Path
import app.config.thresholds as thresholds
from app.config.thresholds import (
    DI_PASS_THRESHOLD,
    DI_WARNING_THRESHOLD,
    PSI_PASS_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    STATUS_WARNING,
    VALID_STATUSES,
    classify_disparate_impact,
    classify_psi,
    worst_status,
)


def test_valid_statuses_defined():
    assert STATUS_PASS == "PASS"
    assert STATUS_WARNING == "WARNING"
    assert STATUS_FAIL == "FAIL"
    assert STATUS_PENDING == "PENDING"
    assert set(VALID_STATUSES) == {"PASS", "WARNING", "FAIL", "PENDING"}


def test_disparate_impact_exact_boundaries():
    # >= 0.80 -> PASS
    assert classify_disparate_impact(1.0) == STATUS_PASS
    assert classify_disparate_impact(0.85) == STATUS_PASS
    assert classify_disparate_impact(0.80) == STATUS_PASS

    # 0.70 <= r < 0.80 -> WARNING
    assert classify_disparate_impact(0.7999) == STATUS_WARNING
    assert classify_disparate_impact(0.75) == STATUS_WARNING
    assert classify_disparate_impact(0.70) == STATUS_WARNING

    # < 0.70 -> FAIL
    assert classify_disparate_impact(0.6999) == STATUS_FAIL
    assert classify_disparate_impact(0.50) == STATUS_FAIL
    assert classify_disparate_impact(0.0) == STATUS_FAIL


def test_psi_exact_boundaries():
    # < 0.10 -> PASS
    assert classify_psi(0.0) == STATUS_PASS
    assert classify_psi(0.05) == STATUS_PASS
    assert classify_psi(0.0999) == STATUS_PASS

    # 0.10 <= p <= 0.25 -> WARNING
    assert classify_psi(0.10) == STATUS_WARNING
    assert classify_psi(0.18) == STATUS_WARNING
    assert classify_psi(0.25) == STATUS_WARNING

    # > 0.25 -> FAIL (asymmetry: 0.25 is WARNING, 0.2501 is FAIL)
    assert classify_psi(0.2501) == STATUS_FAIL
    assert classify_psi(0.2500001) == STATUS_FAIL
    assert classify_psi(0.50) == STATUS_FAIL


def test_no_ks_or_parity_thresholds_exposed():
    # Assert module exposes NO constant whose name contains "KS" or "PARITY"
    exposed_names = [attr for attr in dir(thresholds) if not attr.startswith("__")]
    for name in exposed_names:
        assert "KS" not in name.upper(), f"Unexpected KS threshold constant exposed: {name}"
        assert "PARITY" not in name.upper(), f"Unexpected PARITY threshold constant exposed: {name}"


def test_doc_parity_with_thresholds_md():
    # Verify that the 4 threshold literals match docs/thresholds.md
    docs_path = Path(__file__).parent.parent.parent / "docs" / "thresholds.md"
    assert docs_path.exists(), "docs/thresholds.md does not exist"
    doc_content = docs_path.read_text(encoding="utf-8")

    assert "0.80" in doc_content
    assert "0.70" in doc_content
    assert "0.10" in doc_content
    assert "0.25" in doc_content

    assert DI_PASS_THRESHOLD == 0.80
    assert DI_WARNING_THRESHOLD == 0.70
    assert PSI_PASS_THRESHOLD == 0.10
    assert PSI_WARNING_THRESHOLD == 0.25


# ---------------------------------------------------------------------------
# worst_status -- combining already-classified statuses (Arushi, monitoring)
# ---------------------------------------------------------------------------


def test_worst_status_severity_ordering():
    """FAIL beats WARNING beats PASS, whatever order they arrive in."""
    assert worst_status(STATUS_PASS, STATUS_WARNING, STATUS_FAIL) == STATUS_FAIL
    assert worst_status(STATUS_FAIL, STATUS_PASS) == STATUS_FAIL
    assert worst_status(STATUS_PASS, STATUS_WARNING) == STATUS_WARNING
    assert worst_status(STATUS_WARNING, STATUS_PASS) == STATUS_WARNING
    assert worst_status(STATUS_PASS, STATUS_PASS) == STATUS_PASS
    assert worst_status(STATUS_WARNING) == STATUS_WARNING


def test_worst_status_skips_pending_rather_than_ranking_it():
    """PENDING means "not measured", which is not a severity.

    An unmeasured channel must neither hide a real finding behind a PASS nor be
    reported as a finding itself -- so it is skipped, and the measured channels
    decide the outcome on their own.
    """
    assert worst_status(STATUS_PENDING, STATUS_PASS) == STATUS_PASS
    assert worst_status(STATUS_PENDING, STATUS_FAIL) == STATUS_FAIL
    assert worst_status(STATUS_PENDING, STATUS_WARNING) == STATUS_WARNING
    # A PENDING alongside a PASS must not become WARNING or FAIL...
    assert worst_status(STATUS_PASS, STATUS_PENDING) != STATUS_FAIL
    # ...and must not suppress a FAIL either.
    assert worst_status(STATUS_FAIL, STATUS_PENDING, STATUS_PENDING) == STATUS_FAIL


def test_worst_status_is_pending_when_nothing_was_measured():
    """No arguments, or all PENDING, means "not measured" -- never PASS.

    Returning PASS here would report a clean bill of health for a run in which
    no channel produced a measurement at all.
    """
    assert worst_status() == STATUS_PENDING
    assert worst_status(STATUS_PENDING) == STATUS_PENDING
    assert worst_status(STATUS_PENDING, STATUS_PENDING) == STATUS_PENDING


def test_worst_status_rejects_an_unknown_status():
    """An unrecognised status is refused, not silently dropped.

    Ignoring it could discard a real FAIL from the aggregate and report the run
    as passing.
    """
    import pytest

    with pytest.raises(ValueError, match="Unknown status value"):
        worst_status(STATUS_PASS, "CRITICAL")


def test_worst_status_introduces_no_new_status_value():
    """The vocabulary stays exactly the four existing statuses."""
    combinations = [
        (STATUS_PASS, STATUS_WARNING),
        (STATUS_FAIL, STATUS_PENDING),
        (STATUS_PENDING,),
        (),
    ]
    for combo in combinations:
        assert worst_status(*combo) in VALID_STATUSES
