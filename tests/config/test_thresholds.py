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
