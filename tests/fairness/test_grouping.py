"""Tests for app.fairness.grouping.

Verifies exact category extraction, missing value handling, empty inputs,
absence of A91-A95 meaning lookup tables, and module docstring claims.
"""
import inspect
import pandas as pd
import pytest
import app.fairness.grouping as grouping_module
from app.fairness.grouping import get_observed_categories, validate_fairness_inputs


def test_observed_categories_exact_reporting():
    # Observed categories are returned as-is
    data = ["A91", "A92", "A93", "A91", "A94"]
    categories = get_observed_categories(data)
    assert set(categories) == {"A91", "A92", "A93", "A94"}
    assert len(categories) == 4


def test_observed_categories_handles_missing_and_empty():
    assert get_observed_categories(None) == []
    assert get_observed_categories([]) == []
    assert get_observed_categories(pd.Series([], dtype=object)) == []

    # NaNs and Nones are filtered out
    data = ["A91", None, "A92", float("nan"), "A91"]
    categories = get_observed_categories(data)
    assert set(categories) == {"A91", "A92"}


def test_validate_fairness_inputs_success():
    preds = [1, 0, 1, 0]
    sens = ["A91", "A92", "A91", "A92"]
    clean_p, clean_s = validate_fairness_inputs(preds, sens)
    assert len(clean_p) == 4
    assert len(clean_s) == 4


def test_validate_fairness_inputs_drops_nans():
    preds = [1, None, 0, 1]
    sens = ["A91", "A92", None, "A92"]
    clean_p, clean_s = validate_fairness_inputs(preds, sens)
    assert len(clean_p) == 2
    assert list(clean_p) == [1, 1]
    assert list(clean_s) == ["A91", "A92"]


def test_validate_fairness_inputs_errors():
    # None inputs
    with pytest.raises(ValueError, match="must not be None"):
        validate_fairness_inputs(None, ["A91"])
    with pytest.raises(ValueError, match="must not be None"):
        validate_fairness_inputs([1], None)

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        validate_fairness_inputs([1, 0], ["A91"])

    # Empty inputs
    with pytest.raises(ValueError, match="must not be empty"):
        validate_fairness_inputs([], [])

    # All NaNs
    with pytest.raises(ValueError, match="No valid rows remaining"):
        validate_fairness_inputs([None, None], ["A91", None])


def test_no_codebook_or_meaning_lookup_table():
    # Ensure no code->meaning dictionary or gender mapping exists in grouping
    module_source = inspect.getsource(grouping_module)
    assert "divorced" not in module_source.lower()
    assert "married" not in module_source.lower()
    assert "single" not in module_source.lower()
    assert "male" not in module_source.lower()
    assert "female" not in module_source.lower()


def test_module_docstring_states_raw_categories():
    doc = grouping_module.__doc__
    assert doc is not None
    assert "personal-status-and-sex" in doc
    assert "sex or gender" in doc
