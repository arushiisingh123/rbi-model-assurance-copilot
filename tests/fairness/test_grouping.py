"""Tests for app.fairness.grouping.

Verifies exact category extraction, missing value handling, empty inputs,
absence of A91-A95 meaning lookup tables, and module docstring claims.
"""
import inspect
import pandas as pd
import pytest
import app.fairness.grouping as grouping_module
from app.fairness import fairness_report
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


# --------------------------------------------------------------------------
# Positional pairing contract
#
# predictions and sensitive_feature describe the same records in the same order
# and are paired by POSITION. Pandas aligns binary operations on the index, so a
# pandas input carrying a non-positional index used to be re-paired by index --
# and where the indexes only partially overlapped, the non-matching rows were
# silently dropped. That produced a valid-looking fairness result computed from
# mismatched pairs on a subset of the records.
#
# Chosen behaviour: reject any pandas input whose index is not 0..n-1, rather
# than guess which pairing was intended. Accepting it positionally would also be
# defensible, but silently re-interpreting a caller's index is exactly the class
# of mistake this contract exists to prevent, and every pandas input that
# actually reaches fairness today already carries a 0..n-1 index.
# --------------------------------------------------------------------------


def test_partial_index_overlap_raises_instead_of_silently_dropping_rows():
    """Regression for the Phase 3 audit finding.

    predictions is a 4-element list (index 0..3); sensitive_feature is a
    4-element Series indexed [2, 3, 4, 5]. The indexes overlap only at {2, 3}.
    Previously this silently returned 2 rows, pairing predictions [0, 0] with
    groups ['A', 'A'] -- neither the positional nor the intended pairing.
    """
    preds = [1, 1, 0, 0]
    sens = pd.Series(["A", "A", "B", "B"], index=[2, 3, 4, 5])

    with pytest.raises(ValueError, match="non-positional index"):
        validate_fairness_inputs(preds, sens)


def test_partial_index_overlap_raises_through_fairness_report():
    """The guard must hold at the public entry point, not just the helper."""
    preds = [1, 1, 0, 0]
    sens = pd.Series(["A", "A", "B", "B"], index=[2, 3, 4, 5])

    with pytest.raises(ValueError, match="non-positional index"):
        fairness_report(preds, sens, favorable_label=1)


def test_fully_non_default_index_raises():
    """A wholly disjoint index is rejected for the same reason.

    Before the fix this raised only incidentally -- every row was dropped, so
    the "No valid rows remaining" check fired. It now fails on the real cause.
    """
    preds = [1, 1, 0, 0]
    sens = pd.Series(["A", "A", "B", "B"], index=[10, 11, 12, 13])

    with pytest.raises(ValueError, match="non-positional index"):
        validate_fairness_inputs(preds, sens)


def test_non_positional_predictions_series_also_raises():
    """The rule applies to predictions too, not only to sensitive_feature."""
    preds = pd.Series([1, 1, 0, 0], index=[5, 6, 7, 8])
    sens = ["A", "A", "B", "B"]

    with pytest.raises(ValueError, match="predictions has a non-positional index"):
        validate_fairness_inputs(preds, sens)


def test_error_message_names_the_remedy():
    """A caller hitting this needs to know how to fix it."""
    sens = pd.Series(["A", "B"], index=[7, 8])
    with pytest.raises(ValueError) as exc:
        validate_fairness_inputs([1, 0], sens)

    message = str(exc.value)
    assert "reset_index(drop=True)" in message
    assert "to_numpy()" in message


def test_reset_index_makes_a_rejected_input_acceptable():
    """The documented remedy actually works and preserves order."""
    preds = [1, 1, 0, 0]
    rejected = pd.Series(["A", "A", "B", "B"], index=[2, 3, 4, 5])

    clean_p, clean_s = validate_fairness_inputs(
        preds, rejected.reset_index(drop=True)
    )
    assert len(clean_p) == 4
    assert list(clean_p) == [1, 1, 0, 0]
    assert list(clean_s) == ["A", "A", "B", "B"]


def test_to_numpy_makes_a_rejected_input_acceptable():
    preds = [1, 1, 0, 0]
    rejected = pd.Series(["A", "A", "B", "B"], index=[2, 3, 4, 5])

    clean_p, clean_s = validate_fairness_inputs(preds, rejected.to_numpy())
    assert len(clean_p) == 4
    assert list(clean_s) == ["A", "A", "B", "B"]


def test_default_indexed_series_still_accepted():
    """The shape every real caller uses today must keep working."""
    preds = [1, 1, 0, 0]
    sens = pd.Series(["A", "A", "B", "B"], name="personal_status_and_sex")

    clean_p, clean_s = validate_fairness_inputs(preds, sens)
    assert len(clean_p) == 4
    assert list(clean_s) == ["A", "A", "B", "B"]


def test_integer_index_equal_to_range_is_accepted():
    """Index([0, 1, ...]) is positionally identical to RangeIndex and is fine."""
    preds = pd.Series([1, 0], index=[0, 1])
    sens = pd.Series(["A", "B"], index=[0, 1])

    clean_p, clean_s = validate_fairness_inputs(preds, sens)
    assert list(clean_p) == [1, 0]
    assert list(clean_s) == ["A", "B"]


def test_pairing_is_positional_not_index_joined():
    """Two Series with matching 0..n-1 indexes pair element-for-element."""
    preds = pd.Series([1, 0, 1, 0])
    sens = pd.Series(["A", "A", "B", "B"])

    clean_p, clean_s = validate_fairness_inputs(preds, sens)
    assert list(zip(clean_p, clean_s)) == [(1, "A"), (0, "A"), (1, "B"), (0, "B")]


def test_null_dropping_is_still_row_wise_and_not_index_driven():
    """Nulls remove only their own row; surviving pairs stay correctly matched."""
    preds = pd.Series([1, None, 0, 1])
    sens = pd.Series(["A", "A", None, "B"])

    clean_p, clean_s = validate_fairness_inputs(preds, sens)
    assert list(zip(clean_p, clean_s)) == [(1, "A"), (1, "B")]


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
