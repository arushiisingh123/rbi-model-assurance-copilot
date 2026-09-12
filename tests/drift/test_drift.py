"""Comprehensive tests for app.drift.drift.

Verifies PSI and KS calculation, status classification against the authoritative
central thresholds, MAX aggregation across features, feature-eligibility rules
(numeric only, non-boolean, present in both frames, finite values only), edge
cases that must return PENDING rather than an invented FAIL, and schema
conformance.
"""
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    STATUS_WARNING,
    VALID_STATUSES,
    classify_psi,
)
from app.drift import drift_report
from app.drift import drift as drift_module
from app.drift.drift import (
    _ROUNDING_DP,
    _compute_feature_ks,
    _compute_feature_psi,
    _finite_values,
)
from app.drift.scenario import build_drift_scenario

PER_FEATURE_KEYS = {"feature", "psi", "ks_statistic"}


# --------------------------------------------------------------------------
# Core calculation
# --------------------------------------------------------------------------


def test_identical_frames_report_no_drift():
    vals = np.linspace(10, 100, 100)
    ref = pd.DataFrame({"income": vals, "age": vals / 2})
    cur = pd.DataFrame({"income": vals, "age": vals / 2})

    res = drift_report(ref, cur)

    assert res["psi"] == 0.0
    assert res["ks_statistic"] == 0.0
    assert res["status"] == STATUS_PASS
    assert res["is_mock"] is False
    assert res["features_evaluated"] == ["income", "age"]


def test_known_fixture_hand_computed_ks():
    # ECDF_ref: 1->0.25, 2->0.5, 3->0.75, 4->1.0
    # ECDF_cur: 1->0.0,  2->0.0, 3->0.25, 4->0.5
    # max |diff| = 0.50 at x = 2
    ref = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"val": [3.0, 4.0, 5.0, 6.0]})

    assert drift_report(ref, cur)["ks_statistic"] == 0.50


def test_psi_status_boundaries_via_scenario():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"metric": rng.normal(100, 15, 1000)})

    _, cur_pass = build_drift_scenario(ref, shift_features="metric", shift_amount=0.0)
    res_pass = drift_report(ref, cur_pass)
    assert res_pass["psi"] < 0.10
    assert res_pass["status"] == STATUS_PASS

    _, cur_warn = build_drift_scenario(ref, shift_features="metric", shift_amount=0.35)
    res_warn = drift_report(ref, cur_warn)
    assert 0.10 <= res_warn["psi"] <= 0.25
    assert res_warn["status"] == STATUS_WARNING

    _, cur_fail = build_drift_scenario(ref, shift_features="metric", shift_amount=1.5)
    res_fail = drift_report(ref, cur_fail)
    assert res_fail["psi"] > 0.25
    assert res_fail["status"] == STATUS_FAIL


def test_max_aggregation_drives_status():
    vals = np.linspace(10, 100, 500)
    ref = pd.DataFrame({"stable_feature": vals, "drifted_feature": vals})
    cur = pd.DataFrame({"stable_feature": vals, "drifted_feature": vals + 200.0})

    res = drift_report(ref, cur)
    assert res["psi"] > 0.25
    assert res["status"] == STATUS_FAIL


def test_multiple_features_all_listed():
    rng = np.random.default_rng(123)
    ref = pd.DataFrame(
        {
            "a": rng.normal(50, 10, 300),
            "b": rng.normal(0, 1, 300),
            "c": rng.normal(-5, 2, 300),
        }
    )
    cur = pd.DataFrame(
        {
            "a": rng.normal(65, 10, 300),
            "b": rng.normal(0, 1, 300),
            "c": rng.normal(-5, 2, 300),
        }
    )

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["a", "b", "c"]
    assert res["psi"] > 0.0
    assert res["ks_statistic"] > 0.0


def test_small_dataset_does_not_crash():
    ref = pd.DataFrame({"v": [1.0, 2.0]})
    cur = pd.DataFrame({"v": [3.0, 4.0]})

    res = drift_report(ref, cur)
    assert res["status"] in VALID_STATUSES
    assert res["features_evaluated"] == ["v"]


# --------------------------------------------------------------------------
# Status is driven by PSI only, and always matches the reported PSI
# --------------------------------------------------------------------------


def test_ks_never_affects_status():
    # KS is 0.50 here, but the status must follow PSI alone.
    ref = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"val": [3.0, 4.0, 5.0, 6.0]})
    res = drift_report(ref, cur)

    assert res["ks_statistic"] == 0.50
    assert res["status"] == classify_psi(res["psi"])


def test_status_always_matches_reported_psi():
    """Regression: the reported PSI and the status must never disagree.

    The status is derived from the rounded value that is actually reported, so
    a PSI printed as 0.1 can never carry a PASS status (0.10 is WARNING).
    """
    rng = np.random.default_rng(11)
    ref = pd.DataFrame({"metric": rng.normal(100, 15, 800)})

    for shift in [0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.5, 0.8, 1.2, 2.0]:
        _, cur = build_drift_scenario(ref, shift_features="metric", shift_amount=shift)
        res = drift_report(ref, cur)
        assert res["status"] == classify_psi(res["psi"]), f"disagreed at shift={shift}"


# --------------------------------------------------------------------------
# Feature eligibility
# --------------------------------------------------------------------------


def test_categorical_columns_are_excluded():
    ref = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
    cur = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["num"]
    assert "cat" not in res["features_evaluated"]


def test_boolean_columns_are_excluded():
    """Booleans are numeric to pandas but semantically categorical."""
    ref = pd.DataFrame({"num": [1.0, 2.0, 3.0], "flag": [True, False, True]})
    cur = pd.DataFrame({"num": [1.0, 2.0, 3.0], "flag": [False, False, True]})

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["num"]


def test_features_present_in_only_one_frame_are_excluded():
    ref = pd.DataFrame({"shared": [1.0, 2.0, 3.0], "ref_only": [1.0, 2.0, 3.0]})
    cur = pd.DataFrame({"shared": [1.0, 2.0, 3.0], "cur_only": [1.0, 2.0, 3.0]})

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["shared"]


def test_feature_with_no_usable_values_is_not_listed_as_evaluated():
    """A column that is all-NaN cannot be evaluated, so it must not be claimed."""
    ref = pd.DataFrame({"good": [1.0, 2.0, 3.0], "empty": [np.nan, np.nan, np.nan]})
    cur = pd.DataFrame({"good": [1.0, 2.0, 3.0], "empty": [1.0, 2.0, 3.0]})

    res = drift_report(ref, cur)
    assert res["features_evaluated"] == ["good"]


def test_all_features_unusable_is_pending():
    ref = pd.DataFrame({"v": [np.nan, np.nan, np.nan]})
    cur = pd.DataFrame({"v": [1.0, 2.0, 3.0]})

    res = drift_report(ref, cur)
    assert res["status"] == STATUS_PENDING
    assert res["features_evaluated"] == []


# --------------------------------------------------------------------------
# Non-finite handling
# --------------------------------------------------------------------------


def test_infinities_are_excluded_not_binned():
    """Regression: infinities used to corrupt the quantile edges.

    Before the fix, a single +inf produced a PSI of ~2.58 and a spurious FAIL.
    Dropping the non-finite value must give exactly the same answer as if that
    row had simply been absent.
    """
    ref = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    cur_with_inf = pd.DataFrame({"v": [1.0, 2.0, np.inf, 4.0, 5.0]})
    cur_without = pd.DataFrame({"v": [1.0, 2.0, 4.0, 5.0]})

    assert drift_report(ref, cur_with_inf) == drift_report(ref, cur_without)


def test_infinities_do_not_produce_spurious_fail_or_warnings():
    """A single infinity in an otherwise stable sample must not fabricate drift.

    Before the fix, np.quantile over the infinity emitted a RuntimeWarning and
    produced a PSI of ~2.58, reporting FAIL on data that had not drifted.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(100, 15, 500)
    ref = pd.DataFrame({"v": base})

    contaminated = base.copy()
    contaminated[0] = np.inf
    cur = pd.DataFrame({"v": contaminated})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = drift_report(ref, cur)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, f"unexpected RuntimeWarning: {runtime_warnings}"
    assert res["status"] == STATUS_PASS
    assert res["psi"] < 0.10


def test_negative_infinity_in_reference_is_excluded():
    ref_with_inf = pd.DataFrame({"v": [-np.inf, 1.0, 2.0, 3.0, 4.0]})
    ref_without = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})

    assert drift_report(ref_with_inf, cur) == drift_report(ref_without, cur)


def test_nans_dropped_per_feature():
    ref = pd.DataFrame({"val": [1.0, 2.0, None, 4.0, 5.0]})
    cur = pd.DataFrame({"val": [None, 2.0, 3.0, 4.0, 5.0]})

    res = drift_report(ref, cur)
    assert res["status"] in VALID_STATUSES
    assert res["features_evaluated"] == ["val"]


# --------------------------------------------------------------------------
# Degenerate distributions
# --------------------------------------------------------------------------


def test_constant_feature_reports_zero_psi():
    df = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0]})
    res = drift_report(df, df)

    assert res["psi"] == 0.0
    assert res["status"] == STATUS_PASS


def test_duplicate_quantile_edges_collapse_and_drift_is_still_detected():
    """Heavily tied data produces duplicate quantile edges.

    np.unique collapses them (here 11 raw quantiles reduce to 3 edges). The
    calculation must not error, and a genuine distribution change must still be
    detected rather than swallowed by the collapsed bins.
    """
    ref = pd.DataFrame({"tied": [1.0] * 90 + [2.0] * 10})
    cur = pd.DataFrame({"tied": [1.0] * 10 + [2.0] * 90})

    res = drift_report(ref, cur)

    assert res["status"] == STATUS_FAIL
    assert res["psi"] > 0.25
    assert res["ks_statistic"] == pytest.approx(0.8, abs=1e-4)
    assert res["features_evaluated"] == ["tied"]


def test_all_identical_values_produce_no_drift():
    """A reference with zero spread has nothing to bin against -> PSI 0.0."""
    ref = pd.DataFrame({"tied": [7.0] * 50})
    cur = pd.DataFrame({"tied": [7.0] * 50})

    res = drift_report(ref, cur)
    assert res["psi"] == 0.0
    assert res["status"] == STATUS_PASS


# --------------------------------------------------------------------------
# PENDING / validation
# --------------------------------------------------------------------------


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="must not be None"):
        drift_report(None, pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="must not be None"):
        drift_report(pd.DataFrame({"a": [1]}), None)
    with pytest.raises(ValueError, match="must be pandas DataFrames"):
        drift_report([1, 2], pd.DataFrame({"a": [1]}))


def test_empty_frames_are_pending():
    empty = pd.DataFrame()
    res = drift_report(empty, empty)

    assert res["status"] == STATUS_PENDING
    assert res["psi"] == 0.0
    assert res["ks_statistic"] == 0.0
    assert res["features_evaluated"] == []


def test_empty_current_frame_is_pending():
    ref = pd.DataFrame({"v": [1.0, 2.0]})
    res = drift_report(ref, pd.DataFrame())

    assert res["status"] == STATUS_PENDING


def test_no_common_numeric_columns_is_pending():
    ref = pd.DataFrame({"name": ["Alice", "Bob"]})
    cur = pd.DataFrame({"name": ["Charlie", "David"]})

    res = drift_report(ref, cur)
    assert res["status"] == STATUS_PENDING
    assert res["features_evaluated"] == []


def test_missing_data_is_never_reported_as_fail():
    """Absent data is not evidence of drift."""
    for ref, cur in [
        (pd.DataFrame(), pd.DataFrame()),
        (pd.DataFrame({"a": ["x"]}), pd.DataFrame({"a": ["y"]})),
        (pd.DataFrame({"v": [np.nan]}), pd.DataFrame({"v": [1.0]})),
    ]:
        assert drift_report(ref, cur)["status"] != STATUS_FAIL


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_schema_full_key_set():
    ref = pd.DataFrame({"income": [100.0, 200.0, 300.0]})
    cur = pd.DataFrame({"income": [110.0, 210.0, 310.0]})
    res = drift_report(ref, cur)

    assert set(res.keys()) == {
        "features_evaluated",
        "psi",
        "ks_statistic",
        "status",
        "is_mock",
        "per_feature",
    }
    assert res["is_mock"] is False
    assert res["status"] in VALID_STATUSES
    assert isinstance(res["features_evaluated"], list)


def test_synthetic_scenario_drift_is_detected_end_to_end():
    """The synthetic scenario exists to prove detection works.

    This is demonstrated capability, not observed production drift.
    """
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"credit_amount": rng.normal(3000, 800, 600)})
    _, cur = build_drift_scenario(ref, shift_features="credit_amount", shift_amount=1.5)

    res = drift_report(ref, cur)
    assert res["psi"] > 0.25
    assert res["status"] == STATUS_FAIL
    assert res["is_mock"] is False


# --------------------------------------------------------------------------
# Per-feature detail (Phase 4, additive)
# --------------------------------------------------------------------------


def _four_feature_frames():
    """Four evaluable features with genuinely different drift shapes."""
    rng = np.random.default_rng(11)
    base = rng.normal(50, 10, 400)
    ref = pd.DataFrame(
        {
            "steady": base,
            "compressed": base,
            "shifted": base,
            "tail_heavy": base,
        }
    )
    cur = pd.DataFrame(
        {
            "steady": base,
            "compressed": (base - base.mean()) * 0.5 + base.mean(),
            "shifted": base + 5.0,
            "tail_heavy": np.concatenate([base[:-40] , base[-40:] + 200.0]),
        }
    )
    return ref, cur


def test_per_feature_length_matches_features_evaluated():
    ref, cur = _four_feature_frames()
    res = drift_report(ref, cur)

    assert len(res["per_feature"]) == len(res["features_evaluated"])
    assert len(res["per_feature"]) == 4


def test_per_feature_names_and_order_match_features_evaluated():
    """Alignment is index-for-index, so a consumer can zip the two lists."""
    ref, cur = _four_feature_frames()
    res = drift_report(ref, cur)

    assert [entry["feature"] for entry in res["per_feature"]] == res[
        "features_evaluated"
    ]


def test_per_feature_record_key_set():
    ref, cur = _four_feature_frames()

    for entry in drift_report(ref, cur)["per_feature"]:
        assert set(entry.keys()) == PER_FEATURE_KEYS
        assert isinstance(entry["feature"], str)
        assert isinstance(entry["psi"], float)
        assert isinstance(entry["ks_statistic"], float)


def test_per_feature_values_are_rounded_to_the_existing_precision():
    """Same 4dp precision as the aggregates -- no extra digits introduced."""
    ref, cur = _four_feature_frames()

    for entry in drift_report(ref, cur)["per_feature"]:
        assert entry["psi"] == round(entry["psi"], _ROUNDING_DP)
        assert entry["ks_statistic"] == round(entry["ks_statistic"], _ROUNDING_DP)


def test_max_per_feature_psi_equals_the_reported_psi():
    ref, cur = _four_feature_frames()
    res = drift_report(ref, cur)

    assert max(entry["psi"] for entry in res["per_feature"]) == res["psi"]


def test_max_per_feature_ks_equals_the_reported_ks():
    ref, cur = _four_feature_frames()
    res = drift_report(ref, cur)

    assert (
        max(entry["ks_statistic"] for entry in res["per_feature"])
        == res["ks_statistic"]
    )


def test_psi_and_ks_maxima_may_come_from_different_features():
    """The two maxima are independent -- the detail must show that, not hide it.

    ``compressed`` squeezes the distribution inward: extreme reference bins
    empty out, so PSI is large while the ECDF gap stays modest. ``shifted``
    moves the whole distribution: the ECDF gap is larger, but mass merely
    slides between adjacent bins so PSI is smaller. Different winners.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 500)
    ref = pd.DataFrame({"compressed": base, "shifted": base})
    cur = pd.DataFrame({"compressed": base * 0.5, "shifted": base + 0.5})

    res = drift_report(ref, cur)
    by_name = {entry["feature"]: entry for entry in res["per_feature"]}

    psi_argmax = max(res["per_feature"], key=lambda e: e["psi"])["feature"]
    ks_argmax = max(res["per_feature"], key=lambda e: e["ks_statistic"])["feature"]

    assert psi_argmax == "compressed"
    assert ks_argmax == "shifted"
    assert psi_argmax != ks_argmax

    # The aggregates come from those two different features.
    assert res["psi"] == by_name["compressed"]["psi"]
    assert res["ks_statistic"] == by_name["shifted"]["ks_statistic"]


def test_per_feature_values_match_the_internal_calculators():
    """The detail is the same arithmetic the aggregate was taken from."""
    ref, cur = _four_feature_frames()
    res = drift_report(ref, cur)

    for entry in res["per_feature"]:
        col = entry["feature"]
        ref_vals = _finite_values(ref[col])
        cur_vals = _finite_values(cur[col])

        assert entry["psi"] == round(
            float(_compute_feature_psi(ref_vals, cur_vals)), _ROUNDING_DP
        )
        assert entry["ks_statistic"] == round(
            float(_compute_feature_ks(ref_vals, cur_vals)), _ROUNDING_DP
        )


def test_excluded_features_are_absent_from_per_feature():
    """Whatever is not evaluated appears in neither list.

    Covers every exclusion rule at once: categorical, boolean, present in only
    one frame, and left with no finite values.
    """
    ref = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0],
            "categorical": ["a", "b", "a", "b"],
            "boolean": [True, False, True, False],
            "ref_only": [1.0, 2.0, 3.0, 4.0],
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    cur = pd.DataFrame(
        {
            "numeric": [1.5, 2.5, 3.5, 4.5],
            "categorical": ["a", "a", "b", "b"],
            "boolean": [False, False, True, True],
            "cur_only": [1.0, 2.0, 3.0, 4.0],
            "all_nan": [1.0, 2.0, 3.0, 4.0],
        }
    )

    res = drift_report(ref, cur)
    names = [entry["feature"] for entry in res["per_feature"]]

    assert res["features_evaluated"] == ["numeric"]
    assert names == ["numeric"]
    for excluded in ("categorical", "boolean", "ref_only", "cur_only", "all_nan"):
        assert excluded not in names


@pytest.mark.parametrize(
    "ref, cur",
    [
        (pd.DataFrame(), pd.DataFrame()),
        (pd.DataFrame({"a": [1.0]}), pd.DataFrame()),
        (pd.DataFrame({"a": ["x"]}), pd.DataFrame({"a": ["y"]})),
        (pd.DataFrame({"v": [np.nan]}), pd.DataFrame({"v": [1.0]})),
    ],
)
def test_pending_results_report_an_empty_per_feature_list(ref, cur):
    """Nothing evaluated means no detail -- never a fabricated zero entry."""
    res = drift_report(ref, cur)

    assert res["status"] == STATUS_PENDING
    assert res["features_evaluated"] == []
    assert res["per_feature"] == []


def test_per_feature_carries_no_status_and_no_threshold():
    """Per-feature values are informational; classification stays aggregate-only."""
    ref, cur = _four_feature_frames()
    res = drift_report(ref, cur)

    for entry in res["per_feature"]:
        assert "status" not in entry
        assert "threshold" not in entry
        assert "is_mock" not in entry

    # Status still comes from the aggregate PSI alone.
    assert res["status"] == classify_psi(res["psi"])


def test_per_feature_on_the_known_fixtures():
    """Per-feature output on the two fixtures whose aggregates are pinned above.

    The aggregates themselves are already covered by
    ``test_known_fixture_hand_computed_ks`` and
    ``test_identical_frames_report_no_drift``; only the per-feature detail is
    asserted here.
    """
    # Hand-computed KS fixture (see test_known_fixture_hand_computed_ks).
    ref = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    cur = pd.DataFrame({"val": [3.0, 4.0, 5.0, 6.0]})

    assert drift_report(ref, cur)["per_feature"][0]["ks_statistic"] == 0.50

    # No drift at all still produces a real entry, with zeros rather than a
    # missing feature (see test_identical_frames_report_no_drift).
    vals = np.linspace(10, 100, 100)
    same = pd.DataFrame({"income": vals})

    assert drift_report(same, same)["per_feature"] == [
        {"feature": "income", "psi": 0.0, "ks_statistic": 0.0}
    ]


def test_drift_module_defines_no_thresholds_of_its_own():
    """Thresholds stay in app/config/thresholds.py (docs/thresholds.md).

    The additive per-feature output must not become a place where a local
    severity rule quietly appears.
    """
    # No threshold constant is defined or imported into this namespace.
    assert not [name for name in dir(drift_module) if "THRESHOLD" in name.upper()]

    source = Path(drift_module.__file__).read_text(encoding="utf-8")
    assert "from app.config.thresholds import" in source
    assert "classify_psi(" in source

    # Inspect the code rather than the text: prose may legitimately mention a
    # boundary (the rounding comment explains why 0.10 matters), but no
    # comparison in this module may test a value against a PSI band edge.
    tree = ast.parse(source)

    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and "THRESHOLD" in target.id.upper()
    ], "app/drift must not define its own threshold constant"

    band_edges = {0.10, 0.25}
    offending = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        for operand in [node.left, *node.comparators]
        if isinstance(operand, ast.Constant)
        and isinstance(operand.value, float)
        and operand.value in band_edges
    ]
    assert not offending, (
        "app/drift must not compare against a PSI band edge; classification "
        "belongs to app.config.thresholds.classify_psi"
    )
