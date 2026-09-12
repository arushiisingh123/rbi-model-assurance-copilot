"""Tests for the Explainability panel's pure presentation logic (Phase 4).

The binding Phase 4 constraint is that the dashboard must not recalculate
analytical results. These tests exercise that directly: every number the
presentation layer emits must be the exact float
``app.explainability.explain()`` produced.

Real explain() output is used wherever practical; small hand-built
explanations are used only where a specific malformed or edge case is being
pinned.
"""

import json

import pytest

from app.explainability.evidence import SCALE_BY_METHOD
from app.explainability.explain import RAW_FEATURES, explain
from app.models.model import predict_batch
from dashboard.panels.explainability_presentation import (
    DEFAULT_TOP_N,
    full_contribution_rows,
    global_importance_rows,
    instance_contribution_rows,
    instance_options,
    scale_caption,
    scale_label,
)

EXPLAINED_ROWS = 3


@pytest.fixture(scope="module")
def real_model_output() -> dict:
    full = predict_batch()
    return {**full, "feature_matrix": full["feature_matrix"].head(EXPLAINED_ROWS)}


@pytest.fixture(scope="module")
def shap_explanation(real_model_output) -> dict:
    return explain(model_output=real_model_output, method="shap")


@pytest.fixture(scope="module")
def lime_explanation(real_model_output) -> dict:
    return explain(model_output=real_model_output, method="lime")


@pytest.fixture(scope="module")
def real_instance_ids(real_model_output) -> list:
    return list(real_model_output["instance_ids"][:EXPLAINED_ROWS])


def _synthetic_explanation() -> dict:
    """A tiny hand-built explanation with deliberate signs and magnitudes."""
    return {
        "method": "shap",
        "per_instance": [
            {"row_index": 0, "contributions": {"a": -0.9, "b": 0.5, "c": -0.1}},
            {"row_index": 1, "contributions": {"a": 0.2, "b": -0.7, "c": 0.3}},
        ],
        "global_importance": {"a": 0.55, "b": 0.60, "c": 0.20},
        "is_mock": False,
    }


# ---------------------------------------------------------------------------
# 1, 2 — sorting
# ---------------------------------------------------------------------------


def test_global_importance_is_sorted_descending(shap_explanation):
    rows = global_importance_rows(shap_explanation)
    values = [row["importance"] for row in rows]

    assert values == sorted(values, reverse=True)
    assert len(rows) == len(RAW_FEATURES)
    assert {row["feature"] for row in rows} == set(RAW_FEATURES)


def test_instance_contributions_sorted_by_absolute_magnitude():
    rows = instance_contribution_rows(_synthetic_explanation(), 0)

    assert [row["feature"] for row in rows] == ["a", "b", "c"]
    magnitudes = [row["abs_contribution"] for row in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)


# ---------------------------------------------------------------------------
# 3, 11 — signed values preserved exactly
# ---------------------------------------------------------------------------


def test_signed_contributions_are_preserved_not_absolutised():
    """Ordering may use abs(); the displayed value must stay signed."""
    rows = instance_contribution_rows(_synthetic_explanation(), 0)
    by_feature = {row["feature"]: row["contribution"] for row in rows}

    assert by_feature == {"a": -0.9, "b": 0.5, "c": -0.1}
    assert by_feature["a"] < 0, "a negative contribution must not become positive"


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_displayed_values_are_identical_to_explain_output(method, request):
    """The Phase 4 binding constraint, asserted on real data."""
    explanation = request.getfixturevalue(f"{method}_explanation")

    global_rows = global_importance_rows(explanation)
    assert {r["feature"]: r["importance"] for r in global_rows} == explanation[
        "global_importance"
    ]

    for position, row in enumerate(explanation["per_instance"]):
        presented = instance_contribution_rows(explanation, position)
        assert {r["feature"]: r["contribution"] for r in presented} == row[
            "contributions"
        ], "presentation layer altered a contribution value"


def test_full_matrix_values_are_unchanged(shap_explanation):
    rows = full_contribution_rows(shap_explanation)
    assert len(rows) == EXPLAINED_ROWS * len(RAW_FEATURES)

    for position, source_row in enumerate(shap_explanation["per_instance"]):
        emitted = {
            r["feature"]: r["contribution"]
            for r in rows
            if r["row_index"] == source_row["row_index"]
        }
        assert emitted == source_row["contributions"]


def test_presentation_does_not_mutate_the_explanation(shap_explanation):
    import copy

    before = copy.deepcopy(dict(shap_explanation))
    global_importance_rows(shap_explanation)
    instance_contribution_rows(shap_explanation, 0, top_n=3)
    full_contribution_rows(shap_explanation)
    assert dict(shap_explanation) == before


# ---------------------------------------------------------------------------
# 4 — top-N
# ---------------------------------------------------------------------------


def test_top_n_selects_the_largest_magnitudes(shap_explanation):
    all_rows = instance_contribution_rows(shap_explanation, 0)
    top = instance_contribution_rows(shap_explanation, 0, top_n=5)

    assert len(top) == 5
    assert top == all_rows[:5]


def test_top_n_larger_than_available_returns_everything(shap_explanation):
    rows = instance_contribution_rows(shap_explanation, 0, top_n=999)
    assert len(rows) == len(RAW_FEATURES)


def test_invalid_top_n_raises(shap_explanation):
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="top_n must be a positive int"):
            instance_contribution_rows(shap_explanation, 0, top_n=bad)


def test_default_top_n_is_sensible():
    assert 1 <= DEFAULT_TOP_N <= len(RAW_FEATURES)


# ---------------------------------------------------------------------------
# 5, 6 — scale labels come from the existing source of truth
# ---------------------------------------------------------------------------


def test_shap_scale_label_is_log_odds():
    assert scale_label("shap") == "log_odds"
    assert scale_label("SHAP") == "log_odds"
    assert "log_odds" in scale_caption("shap")


def test_lime_scale_label_is_probability():
    assert scale_label("lime") == "probability"
    assert "probability" in scale_caption("lime")


def test_scale_labels_are_read_from_the_analytical_module():
    """Not hardcoded here — drift with evidence.py must be impossible."""
    for method, expected in SCALE_BY_METHOD.items():
        assert scale_label(method) == expected
    assert scale_label("shap") != scale_label("lime")


def test_unknown_method_scale_raises():
    with pytest.raises(ValueError, match="Unknown explainability method"):
        scale_label("magic")


# ---------------------------------------------------------------------------
# 7 — global and instance stay separate
# ---------------------------------------------------------------------------


def test_global_and_instance_outputs_are_structurally_distinct(shap_explanation):
    global_rows = global_importance_rows(shap_explanation)
    instance_rows = instance_contribution_rows(shap_explanation, 0)

    assert set(global_rows[0]) == {"feature", "importance"}
    assert set(instance_rows[0]) == {"feature", "contribution", "abs_contribution"}

    # A global row carries no per-record identity of any kind.
    for row in global_rows:
        for forbidden in ("contribution", "instance_id", "row_index", "position"):
            assert forbidden not in row


def test_global_importance_is_not_derived_from_per_instance():
    """Global values must be passed through, never recomputed from rows."""
    explanation = _synthetic_explanation()
    # Deliberately inconsistent with per_instance: a recomputing implementation
    # would "correct" these numbers; a presentation layer must not.
    explanation["global_importance"] = {"a": 99.0, "b": 1.0, "c": 0.5}

    rows = global_importance_rows(explanation)
    assert {r["feature"]: r["importance"] for r in rows} == {
        "a": 99.0,
        "b": 1.0,
        "c": 0.5,
    }
    assert rows[0]["feature"] == "a"


# ---------------------------------------------------------------------------
# 9 — malformed input
# ---------------------------------------------------------------------------


def test_malformed_explanation_raises_clearly():
    with pytest.raises(ValueError, match="missing required key"):
        global_importance_rows({"method": "shap"})

    with pytest.raises(ValueError, match="must be a mapping"):
        global_importance_rows("not a dict")

    with pytest.raises(ValueError, match="must be a mapping"):
        global_importance_rows({"method": "shap", "per_instance": [], "global_importance": []})


def test_empty_per_instance_raises_clearly():
    explanation = {
        "method": "shap",
        "per_instance": [],
        "global_importance": {"a": 1.0},
        "is_mock": False,
    }
    with pytest.raises(ValueError, match="is empty"):
        instance_contribution_rows(explanation, 0)
    assert instance_options(explanation) == []


def test_out_of_range_position_raises_clearly(shap_explanation):
    with pytest.raises(ValueError, match="out of range"):
        instance_contribution_rows(shap_explanation, 999)
    with pytest.raises(ValueError, match="out of range"):
        instance_contribution_rows(shap_explanation, -1)


def test_non_numeric_contribution_raises_clearly():
    explanation = {
        "method": "shap",
        "per_instance": [{"row_index": 0, "contributions": {"a": "big"}}],
        "global_importance": {"a": 1.0},
        "is_mock": False,
    }
    with pytest.raises(ValueError, match="must be a number"):
        instance_contribution_rows(explanation, 0)


# ---------------------------------------------------------------------------
# 10 — works with no live API
# ---------------------------------------------------------------------------


def test_panel_data_builds_without_a_live_api():
    """The helper is pure: no network, no API, no Streamlit."""
    explanation = _synthetic_explanation()

    assert global_importance_rows(explanation)
    assert instance_contribution_rows(explanation, 0)
    assert instance_options(explanation)
    assert full_contribution_rows(explanation)


def test_panel_module_imports_without_dashboard_integration():
    """The panel must import even though panels/__init__.py (Khushi) is absent."""
    from dashboard.panels import explainability_panel

    assert callable(explainability_panel.render)


def test_helper_works_on_the_mock_fallback_fixture():
    """The dashboard falls back to mock data when the API is down."""
    from app.api.mock_data import MOCK_EXPLAINABILITY_RESULT_SHAP as mock

    rows = global_importance_rows(mock)
    assert {r["feature"]: r["importance"] for r in rows} == mock["global_importance"]
    assert instance_contribution_rows(mock, 0)


# ---------------------------------------------------------------------------
# 12, 13 — instance identity
# ---------------------------------------------------------------------------


def test_instance_ids_are_used_when_supplied(shap_explanation, real_instance_ids):
    options = instance_options(shap_explanation, real_instance_ids)

    assert [o["label"] for o in options] == [str(i) for i in real_instance_ids]
    assert [o["instance_id"] for o in options] == real_instance_ids
    assert all(o["is_identity"] for o in options)


def test_positional_selection_does_not_fabricate_identity(shap_explanation):
    """With no ids supplied, nothing may look like an applicant identifier."""
    options = instance_options(shap_explanation)

    assert all(o["instance_id"] is None for o in options)
    assert all(o["is_identity"] is False for o in options)
    for option in options:
        assert "position in explained set" in option["label"], (
            "a position must be labelled as a position, never as an identity"
        )


def test_mismatched_instance_ids_raise_rather_than_truncate(shap_explanation):
    """Trimming ids to fit is how one record gets another record's label."""
    with pytest.raises(ValueError, match="does not match the explained rows"):
        instance_options(shap_explanation, ["only-one-id"])

    too_many = [f"id-{i}" for i in range(EXPLAINED_ROWS + 5)]
    with pytest.raises(ValueError, match="does not match the explained rows"):
        instance_options(shap_explanation, too_many)


def test_full_matrix_carries_identity_when_available(
    shap_explanation, real_instance_ids
):
    rows = full_contribution_rows(shap_explanation, real_instance_ids)
    assert {r["instance_id"] for r in rows} == set(real_instance_ids)

    anonymous = full_contribution_rows(shap_explanation)
    assert all(r["instance_id"] is None for r in anonymous)


# ---------------------------------------------------------------------------
# 14 — output is chart/table/JSON safe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_output_is_json_serializable(method, request, real_instance_ids):
    explanation = request.getfixturevalue(f"{method}_explanation")

    payloads = [
        global_importance_rows(explanation),
        instance_contribution_rows(explanation, 0, top_n=5),
        instance_options(explanation, real_instance_ids),
        instance_options(explanation),
        full_contribution_rows(explanation, real_instance_ids),
    ]
    for payload in payloads:
        assert json.loads(json.dumps(payload)) == payload


def test_numbers_are_builtin_floats_not_numpy(shap_explanation):
    """DataFrame- and JSON-safe: no numpy scalars leak through."""
    for row in global_importance_rows(shap_explanation):
        assert type(row["importance"]) is float
    for row in instance_contribution_rows(shap_explanation, 0):
        assert type(row["contribution"]) is float
        assert type(row["abs_contribution"]) is float
