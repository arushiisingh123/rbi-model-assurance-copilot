"""Phase 4 tests: fairness and drift presentation panel (owner: Arushi).

Covers presentation only. This panel does not calculate a fairness metric, a
PSI, or a KS statistic, and does not classify anything -- these tests verify it
renders whatever ``FairnessDriftResult``-shaped dict it is given, faithfully,
including the empty, PENDING, mock, and synthetic-drift states.

Uses the ``streamlit.testing.v1.AppTest`` convention established in
``tests/dashboard/test_dashboard_entrypoint.py`` and followed by
``tests/dashboard/test_compliance_panel.py``, applied to the panel's render
function directly via ``AppTest.from_function`` -- the panel is a plain function
over already-fetched data, with no HTTP call and no Streamlit page of its own.
"""
import ast
import inspect

import pytest
from streamlit.testing.v1 import AppTest

from app.api.mock_data import MOCK_DRIFT_RESULT, MOCK_FAIRNESS_RESULT

REALISTIC_PAYLOAD = {
    "fairness": MOCK_FAIRNESS_RESULT,
    "drift": MOCK_DRIFT_RESULT,
    "note": None,
}


def _run(data: dict, source: str = "api") -> AppTest:
    """Render the panel through AppTest and return the finished app."""

    def _wrapper(data, source):
        from dashboard.panels.fairness_drift_panel import (
            render_fairness_drift_panel,
        )
        render_fairness_drift_panel(data=data, source=source)

    at = AppTest.from_function(_wrapper, kwargs={"data": data, "source": source})
    at.run(timeout=15)
    return at


def _all_text(at: AppTest) -> str:
    """Every text surface the panel writes to, joined for substring checks."""
    parts = []
    for collection in (at.markdown, at.caption, at.info, at.warning, at.subheader):
        parts.extend(element.value for element in collection)
    for metric in at.metric:
        parts.append(f"{metric.label} {metric.value}")
    return "\n".join(parts)


def _chart_frames(at: AppTest):
    """The dataframes handed to st.bar_chart, in render order.

    ``AppTest`` exposes charts as ``vega_lite_chart`` elements with no public
    value accessor in Streamlit 1.62, so the plotted data is read back from the
    element's attached Arrow dataset. Kept in one helper so the whole suite
    changes in one place if that internal shape moves.

    Note the frame contains only the encoded columns (the ``x`` and ``y``
    fields), not every field of the source row.
    """
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

    frames = []
    for element in at.get("vega_lite_chart"):
        dataset = element.proto.datasets[0]
        frames.append(convert_arrow_bytes_to_pandas_df(dataset.data.data))
    return frames


def _chart_specs(at: AppTest):
    """The Vega-Lite spec of each chart, in render order."""
    import json

    return [json.loads(element.proto.spec) for element in at.get("vega_lite_chart")]


def _chart_encodings(at: AppTest):
    """The (x field, y field) each chart actually plots, in render order."""
    encodings = []
    for spec in _chart_specs(at):
        encoding = spec.get("encoding", {})
        encodings.append(
            (encoding.get("x", {}).get("field"), encoding.get("y", {}).get("field"))
        )
    return encodings


def _chart_x_encodings(at: AppTest):
    """The full x encoding of each chart -- field and the axis title shown."""
    return [spec.get("encoding", {}).get("x", {}) for spec in _chart_specs(at)]


# ---------------------------------------------------------------------------
# Realistic rendering
# ---------------------------------------------------------------------------


def test_realistic_payload_renders_without_exception():
    at = _run(REALISTIC_PAYLOAD)
    assert not at.exception, f"Panel raised: {at.exception}"


def test_both_domains_are_rendered():
    at = _run(REALISTIC_PAYLOAD)
    headings = [s.value for s in at.subheader]

    assert "Fairness Evaluation" in headings
    assert "Drift Detection" in headings


def test_data_source_is_shown_for_both_domains():
    at = _run(REALISTIC_PAYLOAD, source="api")
    captions = [c.value for c in at.caption]

    assert sum("Data source: **api**" in c for c in captions) == 2


# ---------------------------------------------------------------------------
# Status rendering goes through the shared helper
# ---------------------------------------------------------------------------


def test_fairness_status_is_rendered_through_render_status():
    from dashboard.api_client import render_status

    at = _run(REALISTIC_PAYLOAD)
    text = _all_text(at)
    expected = render_status(MOCK_FAIRNESS_RESULT["status"])

    assert f"**Fairness Status:** {expected}" in text


def test_drift_status_is_rendered_through_render_status():
    from dashboard.api_client import render_status

    at = _run(REALISTIC_PAYLOAD)
    text = _all_text(at)
    expected = render_status(MOCK_DRIFT_RESULT["status"])

    assert f"**Drift Status:** {expected}" in text


def test_only_the_approved_status_vocabulary_appears():
    at = _run(REALISTIC_PAYLOAD)
    text = _all_text(at)

    for forbidden in ("OK", "GOOD", "BAD", "ALERT", "CRITICAL", "SEVERE"):
        assert f"**Fairness Status:** {forbidden}" not in text
        assert f"**Drift Status:** {forbidden}" not in text
    assert any(tag in text for tag in ("PASS", "WARNING", "FAIL", "PENDING"))


def test_unknown_status_is_not_silently_normalised():
    payload = {
        "fairness": {**MOCK_FAIRNESS_RESULT, "status": "SOMETHING_UNEXPECTED"},
        "drift": MOCK_DRIFT_RESULT,
    }
    at = _run(payload)

    assert not at.exception
    assert "SOMETHING_UNEXPECTED" in _all_text(at)


# ---------------------------------------------------------------------------
# Fairness values are visible
# ---------------------------------------------------------------------------


def test_protected_attribute_and_aggregate_metrics_are_visible():
    at = _run(REALISTIC_PAYLOAD)
    text = _all_text(at)

    assert MOCK_FAIRNESS_RESULT["protected_attribute"] in text
    assert str(MOCK_FAIRNESS_RESULT["demographic_parity_diff"]) in text
    assert str(MOCK_FAIRNESS_RESULT["disparate_impact_ratio"]) in text


def test_every_fairness_group_and_its_values_are_visible():
    at = _run(REALISTIC_PAYLOAD)
    table = at.dataframe[0].value.to_dict("records")

    assert len(table) == len(MOCK_FAIRNESS_RESULT["groups"])
    for rendered, supplied in zip(table, MOCK_FAIRNESS_RESULT["groups"]):
        assert rendered["group"] == supplied["group"]
        assert rendered["count"] == supplied["count"]
        assert rendered["favorable_count"] == supplied["favorable_count"]
        assert rendered["selection_rate"] == supplied["selection_rate"]


def test_fairness_group_table_columns_are_exactly_the_agreed_fields():
    at = _run(REALISTIC_PAYLOAD)
    columns = list(at.dataframe[0].value.columns)

    assert columns == ["group", "count", "favorable_count", "selection_rate"]


def test_fairness_group_order_is_preserved_from_the_payload():
    at = _run(REALISTIC_PAYLOAD)
    rendered = list(at.dataframe[0].value["group"])

    assert rendered == [g["group"] for g in MOCK_FAIRNESS_RESULT["groups"]]


def test_attribute_9_group_labels_are_not_relabelled_as_gender():
    """Attribute 9 combines marital status and sex; it is never shown as gender.

    Checks the chart axis as well as the text. A relabelled axis title is the
    most likely way this slips in, and axis titles live in the Vega-Lite spec
    rather than in any of the text surfaces ``_all_text`` collects -- so a
    text-only assertion would not catch it.
    """
    at = _run(REALISTIC_PAYLOAD)
    text = _all_text(at).lower()

    assert "personal_status_and_sex" in text
    for forbidden in ("gender:", "sex:", "male group", "female group"):
        assert forbidden not in text

    x_encoding = _chart_x_encodings(at)[0]
    assert x_encoding["field"] == "group"
    assert x_encoding["title"] == "Observed group"
    for forbidden in ("gender", "sex", "male", "female"):
        assert forbidden not in str(x_encoding["title"]).lower()


# ---------------------------------------------------------------------------
# Drift values are visible
# ---------------------------------------------------------------------------


def test_aggregate_psi_and_ks_are_visible():
    at = _run(REALISTIC_PAYLOAD)
    text = _all_text(at)

    assert str(MOCK_DRIFT_RESULT["psi"]) in text
    assert str(MOCK_DRIFT_RESULT["ks_statistic"]) in text


def test_features_evaluated_are_visible():
    at = _run(REALISTIC_PAYLOAD)
    text = _all_text(at)

    for feature in MOCK_DRIFT_RESULT["features_evaluated"]:
        assert feature in text


def test_every_per_feature_drift_value_is_visible():
    at = _run(REALISTIC_PAYLOAD)
    # dataframe[0] is the fairness groups table; dataframe[1] is per-feature.
    table = at.dataframe[1].value.to_dict("records")

    assert len(table) == len(MOCK_DRIFT_RESULT["per_feature"])
    for rendered, supplied in zip(table, MOCK_DRIFT_RESULT["per_feature"]):
        assert rendered["feature"] == supplied["feature"]
        assert rendered["psi"] == supplied["psi"]
        assert rendered["ks_statistic"] == supplied["ks_statistic"]


def test_per_feature_table_columns_are_exactly_the_agreed_fields():
    at = _run(REALISTIC_PAYLOAD)
    columns = list(at.dataframe[1].value.columns)

    assert columns == ["feature", "psi", "ks_statistic"]


def test_per_feature_order_is_preserved_from_the_payload():
    at = _run(REALISTIC_PAYLOAD)
    rendered = list(at.dataframe[1].value["feature"])

    assert rendered == [f["feature"] for f in MOCK_DRIFT_RESULT["per_feature"]]


def test_no_per_feature_status_is_invented():
    at = _run(REALISTIC_PAYLOAD)
    columns = list(at.dataframe[1].value.columns)

    assert "status" not in columns
    assert "threshold" not in columns


# ---------------------------------------------------------------------------
# Charts visualise the supplied values only
# ---------------------------------------------------------------------------


def test_three_charts_are_rendered_plotting_the_expected_fields():
    at = _run(REALISTIC_PAYLOAD)

    # Selection rate, per-feature PSI, per-feature KS -- PSI and KS are charted
    # separately because their maxima are evaluated independently.
    assert _chart_encodings(at) == [
        ("group", "selection_rate"),
        ("feature", "psi"),
        ("feature", "ks_statistic"),
    ]


def test_selection_rate_chart_uses_the_supplied_rates():
    at = _run(REALISTIC_PAYLOAD)
    chart = _chart_frames(at)[0]

    assert list(chart["group"]) == [g["group"] for g in MOCK_FAIRNESS_RESULT["groups"]]
    assert list(chart["selection_rate"]) == [
        g["selection_rate"] for g in MOCK_FAIRNESS_RESULT["groups"]
    ]


def test_psi_and_ks_charts_use_the_supplied_per_feature_values():
    at = _run(REALISTIC_PAYLOAD)
    psi_chart, ks_chart = _chart_frames(at)[1], _chart_frames(at)[2]
    supplied = MOCK_DRIFT_RESULT["per_feature"]

    assert list(psi_chart["feature"]) == [f["feature"] for f in supplied]
    assert list(psi_chart["psi"]) == [f["psi"] for f in supplied]
    assert list(ks_chart["ks_statistic"]) == [f["ks_statistic"] for f in supplied]


# ---------------------------------------------------------------------------
# is_mock visibility
# ---------------------------------------------------------------------------


def test_is_mock_true_is_visible_for_both_domains():
    payload = {
        "fairness": {**MOCK_FAIRNESS_RESULT, "is_mock": True},
        "drift": {**MOCK_DRIFT_RESULT, "is_mock": True},
    }
    captions = [c.value for c in _run(payload).caption]

    assert sum("Mock data: **True**" in c for c in captions) == 2


def test_is_mock_false_is_visible_and_not_hidden():
    payload = {
        "fairness": {**MOCK_FAIRNESS_RESULT, "is_mock": False},
        "drift": {**MOCK_DRIFT_RESULT, "is_mock": False, "note": None},
    }
    captions = [c.value for c in _run(payload).caption]

    assert sum("Mock data: **False**" in c for c in captions) == 2


# ---------------------------------------------------------------------------
# Synthetic / fallback provenance is explicit
# ---------------------------------------------------------------------------


def test_synthetic_drift_note_is_surfaced_and_labelled_as_synthetic():
    at = _run(REALISTIC_PAYLOAD)
    warnings = "\n".join(w.value for w in at.warning)

    assert "SYNTHETIC DRIFT SCENARIO" in warnings
    assert "not observed production drift" in warnings
    assert MOCK_DRIFT_RESULT["note"] in warnings


def test_fallback_source_is_flagged_as_not_a_live_measurement():
    at = _run(REALISTIC_PAYLOAD, source="fallback")
    warnings = "\n".join(w.value for w in at.warning)

    assert "mock or fallback data" in warnings
    assert "must not be presented as observed drift" in warnings


def test_real_observed_drift_is_not_labelled_synthetic():
    """A real train/test result carries no synthetic claim."""
    payload = {
        "fairness": {**MOCK_FAIRNESS_RESULT, "is_mock": False},
        "drift": {**MOCK_DRIFT_RESULT, "is_mock": False, "note": None},
    }
    at = _run(payload, source="api")
    text = _all_text(at)

    assert "SYNTHETIC DRIFT SCENARIO" not in text
    # The comparison being made is still stated, so it cannot be mistaken for
    # production monitoring.
    assert "held-out test split" in text


def test_title_case_synthetic_note_is_warned_not_merely_informed():
    """Regression: synthetic labelling must not depend on capitalisation.

    A producer writing "Synthetic drift scenario" rather than shouting
    "SYNTHETIC" previously fell through to an ordinary ``st.info``, leaving
    synthetic drift presentable as observed drift. The check is now
    case-insensitive.
    """
    payload = {
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": {
            **MOCK_DRIFT_RESULT,
            "is_mock": False,
            "note": "Synthetic drift scenario for demo.",
        },
    }
    at = _run(payload, source="api")
    warnings_text = "\n".join(w.value for w in at.warning)
    infos_text = "\n".join(i.value for i in at.info)

    assert "SYNTHETIC DRIFT SCENARIO" in warnings_text
    assert "not observed production drift" in warnings_text
    assert "Synthetic drift scenario for demo." in warnings_text
    # It must be a warning, not a neutral informational message.
    assert "Synthetic drift scenario for demo." not in infos_text


@pytest.mark.parametrize(
    "note",
    [
        "SYNTHETIC DRIFT SCENARIO: all caps.",
        "Synthetic drift scenario: title case.",
        "synthetic drift scenario: lower case.",
        "A sYnThEtIc mixture.",
    ],
)
def test_synthetic_detection_is_case_insensitive(note):
    payload = {
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": {**MOCK_DRIFT_RESULT, "is_mock": False, "note": note},
    }
    at = _run(payload, source="api")
    warnings_text = "\n".join(w.value for w in at.warning)

    assert "SYNTHETIC DRIFT SCENARIO" in warnings_text
    assert note in warnings_text


def test_non_synthetic_note_is_informational_not_a_synthetic_warning():
    """A note that says nothing about synthetic data must not be mislabelled."""
    payload = {
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": {
            **MOCK_DRIFT_RESULT,
            "is_mock": False,
            "note": "Reference is the development training split.",
        },
    }
    at = _run(payload, source="api")
    infos_text = "\n".join(i.value for i in at.info)
    warnings_text = "\n".join(w.value for w in at.warning)

    assert "Reference is the development training split." in infos_text
    assert "SYNTHETIC DRIFT SCENARIO" not in warnings_text


def test_payload_level_note_is_also_surfaced():
    payload = {
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": {**MOCK_DRIFT_RESULT, "note": None},
        "note": "SYNTHETIC DRIFT SCENARIO: payload-level note.",
    }
    at = _run(payload)
    warnings = "\n".join(w.value for w in at.warning)

    assert "payload-level note" in warnings


# ---------------------------------------------------------------------------
# Empty and PENDING states, rendered gracefully and honestly
# ---------------------------------------------------------------------------


def test_empty_fairness_groups_render_gracefully():
    payload = {
        "fairness": {**MOCK_FAIRNESS_RESULT, "groups": []},
        "drift": MOCK_DRIFT_RESULT,
    }
    at = _run(payload)
    infos = "\n".join(i.value for i in at.info)

    assert not at.exception
    assert "No per-group fairness breakdown" in infos
    # Aggregates are still shown; only the breakdown is absent.
    assert str(MOCK_FAIRNESS_RESULT["disparate_impact_ratio"]) in _all_text(at)


def test_missing_fairness_groups_key_renders_gracefully():
    fairness = {k: v for k, v in MOCK_FAIRNESS_RESULT.items() if k != "groups"}
    at = _run({"fairness": fairness, "drift": MOCK_DRIFT_RESULT})

    assert not at.exception
    assert any("No per-group fairness breakdown" in i.value for i in at.info)


def test_empty_per_feature_renders_gracefully():
    payload = {
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": {**MOCK_DRIFT_RESULT, "per_feature": []},
    }
    at = _run(payload)
    infos = "\n".join(i.value for i in at.info)

    assert not at.exception
    assert "No per-feature drift breakdown" in infos
    assert str(MOCK_DRIFT_RESULT["psi"]) in _all_text(at)


def test_missing_per_feature_key_renders_gracefully():
    drift = {k: v for k, v in MOCK_DRIFT_RESULT.items() if k != "per_feature"}
    at = _run({"fairness": MOCK_FAIRNESS_RESULT, "drift": drift})

    assert not at.exception
    assert any("No per-feature drift breakdown" in i.value for i in at.info)


def test_pending_fairness_renders_without_inventing_a_conclusion():
    """PENDING keeps its observed groups and claims neither pass nor failure."""
    payload = {
        "fairness": {
            "protected_attribute": "personal_status_and_sex",
            "demographic_parity_diff": 0.0,
            "disparate_impact_ratio": 1.0,
            "status": "PENDING",
            "is_mock": False,
            "groups": [
                {
                    "group": "A91",
                    "count": 3,
                    "favorable_count": 0,
                    "selection_rate": 0.0,
                }
            ],
        },
        "drift": MOCK_DRIFT_RESULT,
    }
    at = _run(payload)
    text = _all_text(at)

    assert not at.exception
    assert "PENDING" in text
    assert "neither a pass nor a failure" in text
    # The observed group survives into the table.
    assert list(at.dataframe[0].value["group"]) == ["A91"]


def test_pending_drift_renders_without_inventing_a_conclusion():
    payload = {
        "fairness": MOCK_FAIRNESS_RESULT,
        "drift": {
            "features_evaluated": [],
            "psi": 0.0,
            "ks_statistic": 0.0,
            "status": "PENDING",
            "is_mock": False,
            "per_feature": [],
        },
    }
    at = _run(payload)
    text = _all_text(at)

    assert not at.exception
    assert "no feature could be evaluated" in text
    assert "neither a pass nor a failure" in text


def test_wholly_empty_payload_renders_gracefully():
    at = _run({})

    assert not at.exception
    assert "not reported" in _all_text(at)


def test_missing_drift_is_mock_is_presented_consistently():
    """The caption and the warning must agree about the mock state.

    A drift payload with no ``is_mock`` key is captioned as mock (the cautious
    default). The matching warning must then also appear -- claiming the data is
    mock while withholding the warning would understate the caveat.
    """
    at = _run({"fairness": MOCK_FAIRNESS_RESULT, "drift": {}}, source="api")
    captions = "\n".join(c.value for c in at.caption)
    warnings_text = "\n".join(w.value for w in at.warning)

    assert not at.exception
    assert "Mock data: **True**" in captions
    assert "mock or fallback data" in warnings_text


def test_real_api_payload_is_not_warned_as_mock():
    """The fix must not make a genuine observed result look like mock data."""
    payload = {
        "fairness": {**MOCK_FAIRNESS_RESULT, "is_mock": False},
        "drift": {**MOCK_DRIFT_RESULT, "is_mock": False, "note": None},
    }
    at = _run(payload, source="api")
    warnings_text = "\n".join(w.value for w in at.warning)

    assert "mock or fallback data" not in warnings_text


# ---------------------------------------------------------------------------
# No fetching, no recalculation (source inspection)
# ---------------------------------------------------------------------------


def _panel_source() -> str:
    import dashboard.panels.fairness_drift_panel as panel_module

    return inspect.getsource(panel_module)


def test_panel_makes_no_http_calls_itself():
    """Data fetching lives outside the panel (dashboard/api_client.py)."""
    source = _panel_source()

    assert "requests." not in source
    assert "import requests" not in source
    assert "httpx" not in source
    assert "urllib" not in source


def test_panel_does_not_import_analytical_modules():
    """The panel presents results; it must not be able to derive one.

    Asserted by import inspection: if the module never imports the fairness or
    drift modules, the threshold config, RAG, or the report layer, it cannot
    recompute a metric or reclassify a status regardless of its internals.
    """
    import dashboard.panels.fairness_drift_panel as panel_module

    tree = ast.parse(inspect.getsource(panel_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    forbidden_prefixes = (
        "app.fairness",
        "app.drift",
        "app.config",
        "app.rag",
        "app.report",
        "app.models",
        "app.compliance",
    )
    offending = [
        name
        for name in imported
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in forbidden_prefixes
        )
    ]
    assert not offending, f"panel must not import analytical modules: {offending}"


def test_panel_defines_no_threshold_and_performs_no_metric_arithmetic():
    """No threshold constant, and no arithmetic on the reported values.

    Inspects the code rather than the text: the docstring legitimately mentions
    example numbers when explaining why the disparate impact ratio must not be
    re-derived, but no comparison or division may appear in the code itself.
    """
    import dashboard.panels.fairness_drift_panel as panel_module

    assert not [
        name for name in dir(panel_module) if "THRESHOLD" in name.upper()
    ]

    tree = ast.parse(inspect.getsource(panel_module))

    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and "THRESHOLD" in target.id.upper()
    ], "panel must not define a threshold constant"

    # Division and subtraction are how a selection rate, a parity difference,
    # or a disparate impact ratio would be re-derived. Neither belongs here.
    arithmetic = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Div, ast.Sub, ast.Mult, ast.FloorDiv))
    ]
    assert not arithmetic, "panel must not perform metric arithmetic"


def test_panel_does_not_classify_against_the_psi_or_di_bands():
    """No band edge may be compared against in code."""
    import dashboard.panels.fairness_drift_panel as panel_module

    tree = ast.parse(inspect.getsource(panel_module))
    band_edges = {0.10, 0.25, 0.70, 0.80}

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
        "panel must not classify against a threshold band; classification "
        "belongs to the analytical modules"
    )
