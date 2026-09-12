"""Phase 4 tests: model result presentation panel (owner: Namitha).

Covers presentation only. This panel does not train, predict, score, or
evaluate -- these tests verify it renders whatever ``ModelResult``-shaped dict
it is given, faithfully, including the empty, partial and mock states, and that
it never derives an analytical metric of its own.

Uses the ``streamlit.testing.v1.AppTest`` convention established by the other
Phase 4 panel tests, applied to the panel's render function directly via
``AppTest.from_function`` -- the panel is a plain function over already-fetched
data, with no HTTP call and no Streamlit page of its own.
"""
import ast
import inspect

import pytest
from streamlit.testing.v1 import AppTest

from app.api.mock_data import MOCK_MODEL_RESULT


def _run(data, source: str = "api") -> AppTest:
    """Render the panel through AppTest and return the finished app."""

    def _wrapper(data, source):
        from dashboard.panels.model_panel import render_model_panel

        render_model_panel(data=data, source=source)

    at = AppTest.from_function(_wrapper, kwargs={"data": data, "source": source})
    at.run(timeout=30)
    return at


def _all_text(at: AppTest) -> str:
    """Every rendered text-ish element, flattened for substring assertions."""
    parts = []
    for block in (
        at.markdown,
        at.caption,
        at.info,
        at.warning,
        at.error,
        at.subheader,
        at.text,
    ):
        parts.extend(str(element.value) for element in block)
    for metric in at.metric:
        parts.append(f"{metric.label} {metric.value}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Realistic rendering
# ---------------------------------------------------------------------------


def test_realistic_model_result_renders_without_exception():
    at = _run(MOCK_MODEL_RESULT)
    assert not at.exception


def test_renders_the_expected_sections():
    at = _run(MOCK_MODEL_RESULT)
    headings = [element.value for element in at.subheader]

    assert "Model metadata" in headings
    assert "Model performance (held-out test set)" in headings
    assert "Prediction distribution" in headings
    assert "Probability distribution" in headings
    assert "Prediction details" in headings


def test_source_and_is_mock_are_surfaced():
    at = _run(MOCK_MODEL_RESULT, source="fallback")
    text = _all_text(at)
    assert "fallback" in text
    assert "Mock data" in text


# ---------------------------------------------------------------------------
# P4-02 — model_metrics are displayed
# ---------------------------------------------------------------------------


def test_model_metrics_are_displayed_when_present():
    """Every contracted metric field must reach the screen."""
    at = _run(MOCK_MODEL_RESULT)
    labels = " ".join(metric.label for metric in at.metric).upper()

    for field in ("ACCURACY", "PRECISION", "RECALL", "F1", "ROC AUC"):
        assert field in labels, f"{field} was not displayed"


def test_metric_values_are_displayed_unmodified():
    """Displayed metrics must equal the API's values, not a recomputation."""
    metrics = MOCK_MODEL_RESULT["model_metrics"]
    at = _run(MOCK_MODEL_RESULT)
    shown = {metric.label.upper(): metric.value for metric in at.metric}

    for field in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        label = field.replace("_", " ").upper()
        assert shown[label] == f"{metrics[field]:.4f}"


def test_n_test_samples_is_reported():
    at = _run(MOCK_MODEL_RESULT)
    assert str(MOCK_MODEL_RESULT["model_metrics"]["n_test_samples"]) in _all_text(at)


def test_mock_metrics_are_flagged():
    """A mock evaluation must not be readable as a real one."""
    data = {
        **MOCK_MODEL_RESULT,
        "model_metrics": {**MOCK_MODEL_RESULT["model_metrics"], "is_mock": True},
    }
    at = _run(data)
    assert any("MOCK" in str(w.value).upper() for w in at.warning)


def test_missing_metrics_are_handled_gracefully():
    """Absent metrics must explain themselves, not crash or fabricate values."""
    data = {k: v for k, v in MOCK_MODEL_RESULT.items() if k != "model_metrics"}
    at = _run(data)

    assert not at.exception
    text = _all_text(at)
    assert "No model metrics" in text
    assert "never computed in the dashboard" in text


# ---------------------------------------------------------------------------
# P4-01 — distributions are visualised
# ---------------------------------------------------------------------------


def test_two_distribution_charts_are_rendered():
    """Prediction and probability distributions must be charts, not tables."""
    at = _run(MOCK_MODEL_RESULT)
    assert len(at.get("vega_lite_chart")) >= 2


def test_prediction_distribution_counts_are_faithful():
    """The chart's counts must match the supplied predictions exactly."""
    from dashboard.panels.model_panel import _class_label

    predictions = MOCK_MODEL_RESULT["predictions"]
    semantics = MOCK_MODEL_RESULT["model_metadata"].get("label_semantics")
    expected = {}
    for value in predictions:
        expected[_class_label(value, semantics)] = expected.get(
            _class_label(value, semantics), 0
        ) + 1

    at = _run(MOCK_MODEL_RESULT)
    rendered = at.dataframe[0].value
    actual = {row["class"]: row["count"] for _, row in rendered.iterrows()}

    assert actual == expected
    assert sum(actual.values()) == len(predictions)


def test_probability_bands_cover_every_probability():
    """No probability may be dropped by the display banding, 1.0 included."""
    data = {
        **MOCK_MODEL_RESULT,
        "probabilities": [0.0, 0.05, 0.5, 0.99, 1.0],
        "predictions": [0, 0, 1, 1, 1],
        "instance_ids": ["a", "b", "c", "d", "e"],
    }
    at = _run(data)
    assert not at.exception
    assert "5 predicted probabilities" in _all_text(at)


def test_class_labels_come_from_label_semantics_not_hardcoded():
    """Class wording must follow the model module, not a dashboard constant."""
    data = {
        **MOCK_MODEL_RESULT,
        "model_metadata": {
            **MOCK_MODEL_RESULT["model_metadata"],
            "label_semantics": {"0": "CUSTOM-GOOD", "1": "CUSTOM-BAD"},
        },
    }
    at = _run(data)
    rendered = at.dataframe[0].value
    labels = " ".join(str(v) for v in rendered["class"])

    assert "CUSTOM-GOOD" in labels or "CUSTOM-BAD" in labels


def test_absent_label_semantics_does_not_invent_meaning():
    metadata = {
        k: v
        for k, v in MOCK_MODEL_RESULT["model_metadata"].items()
        if k != "label_semantics"
    }
    at = _run({**MOCK_MODEL_RESULT, "model_metadata": metadata})

    assert not at.exception
    rendered = at.dataframe[0].value
    labels = " ".join(str(v) for v in rendered["class"])
    assert "GOOD" not in labels.upper() and "BAD" not in labels.upper()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _prediction_table(at: AppTest):
    """The per-record table, selected by its columns rather than by position."""
    for element in at.dataframe:
        if "record" in element.value.columns:
            return element.value
    raise AssertionError("prediction details table was not rendered")


def test_instance_ids_are_used_when_available():
    at = _run(MOCK_MODEL_RESULT)
    assert "stable `instance_id`" in _all_text(at)

    table = _prediction_table(at)
    assert MOCK_MODEL_RESULT["instance_ids"][0] in list(table["record"].astype(str))


def test_missing_instance_ids_fall_back_to_labelled_positions():
    """A position must be labelled a position, never presented as identity."""
    data = {k: v for k, v in MOCK_MODEL_RESULT.items() if k != "instance_ids"}
    at = _run(data)
    text = _all_text(at)

    assert "labelled by POSITION" in text
    assert "not an applicant identity" in text


def test_mismatched_instance_ids_are_not_silently_reused():
    """Truncating an id list is how one record acquires another's identity."""
    data = {**MOCK_MODEL_RESULT, "instance_ids": ["only-one"]}
    at = _run(data)

    assert not at.exception
    assert "labelled by POSITION" in _all_text(at)


# ---------------------------------------------------------------------------
# Degraded inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [{}, None])
def test_empty_payload_renders_without_exception(payload):
    at = _run(payload)
    assert not at.exception
    assert "No predictions were returned." in _all_text(at)


def test_missing_predictions_handled():
    data = {k: v for k, v in MOCK_MODEL_RESULT.items() if k != "predictions"}
    at = _run(data)
    assert not at.exception
    assert "No predictions were returned." in _all_text(at)


def test_non_numeric_probabilities_handled():
    data = {**MOCK_MODEL_RESULT, "probabilities": ["n/a"] * len(MOCK_MODEL_RESULT["predictions"])}
    at = _run(data)
    assert not at.exception
    assert "no numeric values" in _all_text(at)


def test_empty_feature_matrix_handled():
    at = _run({**MOCK_MODEL_RESULT, "feature_matrix": []})
    assert not at.exception


# ---------------------------------------------------------------------------
# No recalculation (the Phase 4 binding constraint)
# ---------------------------------------------------------------------------


def test_panel_does_not_import_analytical_modules():
    """Presentation must not reach into the model or any analytical package."""
    import dashboard.panels.model_panel as panel

    tree = ast.parse(inspect.getsource(panel))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("app.models", "sklearn", "numpy", "app.fairness", "app.drift")
    for name in imported:
        assert not name.startswith(forbidden), f"panel imports {name}"


def test_panel_source_contains_no_metric_arithmetic():
    """Guard against a future edit deriving accuracy/F1 in the dashboard."""
    import dashboard.panels.model_panel as panel

    source = inspect.getsource(panel).lower()
    for banned in ("accuracy_score", "precision_score", "recall_score", "f1_score", "roc_auc_score"):
        assert banned not in source
