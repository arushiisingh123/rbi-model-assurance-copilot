"""Monitoring dashboard panel and client (owner: Arushi).

Covers rendering only. The panel calculates nothing -- these tests verify it
renders whatever ``MonitoringAssuranceResult``-shaped payload it is given,
faithfully, including the unavailable, PENDING and undeclared-attribute states.

Uses the ``streamlit.testing.v1.AppTest`` convention established by the other
panel tests, applied to the render function directly: the panel is a plain
function over already-fetched data, with no HTTP call of its own.
"""
import requests
from streamlit.testing.v1 import AppTest

from dashboard.monitoring_client import (
    SOURCE_UNAVAILABLE,
    UNAVAILABLE_PAYLOAD,
    get_monitoring,
)
from tests.dashboard.test_monitoring_presentation import payload


def _run(data, source: str = "API") -> AppTest:
    def _wrapper(data, source):
        from dashboard.panels.monitoring_panel import render_monitoring_panel

        render_monitoring_panel(data=data, source=source)

    at = AppTest.from_function(_wrapper, kwargs={"data": data, "source": source})
    at.run(timeout=30)
    return at


def _all_text(at: AppTest) -> str:
    parts = []
    for collection in (
        at.markdown,
        at.caption,
        at.info,
        at.warning,
        at.success,
        at.subheader,
        at.header,
    ):
        parts.extend(element.value for element in collection)
    for metric in at.metric:
        parts.append(f"{metric.label} {metric.value}")
    return "\n".join(parts)


def _table_cells(at: AppTest) -> set:
    """Every rendered table cell, as strings.

    Reads the DataFrames' actual values rather than ``str(df)``: pandas elides
    middle columns from its string repr with "...", which would silently make a
    substring assertion pass or fail for the wrong reason.
    """
    cells = set()
    for table in at.table:
        frame = table.value
        for column in frame.columns:
            cells.update(str(value) for value in frame[column].tolist())
    return cells


# ---------------------------------------------------------------------------
# Rendering the full result
# ---------------------------------------------------------------------------


def test_the_panel_renders_a_full_result_without_error():
    at = _run(payload())

    assert not at.exception
    text = _all_text(at)
    assert "Monitoring" in text
    assert "Data source: API" in text


def test_the_headline_status_and_windows_are_surfaced():
    at = _run(payload())
    text = _all_text(at)

    assert "FAIL" in text
    assert "2026-Q2" in text and "2026-Q3" in text


def test_model_identity_is_surfaced():
    at = _run(payload())

    cells = _table_cells(at)
    assert "some-model" in cells
    assert "run-1" in cells


def test_both_prediction_drift_channels_are_shown_separately():
    at = _run(payload())
    cells = _table_cells(at)

    assert "Label (categorical PSI)" in cells
    assert "Score (quantile PSI)" in cells


def test_unstated_provenance_is_never_rendered_as_observed():
    """The current window's provenance is None in the fixture."""
    at = _run(payload())
    cells = _table_cells(at)

    assert "not stated" in cells
    # And the stated one is still shown as exactly what it is.
    assert "observed" in cells


def test_the_panel_does_not_recalculate_any_metric():
    """No analytical import may appear in the panel module."""
    from pathlib import Path

    source = Path("dashboard/panels/monitoring_panel.py").read_text(encoding="utf-8")
    for forbidden in (
        "app.config.thresholds",
        "app.drift",
        "app.fairness",
        "app.monitoring",
        "classify_",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Degraded and unavailable states
# ---------------------------------------------------------------------------


def test_an_empty_payload_renders_an_explicit_no_data_state():
    at = _run(dict(UNAVAILABLE_PAYLOAD), source=SOURCE_UNAVAILABLE)

    assert not at.exception
    text = _all_text(at)
    assert "No monitoring result available" in text
    assert SOURCE_UNAVAILABLE in text


def test_an_undeclared_protected_attribute_is_explained_not_hidden():
    data = payload()
    data["result"]["fairness"] = None
    data["result"]["channel_status"]["fairness"] = "PENDING"
    data["protected_attribute"] = None

    at = _run(data)
    text = _all_text(at)

    assert "declares no protected attribute" in text
    assert "never inferred" in text


def test_an_unavailable_score_channel_is_visible_as_a_limitation():
    data = payload()
    data["result"]["prediction_drift"].update(
        {
            "score_psi": None,
            "score_ks_statistic": None,
            "score_status": "PENDING",
            "score_availability": "unavailable_no_scores",
        }
    )

    at = _run(data)
    assert "no probability output" in _all_text(at)


def test_evidence_records_are_listed_when_present():
    data = payload()
    data["evidence"] = [
        {
            "evidence_type": "monitoring_summary",
            "status": "FAIL",
            "model_id": "some-model",
        }
    ]

    at = _run(data)
    assert "monitoring_summary" in _table_cells(at)


# ---------------------------------------------------------------------------
# The client never fabricates a monitoring verdict
# ---------------------------------------------------------------------------


def test_the_client_returns_empty_data_when_the_api_is_unreachable(monkeypatch):
    """A fabricated PSI or PASS must never reach a risk dashboard."""

    def _boom(*args, **kwargs):
        raise requests.RequestException("unreachable")

    monkeypatch.setattr(requests, "get", _boom)

    data, source = get_monitoring()

    assert data["result"] is None
    assert data["evidence"] == []
    assert source == SOURCE_UNAVAILABLE


def test_the_client_returns_the_api_payload_on_success(monkeypatch):
    expected = payload()

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return expected

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())

    data, source = get_monitoring(model_id="some-model")

    assert data == expected
    assert source == "API"


def test_the_client_imports_no_mock_fixture():
    """Unlike the shared client, monitoring has no mock fallback by design."""
    from pathlib import Path

    source = Path("dashboard/monitoring_client.py").read_text(encoding="utf-8")
    assert "mock_data" not in source
    assert "MOCK_" not in source
