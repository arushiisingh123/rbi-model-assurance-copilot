"""Phase 4 tests: explainability panel RENDERING (owner: Manas).

Why this file exists separately from ``test_explainability_presentation.py``:
that file tests the pure helper (sorting, top-N, scale labels, signed-value
preservation) and never renders anything. A live Phase 4 verification failure
showed the gap — the panel raised

    AttributeError: 'NoneType' object has no attribute 'removeprefix'

for BOTH shap and lime, because ``st.bar_chart(..., sort=None)`` is invalid:
Streamlit treats a non-bool ``sort`` as a column name and calls
``.removeprefix()`` on it (``streamlit/elements/lib/built_in_chart_utils.py::
_parse_sort_column``). Every value the helper produced was correct, so no
pure-helper test could have caught it. These tests render the panel.

Uses the ``streamlit.testing.v1.AppTest`` convention established by the other
Phase 4 panel tests.
"""
import pytest
from streamlit.testing.v1 import AppTest

from app.api.mock_data import (
    MOCK_EXPLAINABILITY_RESULT_LIME,
    MOCK_EXPLAINABILITY_RESULT_SHAP,
)


def _run(method: str = "shap", *, source: str = "api", data=None) -> AppTest:
    """Render the panel for one method through AppTest."""

    def _wrapper(method, source, data):
        # Imports must live INSIDE the wrapper: AppTest.from_function extracts
        # only this function's source into a temp script, so names imported at
        # this test module's top level are not in scope there.
        from app.api.mock_data import (
            MOCK_EXPLAINABILITY_RESULT_LIME,
            MOCK_EXPLAINABILITY_RESULT_SHAP,
        )
        from dashboard.panels.explainability_panel import render

        payload = data if data is not None else (
            MOCK_EXPLAINABILITY_RESULT_LIME
            if method == "lime"
            else MOCK_EXPLAINABILITY_RESULT_SHAP
        )

        def fetch(requested_method):
            return payload, source

        render(fetch_explainability=fetch)

    at = AppTest.from_function(
        _wrapper, kwargs={"method": method, "source": source, "data": data}
    )
    # Generous timeout: the first render in a cold process pays the transitive
    # shap/lime import cost, which exceeded a 60s budget on one run while the
    # same test passed in ~25s once warm. The panel itself does no analytical
    # work; this guards the harness, not the code under test.
    at.run(timeout=180)
    return at


def _all_text(at: AppTest) -> str:
    parts = []
    for block in (at.markdown, at.caption, at.info, at.warning, at.error, at.subheader):
        parts.extend(str(element.value) for element in block)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The regression: the panel must actually render
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_panel_renders_without_exception(method):
    """Regression guard for the live `sort=None` AttributeError.

    Before the fix this raised for BOTH methods, after the global-importance
    subheader had already been written — which is exactly what live
    verification reported.
    """
    at = _run(method)
    assert not at.exception, f"{method} render raised: {[e.value for e in at.exception]}"


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_panel_reports_no_error_element(method):
    """`_safe_render` converts an exception into st.error — none may appear."""
    at = _run(method)
    assert not at.error, [str(e.value) for e in at.error]


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_charts_are_actually_drawn(method):
    """The failure happened AT the chart call, so assert charts exist."""
    at = _run(method)
    assert len(at.get("vega_lite_chart")) >= 1, "no chart was rendered"


def test_bar_chart_sort_argument_is_never_none():
    """Pin the specific invalid argument that caused the outage.

    Streamlit's `_parse_sort_column` accepts `bool | str`; `None` falls through
    to the string branch and raises AttributeError. Guard against a reviewer
    "tidying" False back to None.
    """
    import inspect

    import dashboard.panels.explainability_panel as panel

    source = inspect.getsource(panel)
    assert "sort=None" not in source, (
        "st.bar_chart(sort=None) raises AttributeError inside Streamlit; "
        "use sort=False to preserve the supplied ordering"
    )
    assert "sort=False" in source


# ---------------------------------------------------------------------------
# Presentation contract still holds after the fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_global_and_instance_sections_both_render(method):
    at = _run(method)
    headings = " ".join(element.value for element in at.subheader)

    assert "Global feature importance" in headings
    assert "Single-record explanation" in headings


def test_shap_scale_is_shown_as_log_odds():
    at = _run("shap")
    assert "log_odds" in _all_text(at)


def test_lime_scale_is_shown_as_probability():
    at = _run("lime")
    assert "probability" in _all_text(at)


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_scale_incomparability_caption_survives(method):
    at = _run(method)
    assert "NOT comparable" in _all_text(at)


# ---------------------------------------------------------------------------
# Mock / real distinction (the live LIME path was fallback+mock)
# ---------------------------------------------------------------------------


def test_mock_fallback_data_shows_the_mock_warning():
    """The live LIME path was source=fallback, is_mock=True — keep that visible."""
    at = _run("lime", source="fallback")
    text = _all_text(at)

    assert "fallback" in text
    assert any("MOCK" in str(w.value).upper() for w in at.warning)
    assert not at.exception


def test_real_data_shows_no_mock_warning():
    real = {**MOCK_EXPLAINABILITY_RESULT_SHAP, "is_mock": False}
    at = _run("shap", source="api", data=real)

    assert "Mock data: **False**" in _all_text(at)
    assert not any("MOCK data" in str(w.value) for w in at.warning)


# ---------------------------------------------------------------------------
# Defensive: missing / empty presentation values
# ---------------------------------------------------------------------------


def test_empty_global_importance_does_not_crash():
    data = {**MOCK_EXPLAINABILITY_RESULT_SHAP, "global_importance": {}}
    at = _run("shap", data=data)

    assert not at.exception
    assert "No global importance values" in _all_text(at)


def test_empty_per_instance_does_not_crash():
    data = {**MOCK_EXPLAINABILITY_RESULT_SHAP, "per_instance": []}
    at = _run("shap", data=data)

    assert not at.exception
    assert "No explained records" in _all_text(at)


def test_malformed_payload_is_reported_not_raised():
    """A broken explanation must surface a message, never fabricate numbers."""
    at = _run("shap", data={"method": "shap", "is_mock": False})

    assert not at.exception
    assert "Could not present this explanation" in _all_text(at)
