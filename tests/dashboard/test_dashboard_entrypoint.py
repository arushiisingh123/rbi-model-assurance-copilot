"""Regression and shell-integration tests for dashboard/dashboard_app.py (Owner: Khushi).

Covers two things:
1. The dashboard/app.py vs. app/ package name collision regression (unchanged
   from earlier phases -- see the docstrings below).
2. Phase 4 shell behaviour: the entrypoint must delegate rendering to each
   domain's own panel rather than reimplementing it, must not blank the whole
   page when one panel fails, and must carry no stale phase/status language.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


def test_dashboard_app_py_no_longer_exists():
    assert not (DASHBOARD_DIR / "app.py").exists()
    assert (DASHBOARD_DIR / "dashboard_app.py").exists()


def test_api_client_import_survives_streamlit_style_sys_path():
    """Simulates Streamlit prepending the script directory to sys.path."""
    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(DASHBOARD_DIR)!r})\n"
        "import dashboard.api_client as api_client\n"
        "assert hasattr(api_client, 'get_compliance')\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "ImportError" not in result.stderr


def test_dashboard_app_source_has_no_stale_phase_or_status_language():
    """dashboard_app.py must not claim an outdated phase or a fixed mock/provisional
    status -- both are now determined per-tab, at render time, by the fetched data.
    """
    source_text = (DASHBOARD_DIR / "dashboard_app.py").read_text(encoding="utf-8")
    assert "Phase 2" not in source_text
    assert "Phase 3" not in source_text
    assert "PROVISIONAL" not in source_text
    assert "Phase 4" in source_text


def test_dashboard_app_delegates_rendering_to_owning_panels():
    """The shell must import and call each domain's own panel rather than
    reimplementing that domain's presentation logic inline.
    """
    source_text = (DASHBOARD_DIR / "dashboard_app.py").read_text(encoding="utf-8")
    assert "from dashboard.panels.compliance_panel import render_compliance_panel" in source_text
    assert "from dashboard.panels.report_panel import render_report_panel" in source_text
    assert (
        "from dashboard.panels.fairness_drift_panel import render_fairness_drift_panel"
        in source_text
    )
    assert (
        "from dashboard.panels.explainability_panel import render as render_explainability_panel"
        in source_text
    )


def test_dashboard_app_renders_without_error_and_shows_mock_labelling(monkeypatch):
    """Run dashboard_app via Streamlit AppTest (API unreachable -> every tab falls
    back to mock data) and verify zero exceptions and visible mock labelling.

    The unreachable state is FORCED rather than assumed. Previously this test
    relied on nothing listening on the default port 8000, so it failed on any
    machine where a developer had ``uvicorn`` running for live verification --
    every tab then reported real data and only 2 of the expected 4 mock captions
    appeared. Pointing ``API_BASE_URL`` at a reserved port makes the fallback
    path deterministic. Production fallback behaviour is unchanged; only the
    address the client dials during this test is.
    """
    from streamlit.testing.v1 import AppTest

    # Port 1 on loopback is reserved and refuses immediately, so each client
    # call takes the RequestException branch without waiting for a timeout.
    monkeypatch.setattr("dashboard.api_client.API_BASE_URL", "http://127.0.0.1:1")

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Dashboard raised unexpected exception: {at.exception}"

    caption_texts = [c.value for c in at.caption]
    mock_labelled = [
        c
        for c in caption_texts
        if "**True**" in c and ("Mock data" in c or "is_mock" in c)
    ]
    assert len(mock_labelled) >= 4, (
        "Expected mock/fallback labelling visible on multiple tabs, found "
        f"captions: {caption_texts}"
    )

    # The drift half of the Fairness & Drift tab must still visibly flag
    # fallback data as synthetic, not just the fairness half's caption.
    warning_texts = [w.value for w in at.warning]
    assert any("SYNTHETIC DRIFT SCENARIO" in w for w in warning_texts), (
        f"Expected the drift panel's synthetic-scenario warning under fallback, "
        f"found warnings: {warning_texts}"
    )


def test_dashboard_app_renders_report_tab():
    """Verify that the dashboard initializes with the Report tab and no errors."""
    from streamlit.testing.v1 import AppTest

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Dashboard raised unexpected exception: {at.exception}"
    tab_labels = [t.label for t in at.tabs]
    assert "Report" in tab_labels


def test_dashboard_app_invokes_all_four_wired_panels(monkeypatch):
    """Each of the four delegated panels must be invoked exactly once per page run."""
    from streamlit.testing.v1 import AppTest

    calls = {"compliance": 0, "report": 0, "fairness_drift": 0, "explainability": 0}

    def _fake_compliance(data, source):
        calls["compliance"] += 1

    def _fake_report(data, source):
        calls["report"] += 1

    def _fake_fairness_drift(data, source):
        calls["fairness_drift"] += 1

    def _fake_explainability(fetch_explainability, **kwargs):
        calls["explainability"] += 1

    monkeypatch.setattr(
        "dashboard.panels.compliance_panel.render_compliance_panel", _fake_compliance
    )
    monkeypatch.setattr("dashboard.panels.report_panel.render_report_panel", _fake_report)
    monkeypatch.setattr(
        "dashboard.panels.fairness_drift_panel.render_fairness_drift_panel",
        _fake_fairness_drift,
    )
    monkeypatch.setattr("dashboard.panels.explainability_panel.render", _fake_explainability)

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Dashboard raised unexpected exception: {at.exception}"

    assert calls == {
        "compliance": 1,
        "report": 1,
        "fairness_drift": 1,
        "explainability": 1,
    }, calls


def test_dashboard_app_isolates_tab_failures(monkeypatch):
    """One panel raising must not blank the page or stop the other tabs rendering."""
    from streamlit.testing.v1 import AppTest

    def _boom(data, source):
        raise RuntimeError("simulated panel failure")

    monkeypatch.setattr("dashboard.panels.compliance_panel.render_compliance_panel", _boom)

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, (
        f"A single panel failure must not raise a top-level exception: {at.exception}"
    )

    error_texts = [e.value for e in at.error]
    assert any("Compliance" in e and "failed to render" in e for e in error_texts), error_texts

    # The Report tab (rendered after Compliance) must still have produced its
    # own content -- the failure must not have stopped the rest of the page.
    caption_texts = [c.value for c in at.caption]
    assert any("Report is_mock" in c for c in caption_texts), caption_texts


def test_dashboard_app_renders_health_banner(monkeypatch):
    """The reachability banner must render without error in both reachable and fallback states."""
    from streamlit.testing.v1 import AppTest

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")

    # 1. Fallback / unreachable state
    monkeypatch.setattr(
        "dashboard.api_client.get_health",
        lambda: ({"status": "unreachable"}, "fallback"),
    )
    at_fallback = AppTest.from_file(app_path)
    at_fallback.run(timeout=30)
    assert not at_fallback.exception, f"Unexpected exception in fallback state: {at_fallback.exception}"
    warning_texts = [w.value for w in at_fallback.warning]
    assert any("Unreachable" in w for w in warning_texts), (
        f"Expected unreachable warning banner, found: {warning_texts}"
    )

    # 2. Reachable / live API state
    monkeypatch.setattr(
        "dashboard.api_client.get_health",
        lambda: ({"status": "ok"}, "api"),
    )
    at_api = AppTest.from_file(app_path)
    at_api.run(timeout=30)
    assert not at_api.exception, f"Unexpected exception in api state: {at_api.exception}"
    success_texts = [s.value for s in at_api.success]
    assert any("Connected" in s for s in success_texts), (
        f"Expected connected success banner, found: {success_texts}"
    )


def test_dashboard_app_builds_six_tabs():
    """Verify dashboard builds exactly 6 tabs including Model Comparison."""
    from streamlit.testing.v1 import AppTest

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Dashboard raised unexpected exception: {at.exception}"
    tab_labels = [t.label for t in at.tabs]
    assert len(tab_labels) == 6
    assert tab_labels == [
        "Model",
        "Explainability",
        "Fairness & Drift",
        "Compliance",
        "Report",
        "Model Comparison",
    ]


def test_model_comparison_tab_unavailable_fallback_isolates_failure(monkeypatch):
    """When get_drift_comparison() reports unavailable, tab shows warning and doesn't crash."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        "dashboard.api_client.get_drift_comparison",
        lambda: (None, "unavailable"),
    )

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, (
        f"Unexpected exception when comparison unavailable: {at.exception}"
    )

    # Warning for live API requirement is present
    warning_texts = [w.value for w in at.warning]
    assert any("Model comparison requires the live API" in w for w in warning_texts), (
        f"Expected live API required warning, found: {warning_texts}"
    )

    # All 6 tabs still rendered without crashing
    tab_labels = [t.label for t in at.tabs]
    assert len(tab_labels) == 6
    assert "Model Comparison" in tab_labels


def test_model_comparison_tab_renders_live_comparison(monkeypatch):
    """When get_drift_comparison() returns comparison data, renders metrics and comparability."""
    from streamlit.testing.v1 import AppTest

    mock_comparison = {
        "comparability": "COMPARABLE",
        "reason": "feature spaces and datasets match.",
        "drift_a": {
            "context": {
                "model_id": "german-credit-logistic-regression",
                "model_version": "0.1.0",
                "assurance_run_id": "run-a-12345678",
                "adapter_id": None,
            },
            "result": {
                "status": "PASS",
                "psi": 0.0725,
                "ks_statistic": 0.0737,
            },
        },
        "drift_b": {
            "context": {
                "model_id": "german-credit-random-forest",
                "model_version": "0.1.0",
                "assurance_run_id": "run-b-12345678",
                "adapter_id": None,
            },
            "result": {
                "status": "PASS",
                "psi": 0.0725,
                "ks_statistic": 0.0737,
            },
        },
    }

    monkeypatch.setattr(
        "dashboard.api_client.get_drift_comparison",
        lambda: (mock_comparison, "api"),
    )

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Unexpected exception in live comparison: {at.exception}"

    success_texts = [s.value for s in at.success]
    assert any("COMPARABLE" in s for s in success_texts), success_texts

    subheaders = [s.value for s in at.subheader]
    assert any("Cross-Model Drift Comparison" in s for s in subheaders), subheaders

    markdowns = [m.value for m in at.markdown]
    assert any("german-credit-logistic-regression" in m for m in markdowns)
    assert any("german-credit-random-forest" in m for m in markdowns)


