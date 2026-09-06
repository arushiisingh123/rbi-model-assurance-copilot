"""Regression test for the dashboard/app.py vs. app/ package name collision (Owner: Khushi).

Streamlit's ``streamlit run <script>`` puts the script's own directory at the
front of ``sys.path``. When the Streamlit entry point lived at
``dashboard/app.py``, that put a file literally named ``app.py`` ahead of the
top-level ``app/`` package, so ``dashboard/api_client.py``'s
``from app.api.mock_data import ...`` resolved to the wrong module and raised
``ImportError: cannot import name 'get_compliance' from partially initialized
module 'dashboard.api_client'``. The entry point was renamed to
``dashboard/dashboard_app.py`` to remove the collision; this test simulates
Streamlit's sys.path insertion directly (without spinning up a Streamlit
server) so a future re-introduction of a ``dashboard/app.py`` file is caught.
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


def test_dashboard_app_source_contains_drift_disclaimer():
    """Verify that dashboard_app.py source code contains the approved synthetic drift disclaimer."""
    source_text = (DASHBOARD_DIR / "dashboard_app.py").read_text(encoding="utf-8")
    assert "Synthetic / controlled drift scenario" in source_text
    assert "NOT observed production drift" in source_text
    assert "2026-08-27" in source_text


def test_dashboard_app_renders_without_error_and_shows_disclaimer():
    """Run dashboard_app via Streamlit AppTest and verify zero exceptions and disclaimer presence."""
    from streamlit.testing.v1 import AppTest

    app_path = str(DASHBOARD_DIR / "dashboard_app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Dashboard raised unexpected exception: {at.exception}"

    caption_texts = [c.value for c in at.caption]
    matching = [
        c for c in caption_texts
        if "Synthetic / controlled drift scenario" in c and "NOT observed production drift" in c
    ]
    assert len(matching) >= 1, (
        f"Expected visible synthetic drift disclaimer in dashboard captions, found: {caption_texts}"
    )
