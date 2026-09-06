import ast
import json
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path so run_assurance can be imported directly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

import run_assurance
from app.api.orchestration import (
    ASSURANCE_NOTE,
    build_assurance_result,
    summarize,
)


def test_orchestration_zero_fastapi_imports():
    """Verify app/api/orchestration.py does not import fastapi, uvicorn, or starlette."""
    file_path = Path("app/api/orchestration.py")
    tree = ast.parse(file_path.read_text(encoding="utf-8"))

    forbidden_modules = {"fastapi", "uvicorn", "starlette"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                assert root_pkg not in forbidden_modules, f"Forbidden import found: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                assert root_pkg not in forbidden_modules, f"Forbidden from-import found: {node.module}"


def test_build_assurance_result_structure():
    """Verify build_assurance_result returns the expected schema and metadata."""
    res = build_assurance_result()

    expected_top_keys = {"model", "explainability", "fairness_drift", "compliance", "note"}
    assert set(res.keys()) == expected_top_keys

    # Model domain
    assert res["model"]["is_mock"] is False
    assert len(res["model"]["predictions"]) == 200
    assert len(res["model"]["probabilities"]) == 200
    assert isinstance(res["model"]["feature_matrix"], list)

    # Explainability domain
    assert res["explainability"]["is_mock"] is False
    assert res["explainability"]["method"] == "shap"
    assert len(res["explainability"]["global_importance"]) > 0

    # Fairness & Drift domain
    assert res["fairness_drift"]["fairness"]["is_mock"] is False
    assert res["fairness_drift"]["drift"]["is_mock"] is False
    assert set(res["fairness_drift"].keys()) == {"fairness", "drift"}
    assert set(res["fairness_drift"]["drift"].keys()) == {
        "features_evaluated",
        "psi",
        "ks_statistic",
        "status",
        "is_mock",
    }

    # Compliance domain
    assert res["compliance"]["is_mock"] is True
    assert len(res["compliance"]["findings"]) == 6

    # Overall note
    assert res["note"] == ASSURANCE_NOTE


def test_summarize_real_pipeline():
    """Verify summarize() derives sensible per-domain statuses on the real pipeline output."""
    res = build_assurance_result()
    summary = summarize(res)

    expected_keys = {"model", "explainability", "fairness", "drift", "compliance"}
    assert set(summary.keys()) == expected_keys

    assert summary["model"] == "PASS"
    assert summary["explainability"] == "PASS"
    assert summary["fairness"] == "FAIL"
    assert summary["drift"] == "PASS"
    assert summary["compliance"] == "FAIL"


def test_summarize_unit_variations():
    """Verify summarize() dynamically evaluates statuses across edge cases."""
    # 1. Model variations
    bad_model = {"model": {"predictions": [], "probabilities": []}}
    assert summarize(bad_model)["model"] == "FAIL"

    mock_model = {"model": {"predictions": [1], "probabilities": [0.8], "is_mock": True}}
    assert summarize(mock_model)["model"] == "PASS"

    # 2. Explainability variations
    bad_explain = {"explainability": {"global_importance": {}, "per_instance": []}}
    assert summarize(bad_explain)["explainability"] == "FAIL"

    mock_explain = {
        "explainability": {
            "global_importance": {"f1": 0.5},
            "per_instance": [{"row_index": 0}],
            "is_mock": True,
        }
    }
    assert summarize(mock_explain)["explainability"] == "PASS"

    # 3. Drift without synthetic note
    real_drift = {"fairness_drift": {"drift": {"status": "PASS", "note": "Real drift"}}}
    assert summarize(real_drift)["drift"] == "PASS"

    # 4. Compliance variations
    all_pass = {
        "compliance": {
            "findings": [{"status": "PASS"}, {"status": "PASS"}],
        }
    }
    assert summarize(all_pass)["compliance"] == "PASS"

    all_fail = {
        "compliance": {
            "findings": [{"status": "FAIL"}, {"status": "FAIL"}],
        }
    }
    assert summarize(all_fail)["compliance"] == "FAIL"

    warn_only = {
        "compliance": {
            "findings": [{"status": "PASS"}, {"status": "WARNING"}],
        }
    }
    assert summarize(warn_only)["compliance"] == "WARNING"

    no_findings = {"compliance": {"findings": []}}
    assert summarize(no_findings)["compliance"] == "PENDING"


def test_run_assurance_cli_no_args(capsys):
    """Verify run_assurance.py main() runs with zero arguments, returns 0, and prints report."""
    exit_code = run_assurance.main([])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "AI Model Risk & Assurance Copilot - Assurance Report" in captured.out
    assert "Model:          PASS" in captured.out
    assert "Explainability: PASS" in captured.out
    assert "Fairness:       FAIL" in captured.out
    assert "Drift:          PASS" in captured.out
    assert "Compliance:     FAIL" in captured.out
    assert "Compliance findings are evaluated against illustrative sample RBI rules" in captured.out
    assert "(is_mock: True)" in captured.out


def test_run_assurance_cli_json_export(tmp_path, capsys):
    """Verify run_assurance.py exports full JSON when --json PATH is passed."""
    export_file = tmp_path / "assurance_output.json"
    exit_code = run_assurance.main(["--json", str(export_file)])
    assert exit_code == 0
    assert export_file.exists()

    with open(export_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert set(data.keys()) == {"model", "explainability", "fairness_drift", "compliance", "note"}
    captured = capsys.readouterr()
    assert str(export_file) in captured.out


def test_run_assurance_cli_error_handling(monkeypatch, capsys):
    """Verify run_assurance.py prints [ERROR] to stderr and returns 1 on exception."""
    def _mock_failure():
        raise RuntimeError("Simulated pipeline crash")

    monkeypatch.setattr("run_assurance.build_assurance_result", _mock_failure)
    exit_code = run_assurance.main([])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "[ERROR]" in captured.err
    assert "Simulated pipeline crash" in captured.err


def test_run_assurance_subprocess():
    """Verify executing python run_assurance.py via subprocess runs successfully end-to-end."""
    proc = subprocess.run(
        [sys.executable, "run_assurance.py"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "AI Model Risk & Assurance Copilot - Assurance Report" in proc.stdout
    assert "Drift:          PASS" in proc.stdout
    assert "Per-Domain Status:" in proc.stdout
