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
    _current_model_id,
    _derive_feature_space,
    build_assurance_result,
    build_assurance_run_context,
    build_drift_assurance_envelope,
    build_drift_comparison,
    build_fairness_assurance_envelope,
    compute_real_drift,
    compute_real_fairness,
    compute_real_model,
    compute_real_model_metrics,
    mint_assurance_run_id,
    summarize,
)
from app.api.schemas import DriftAssuranceEnvelope, FairnessAssuranceEnvelope, FairnessResult
from app.models.model import RandomForestAdapter, predict_batch
from app.models.preprocessing import DEFAULT_DATASET_PATH


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
    assert set(res["model"].keys()) == {
        "predictions",
        "probabilities",
        "instance_ids",
        "feature_matrix",
        "model_metadata",
        "is_mock",
        "model_metrics",
    }
    assert res["model"]["model_metrics"] is not None
    assert res["model"]["model_metrics"]["is_mock"] is False
    assert res["model"]["model_metrics"]["n_test_samples"] == 200

    # Explainability domain
    assert res["explainability"]["is_mock"] is False
    assert res["explainability"]["method"] == "shap"
    assert len(res["explainability"]["global_importance"]) > 0

    # Fairness & Drift domain
    assert res["fairness_drift"]["fairness"]["is_mock"] is False
    assert res["fairness_drift"]["drift"]["is_mock"] is False
    assert set(res["fairness_drift"].keys()) == {"fairness", "drift"}
    assert set(res["fairness_drift"]["fairness"].keys()) == {
        "protected_attribute",
        "demographic_parity_diff",
        "disparate_impact_ratio",
        "status",
        "is_mock",
        "groups",
    }
    assert len(res["fairness_drift"]["fairness"]["groups"]) > 0
    assert set(res["fairness_drift"]["drift"].keys()) == {
        "features_evaluated",
        "psi",
        "ks_statistic",
        "status",
        "is_mock",
        "per_feature",
    }
    assert len(res["fairness_drift"]["drift"]["per_feature"]) > 0

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


def test_mint_assurance_run_id_consecutive_calls_unique():
    """Two consecutive calls that mint an assurance_run_id produce two different values."""
    id1 = mint_assurance_run_id()
    id2 = mint_assurance_run_id()
    assert isinstance(id1, str) and len(id1) > 0
    assert isinstance(id2, str) and len(id2) > 0
    assert id1 != id2


def test_current_model_id_default():
    """_current_model_id() returns 'german-credit-logistic-regression' today."""
    assert _current_model_id() == "german-credit-logistic-regression"


def test_swap_proof_current_model_id_monkeypatch(monkeypatch):
    """Assert AssuranceRunContext built from _current_model_id reflects monkeypatched value."""
    monkeypatch.setattr(
        "app.api.orchestration._current_model_id",
        lambda: "german-credit-random-forest",
    )
    ctx = build_assurance_run_context(
        assurance_run_id="run-test-123",
        model_version="0.2.0",
    )
    assert ctx.model_id == "german-credit-random-forest"
    assert ctx.model_version == "0.2.0"
    assert ctx.assurance_run_id == "run-test-123"


def test_derive_feature_space_deterministic():
    """_derive_feature_space(): same input list twice -> same hash (string equal and length 16)."""
    features = ["duration_months", "credit_amount", "age_years"]
    h1 = _derive_feature_space(features)
    h2 = _derive_feature_space(features)
    assert h1 == h2
    assert len(h1) == 16
    assert isinstance(h1, str)


def test_derive_feature_space_different_inputs():
    """_derive_feature_space(): two different feature lists -> different hashes."""
    h1 = _derive_feature_space(["feat_a", "feat_b"])
    h2 = _derive_feature_space(["feat_a", "feat_c"])
    assert h1 != h2
    assert len(h1) == 16
    assert len(h2) == 16


def test_derive_feature_space_empty_raises():
    """_derive_feature_space(): empty list -> raises ValueError with message mentioning PENDING or empty."""
    with pytest.raises(ValueError) as exc_info:
        _derive_feature_space([])
    msg = str(exc_info.value)
    assert "PENDING" in msg or "empty" in msg


def test_derive_feature_space_real_drift_output():
    """_derive_feature_space(): called against a REAL drift_report() output."""
    raw_model = compute_real_model()
    drift_res = compute_real_drift(raw_model)
    features_evaluated = drift_res["features_evaluated"]

    assert isinstance(features_evaluated, list)
    assert len(features_evaluated) > 0
    assert all(isinstance(f, str) for f in features_evaluated)

    fingerprint = _derive_feature_space(features_evaluated)
    assert len(fingerprint) == 16
    assert isinstance(fingerprint, str)
    # Confirm stability
    assert fingerprint == _derive_feature_space(features_evaluated)


def test_build_fairness_assurance_envelope_with_real_fairness():
    """Wrap real compute_real_fairness() output in FairnessAssuranceEnvelope."""
    raw_model = compute_real_model()
    real_fairness = compute_real_fairness(raw_model)
    model_version = "0.1.0"

    envelope_dict = build_fairness_assurance_envelope(
        real_fairness, model_version=model_version
    )

    # Validates against FairnessAssuranceEnvelope
    parsed = FairnessAssuranceEnvelope(**envelope_dict)
    assert parsed.context.model_id == "german-credit-logistic-regression"
    assert parsed.context.model_version == "0.1.0"
    assert isinstance(parsed.context.assurance_run_id, str)
    assert len(parsed.context.assurance_run_id) > 0
    assert parsed.context.adapter_id is None

    # Result matches the input fairness_dict's fields exactly (byte-identical)
    assert envelope_dict["result"] == real_fairness
    assert parsed.result.model_dump() == FairnessResult(**real_fairness).model_dump()


def test_build_assurance_result_exact_five_top_level_keys():
    """Verify build_assurance_result still returns exactly the 5 original top-level keys."""
    res = build_assurance_result()
    expected_keys = {"model", "explainability", "fairness_drift", "compliance", "note"}
    assert set(res.keys()) == expected_keys
    assert len(res.keys()) == 5


def test_compute_real_model_metrics_returns_computed_roc_auc_status():
    """Assert compute_real_model_metrics() against real LR model returns roc_auc_status == 'computed'."""
    metrics = compute_real_model_metrics()
    assert metrics["roc_auc_status"] == "computed"
    assert isinstance(metrics["roc_auc"], float)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["is_mock"] is False


def test_build_drift_assurance_envelope_with_real_drift():
    """Wrap real compute_real_drift() output in DriftAssuranceEnvelope."""
    raw_model = compute_real_model()
    real_drift = compute_real_drift(raw_model)
    model_version = "0.1.0"

    envelope_dict = build_drift_assurance_envelope(
        real_drift, model_version=model_version
    )

    # Validates against DriftAssuranceEnvelope
    parsed = DriftAssuranceEnvelope(**envelope_dict)
    assert parsed.context.model_id == "german-credit-logistic-regression"
    assert parsed.context.model_version == "0.1.0"
    assert isinstance(parsed.context.assurance_run_id, str)
    assert len(parsed.context.assurance_run_id) > 0
    assert parsed.dataset_id == DEFAULT_DATASET_PATH
    assert parsed.dataset_version is None
    assert len(parsed.feature_space) == 16

    # Result matches the input drift_dict field-for-field
    assert envelope_dict["result"] == real_drift


def test_build_drift_assurance_envelope_empty_features_raises():
    """Empty features_evaluated (PENDING path) raises ValueError."""
    fake_drift = {
        "drift_detected": False,
        "features_evaluated": [],
        "features_with_drift": [],
        "per_feature": {},
        "is_mock": False,
    }
    with pytest.raises(ValueError) as exc_info:
        build_drift_assurance_envelope(fake_drift, model_version="0.1.0")
    assert "PENDING" in str(exc_info.value) or "empty" in str(exc_info.value)


def test_build_assurance_run_context_explicit_model_id_overrides_default():
    """Explicit model_id parameter overrides default model id."""
    ctx = build_assurance_run_context(
        "run-1", model_id="german-credit-random-forest", model_version="0.1.0"
    )
    assert ctx.model_id == "german-credit-random-forest"


def test_build_assurance_run_context_omitted_model_id_falls_back():
    """Omitted model_id parameter falls back to default LR model id."""
    ctx = build_assurance_run_context("run-1", model_version="0.1.0")
    assert ctx.model_id == "german-credit-logistic-regression"


def test_build_fairness_assurance_envelope_with_explicit_model_id():
    """Wrap real fairness output with an explicit model_id."""
    raw_model = compute_real_model()
    real_fairness = compute_real_fairness(raw_model)
    envelope = build_fairness_assurance_envelope(
        real_fairness,
        model_version="0.1.0",
        model_id="german-credit-random-forest",
    )
    parsed = FairnessAssuranceEnvelope(**envelope)
    assert parsed.context.model_id == "german-credit-random-forest"


def test_build_drift_assurance_envelope_with_explicit_model_id():
    """Wrap real drift output with an explicit model_id."""
    raw_model = compute_real_model()
    real_drift = compute_real_drift(raw_model)
    envelope = build_drift_assurance_envelope(
        real_drift,
        model_version="0.1.0",
        model_id="german-credit-random-forest",
    )
    parsed = DriftAssuranceEnvelope(**envelope)
    assert parsed.context.model_id == "german-credit-random-forest"


def test_compute_real_model_with_rf_adapter_matches_predict_batch():
    """compute_real_model(adapter=adapter) delegates faithfully to predict_batch."""
    adapter = RandomForestAdapter.load_default()
    real_res = compute_real_model(adapter=adapter)
    expected_res = predict_batch(adapter=adapter)
    assert real_res["predictions"] == expected_res["predictions"]
    assert real_res["probabilities"] == expected_res["probabilities"]
    assert real_res["instance_ids"] == expected_res["instance_ids"]
    assert real_res["model_metadata"] == expected_res["model_metadata"]
    assert real_res["is_mock"] == expected_res["is_mock"]


def test_build_drift_comparison_returns_comparable_with_correct_identities():
    """build_drift_comparison compares LR and RF drift envelopes."""
    result = build_drift_comparison()
    assert result["comparability"] == "COMPARABLE"
    assert result["drift_a"]["context"]["model_id"] == "german-credit-logistic-regression"
    assert result["drift_b"]["context"]["model_id"] == "german-credit-random-forest"
    assert result["drift_a"]["context"]["model_id"] != result["drift_b"]["context"]["model_id"]

    # Validate both envelopes parse as DriftAssuranceEnvelope
    DriftAssuranceEnvelope(**result["drift_a"])
    DriftAssuranceEnvelope(**result["drift_b"])



