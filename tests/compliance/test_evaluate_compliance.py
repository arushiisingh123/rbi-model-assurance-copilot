"""Phase 1 end-to-end tests for evaluate_compliance()."""
from app.compliance import evaluate_compliance
from app.compliance.engine import STATUSES
from app.compliance.mock_findings import MOCK_TECHNICAL_FINDINGS
from app.rbi.rules import load_rules

APPROVED_FINDING_KEYS = {
    "rule_id",
    "rule_description",
    "technical_finding_ref",
    "status",
    "evidence_chunks",
}


def test_output_matches_approved_shape():
    result = evaluate_compliance(MOCK_TECHNICAL_FINDINGS)
    assert set(result) == {"findings", "is_mock"}
    assert result["is_mock"] is True
    assert len(result["findings"]) == len(load_rules())
    for finding in result["findings"]:
        assert set(finding) == APPROVED_FINDING_KEYS
        assert finding["evidence_chunks"] == []
        assert finding["status"] in STATUSES


def test_mock_findings_produce_a_mix_of_statuses():
    result = evaluate_compliance(MOCK_TECHNICAL_FINDINGS)
    statuses = {f["rule_id"]: f["status"] for f in result["findings"]}
    # RBI-FAIR-01 / RBI-DRIFT-01 mirror the mock's own "status" field
    # (WARNING / PASS) rather than re-deriving it from the raw ratio/PSI --
    # see docs/decisions.md, "Analytical threshold authority". RBI-FAIR-02
    # / RBI-DRIFT-02 are presence-only (docs/thresholds.md sec 4: no
    # threshold defined for demographic parity diff or KS), so they PASS
    # once the metric is reported. Explainability + model metadata
    # present -> PASS.
    assert statuses["RBI-FAIR-01"] == "WARNING"
    assert statuses["RBI-FAIR-02"] == "PASS"
    assert statuses["RBI-DRIFT-01"] == "PASS"
    assert statuses["RBI-DRIFT-02"] == "PASS"
    assert statuses["RBI-EXPL-01"] == "PASS"
    assert statuses["RBI-MODEL-01"] == "PASS"
    assert any(s != "PENDING" for s in statuses.values())


def test_mock_fairness_and_drift_status_are_consistent_with_thresholds_py():
    # The mock's own "status" values must not silently drift out of sync
    # with the authoritative classifiers Arushi's real modules use.
    from app.config.thresholds import classify_disparate_impact, classify_psi

    fairness = MOCK_TECHNICAL_FINDINGS["fairness"]
    drift = MOCK_TECHNICAL_FINDINGS["drift"]
    assert fairness["status"] == classify_disparate_impact(
        fairness["disparate_impact_ratio"]
    )
    assert drift["status"] == classify_psi(drift["psi"])


def test_fairness_and_drift_metric_rules_use_canonical_thresholds_only():
    # Locks in the "Analytical threshold authority" fix: the compliance
    # rule set must not define its own competing fairness/drift numbers.
    rules = {r["rule_id"]: r for r in load_rules()}
    for rule_id in ("RBI-FAIR-01", "RBI-DRIFT-01"):
        assert rules[rule_id]["evaluation"] == {"operator": "mirror_status"}
    for rule_id in ("RBI-FAIR-02", "RBI-DRIFT-02"):
        assert rules[rule_id]["evaluation"] == {"operator": "presence"}


def test_none_input_gives_all_not_evaluated():
    result = evaluate_compliance(None)
    assert result["is_mock"] is True
    assert len(result["findings"]) >= 1
    assert all(f["status"] == "PENDING" for f in result["findings"])


def test_non_dict_input_is_handled():
    result = evaluate_compliance("not a dict")
    assert all(f["status"] == "PENDING" for f in result["findings"])
