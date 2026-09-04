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
    # From the mock values: DI ratio 0.78 -> FAIL; parity diff 0.14 -> WARNING;
    # psi 0.09 / ks 0.11 -> PASS; explainability + model metadata present -> PASS.
    assert statuses["RBI-FAIR-01"] == "FAIL"
    assert statuses["RBI-FAIR-02"] == "WARNING"
    assert statuses["RBI-DRIFT-01"] == "PASS"
    assert statuses["RBI-EXPL-01"] == "PASS"
    assert any(s != "NOT_EVALUATED" for s in statuses.values())


def test_none_input_gives_all_not_evaluated():
    result = evaluate_compliance(None)
    assert result["is_mock"] is True
    assert len(result["findings"]) >= 1
    assert all(f["status"] == "NOT_EVALUATED" for f in result["findings"])


def test_non_dict_input_is_handled():
    result = evaluate_compliance("not a dict")
    assert all(f["status"] == "NOT_EVALUATED" for f in result["findings"])
