"""Phase 5 (evidence_chunks gap closure): RBI evidence attached to findings.

evaluate_compliance() previously hardcoded evidence_chunks: [] for every
finding (docs/decisions.md, "Phase 5D: Compliance + RAG evidence isolation
verified" -- "ComplianceFinding.evidence_chunks remains hardcoded to []
everywhere"). This file covers the new evidence_by_rule parameter that lets
a caller attach already-retrieved app.rag.evidence.RBIEvidence records to
the specific rule finding they support.

Deliberately does not exercise app/rag/retrieval.py or ChromaDB -- this is a
mapping/assembly test, not a retrieval test (RAG retrieval already has its
own coverage in tests/rag/). RBIEvidence records are hand-built here, the
same pattern tests/rag/test_evidence_records.py uses.
"""
import pytest

from app.compliance.compliance import evaluate_compliance
from app.compliance.mock_findings import MOCK_TECHNICAL_FINDINGS
from app.compliance.technical_findings import run_compliance
from app.rag.evidence import RBIEvidence
from app.rbi.rules import load_rules

APPROVED_FINDING_KEYS = {
    "rule_id",
    "rule_description",
    "technical_finding_ref",
    "status",
    "evidence_chunks",
}


def _evidence(
    *, chunk_id: str, doc_id: str = "doc-x", chunk_index: int = 0, text: str = "some rbi text"
) -> RBIEvidence:
    """A minimal, valid RBIEvidence -- an empty provenance dict is accepted
    (RBIEvidence's consistency check only compares keys actually present in
    provenance), so this stays a one-liner per record."""
    return RBIEvidence(
        text=text,
        doc_id=doc_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        source="Some RBI Title",
        source_url="https://rbi.example/circular",
        title="Some RBI Title",
        publication_date="2014-07-01",
        document_type="Master Circular",
        is_excerpt=True,
        is_current=False,
        provenance={},
    )


def _rule_ids() -> list[str]:
    return [rule["rule_id"] for rule in load_rules()]


# ---------------------------------------------------------------------------
# 1. Matching evidence is attached to the correct finding
# ---------------------------------------------------------------------------


def test_finding_receives_matching_evidence_chunk_id():
    ev = _evidence(chunk_id="chunk-fair-01-a")
    result = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS, evidence_by_rule={"RBI-FAIR-01": [ev]}
    )
    findings = {f["rule_id"]: f for f in result["findings"]}
    assert findings["RBI-FAIR-01"]["evidence_chunks"] == ["chunk-fair-01-a"]


# ---------------------------------------------------------------------------
# 2. Evidence does not leak to unrelated rule findings
# ---------------------------------------------------------------------------


def test_evidence_does_not_leak_to_other_rules():
    ev = _evidence(chunk_id="chunk-fair-01-a")
    result = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS, evidence_by_rule={"RBI-FAIR-01": [ev]}
    )
    findings = {f["rule_id"]: f for f in result["findings"]}
    other_rule_ids = [rid for rid in _rule_ids() if rid != "RBI-FAIR-01"]
    assert other_rule_ids, "expected at least one other loaded rule to check isolation against"
    for rule_id in other_rule_ids:
        assert findings[rule_id]["evidence_chunks"] == []


def test_two_rules_each_get_only_their_own_evidence():
    fair_ev = _evidence(chunk_id="chunk-fair-01")
    drift_ev = _evidence(chunk_id="chunk-drift-01", doc_id="doc-y")
    result = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS,
        evidence_by_rule={
            "RBI-FAIR-01": [fair_ev],
            "RBI-DRIFT-01": [drift_ev],
        },
    )
    findings = {f["rule_id"]: f for f in result["findings"]}
    assert findings["RBI-FAIR-01"]["evidence_chunks"] == ["chunk-fair-01"]
    assert findings["RBI-DRIFT-01"]["evidence_chunks"] == ["chunk-drift-01"]
    for rule_id in _rule_ids():
        if rule_id not in ("RBI-FAIR-01", "RBI-DRIFT-01"):
            assert findings[rule_id]["evidence_chunks"] == []


# ---------------------------------------------------------------------------
# 3. Multiple evidence chunks: order + source/chunk/provenance preserved
# ---------------------------------------------------------------------------


def test_multiple_evidence_chunks_preserve_order_and_are_not_collapsed():
    ev_a = _evidence(chunk_id="chunk-a", doc_id="doc-1", chunk_index=0)
    ev_b = _evidence(chunk_id="chunk-b", doc_id="doc-1", chunk_index=1)
    ev_c = _evidence(chunk_id="chunk-c", doc_id="doc-2", chunk_index=0)
    result = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS,
        evidence_by_rule={"RBI-FAIR-01": [ev_a, ev_b, ev_c]},
    )
    findings = {f["rule_id"]: f for f in result["findings"]}
    # Ranked order supplied by the caller is preserved exactly, not
    # deduplicated, reordered, or merged into a single value.
    assert findings["RBI-FAIR-01"]["evidence_chunks"] == ["chunk-a", "chunk-b", "chunk-c"]


def test_source_and_provenance_stay_reachable_via_the_evidence_object():
    # evaluate_compliance() maps to chunk_id strings (the finding-level
    # contract, ComplianceFinding.evidence_chunks: list[str]), but it must
    # not mutate or discard the caller's RBIEvidence objects -- their full
    # source/provenance stays intact and reachable through the RAG layer
    # that produced them.
    ev = _evidence(
        chunk_id="chunk-with-provenance",
        doc_id="doc-irac-2014",
        text="verbatim retrieved rbi text",
    )
    evaluate_compliance(MOCK_TECHNICAL_FINDINGS, evidence_by_rule={"RBI-MODEL-01": [ev]})
    assert ev.doc_id == "doc-irac-2014"
    assert ev.source_url == "https://rbi.example/circular"
    assert ev.is_excerpt is True
    assert ev.is_current is False
    assert ev.text == "verbatim retrieved rbi text"
    assert ev.provenance == {}


# ---------------------------------------------------------------------------
# 4. No verified evidence -> [] , never fabricated
# ---------------------------------------------------------------------------


def test_omitting_evidence_by_rule_matches_existing_baseline():
    result = evaluate_compliance(MOCK_TECHNICAL_FINDINGS)
    assert all(f["evidence_chunks"] == [] for f in result["findings"])
    # Exact key-set unchanged -- no new key leaks into a finding when the
    # new parameter is not used.
    assert all(set(f) == APPROVED_FINDING_KEYS for f in result["findings"])


def test_empty_evidence_by_rule_dict_yields_no_evidence():
    result = evaluate_compliance(MOCK_TECHNICAL_FINDINGS, evidence_by_rule={})
    assert all(f["evidence_chunks"] == [] for f in result["findings"])


def test_rule_mapped_to_empty_list_yields_no_evidence():
    result = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS, evidence_by_rule={"RBI-FAIR-01": []}
    )
    findings = {f["rule_id"]: f for f in result["findings"]}
    assert findings["RBI-FAIR-01"]["evidence_chunks"] == []


def test_evidence_for_unknown_rule_id_is_ignored_not_an_error():
    ev = _evidence(chunk_id="chunk-orphan")
    result = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS, evidence_by_rule={"RBI-DOES-NOT-EXIST": [ev]}
    )
    assert all(f["evidence_chunks"] == [] for f in result["findings"])


def test_evidence_item_without_chunk_id_raises_instead_of_fabricating():
    class _NotEvidence:
        pass

    with pytest.raises(TypeError):
        evaluate_compliance(
            MOCK_TECHNICAL_FINDINGS,
            evidence_by_rule={"RBI-FAIR-01": [_NotEvidence()]},
        )


# ---------------------------------------------------------------------------
# 5. Existing model/run identity isolation keeps working alongside evidence
# ---------------------------------------------------------------------------


def test_evidence_and_identity_kwargs_coexist_without_interference():
    ev = _evidence(chunk_id="chunk-with-identity")
    result = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS,
        model_id="german-credit-random-forest",
        assurance_run_id="run-xyz",
        evidence_by_rule={"RBI-FAIR-01": [ev]},
    )
    findings = {f["rule_id"]: f for f in result["findings"]}
    fair_01 = findings["RBI-FAIR-01"]
    assert fair_01["evidence_chunks"] == ["chunk-with-identity"]
    assert fair_01["model_id"] == "german-credit-random-forest"
    assert fair_01["assurance_run_id"] == "run-xyz"
    # Every other finding still carries identity but no evidence.
    for rule_id, finding in findings.items():
        if rule_id != "RBI-FAIR-01":
            assert finding["evidence_chunks"] == []
        assert finding["model_id"] == "german-credit-random-forest"
        assert finding["assurance_run_id"] == "run-xyz"


def test_two_calls_with_different_evidence_do_not_leak_into_each_other():
    # No hidden global/module-level state: two independent calls (as if
    # for two different models/runs) must not contaminate one another.
    ev_1 = _evidence(chunk_id="chunk-run-1")
    ev_2 = _evidence(chunk_id="chunk-run-2")

    result_1 = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS,
        model_id="model-a",
        evidence_by_rule={"RBI-FAIR-01": [ev_1]},
    )
    result_2 = evaluate_compliance(
        MOCK_TECHNICAL_FINDINGS,
        model_id="model-b",
        evidence_by_rule={"RBI-FAIR-01": [ev_2]},
    )

    findings_1 = {f["rule_id"]: f for f in result_1["findings"]}
    findings_2 = {f["rule_id"]: f for f in result_2["findings"]}
    assert findings_1["RBI-FAIR-01"]["evidence_chunks"] == ["chunk-run-1"]
    assert findings_2["RBI-FAIR-01"]["evidence_chunks"] == ["chunk-run-2"]
    assert findings_1["RBI-FAIR-01"]["model_id"] == "model-a"
    assert findings_2["RBI-FAIR-01"]["model_id"] == "model-b"


# ---------------------------------------------------------------------------
# run_compliance() forwards evidence_by_rule (technical_findings.py wrapper)
# ---------------------------------------------------------------------------


def test_run_compliance_forwards_evidence_by_rule():
    ev = _evidence(chunk_id="chunk-via-run-compliance")
    result = run_compliance(
        fairness=MOCK_TECHNICAL_FINDINGS["fairness"],
        drift=MOCK_TECHNICAL_FINDINGS["drift"],
        explainability=MOCK_TECHNICAL_FINDINGS["explainability"],
        model=MOCK_TECHNICAL_FINDINGS["model"],
        evidence_by_rule={"RBI-FAIR-01": [ev]},
    )
    findings = {f["rule_id"]: f for f in result["findings"]}
    assert findings["RBI-FAIR-01"]["evidence_chunks"] == ["chunk-via-run-compliance"]


def test_run_compliance_omitting_evidence_by_rule_matches_existing_baseline():
    result = run_compliance(
        fairness=MOCK_TECHNICAL_FINDINGS["fairness"],
        drift=MOCK_TECHNICAL_FINDINGS["drift"],
        explainability=MOCK_TECHNICAL_FINDINGS["explainability"],
        model=MOCK_TECHNICAL_FINDINGS["model"],
    )
    assert all(f["evidence_chunks"] == [] for f in result["findings"])
