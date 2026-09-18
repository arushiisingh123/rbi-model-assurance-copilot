"""The two compliance layers coexist; neither contaminates the other.

MIGRATION STATE
    The existing six illustrative rules and their engine remain fully
    operational and unmodified. The new corpus architecture (manifest ->
    applicability -> requirements) runs alongside them.

    That is deliberate. The six rules are wired into /compliance, the
    evidence builders and the report, and 1,600+ tests depend on them. The
    new layer cannot replace them until the corpus actually contains source
    documents — which it does not.

WHAT THESE TESTS PROTECT
    1. The old engine still works exactly as before.
    2. The two status vocabularies stay separate. Silently mapping
       EVIDENCE_MISSING onto PENDING (or PARTIAL onto WARNING) would let an
       unassessed requirement be counted as an assessed one.
    3. Neither layer's honesty markers are lost: the six rules stay
       ILLUSTRATIVE, and the 19 documents stay unresolved.
"""

from __future__ import annotations

from app.compliance.engine import PASS as ENGINE_PASS, evaluate_rule, map_findings_to_rules
from app.config.thresholds import VALID_STATUSES
from app.rbi.applicability import AssessmentContext
from app.rbi.manifest import load_manifest
from app.rbi.requirements import (
    EVIDENCE_MISSING,
    NOT_ASSESSED,
    REQUIREMENT_STATUSES,
    assess_corpus,
)
from app.rbi.rules import load_rules


# ===========================================================================
# The existing six-rule engine is untouched
# ===========================================================================


def test_the_six_illustrative_rules_are_still_present():
    rules = load_rules()
    assert len(rules) == 6
    assert {r["rule_id"] for r in rules} == {
        "RBI-FAIR-01",
        "RBI-FAIR-02",
        "RBI-DRIFT-01",
        "RBI-DRIFT-02",
        "RBI-EXPL-01",
        "RBI-MODEL-01",
    }


def test_the_six_rules_still_declare_themselves_illustrative():
    """Their honesty markers must survive the corpus work.

    These rules cite no real RBI clause on purpose — the one document the
    repo holds is an NPA circular that covers none of these topics. That
    must not quietly change just because a 19-document manifest now exists.
    """
    for rule in load_rules():
        assert rule["is_mock"] is True
        assert rule["rbi_source"].startswith("ILLUSTRATIVE")
        assert rule["clause_reference"] is None


def test_the_six_rule_engine_still_evaluates():
    findings = {
        "fairness": {"status": "PASS", "demographic_parity_diff": 0.05},
        "drift": {"status": "PASS", "ks_statistic": 0.1},
        "explainability": {"global_importance": {"a": 1.0}},
        "model": {"model_metadata": {"model_type": "logistic_regression"}},
    }
    results = map_findings_to_rules(findings)

    assert len(results) == 6
    # map_findings_to_rules reports resolution, not status; evaluate_rule is
    # what produces a status. Both are exercised so the engine is genuinely
    # covered rather than merely imported.
    for result in results:
        assert result["rule_id"]
        assert result["resolved"] is True

    for rule in load_rules():
        assert evaluate_rule(rule, findings) in VALID_STATUSES


def test_engine_statuses_are_unchanged():
    assert set(VALID_STATUSES) == {"PASS", "WARNING", "FAIL", "PENDING"}


# ===========================================================================
# The two vocabularies stay separate
# ===========================================================================


def test_requirement_statuses_and_engine_statuses_are_distinct_sets():
    requirement_only = set(REQUIREMENT_STATUSES) - set(VALID_STATUSES)
    engine_only = set(VALID_STATUSES) - set(REQUIREMENT_STATUSES)

    assert requirement_only == {"PARTIAL", "NOT_ASSESSED", "APPLICABILITY_UNCLEAR", "EVIDENCE_MISSING"}
    # WARNING and PENDING exist ONLY in the engine. PENDING in particular is
    # not an alias for EVIDENCE_MISSING: it means the referenced value was
    # missing or unusable, not that an applicable obligation has no evidence.
    assert engine_only == {"WARNING", "PENDING"}
    # They share PASS and FAIL, which mean the same thing in both.
    assert set(REQUIREMENT_STATUSES) & set(VALID_STATUSES) == {"PASS", "FAIL"}


def test_evidence_missing_is_not_a_valid_engine_status():
    """Guards against a future 'just map it to PENDING' shortcut.

    PENDING in the old engine means "the value was missing or unusable".
    EVIDENCE_MISSING means "the requirement applies and we have nothing".
    Collapsing them would make an unmet obligation look like a data gap.
    """
    assert EVIDENCE_MISSING not in VALID_STATUSES
    assert NOT_ASSESSED not in VALID_STATUSES
    assert "APPLICABILITY_UNCLEAR" not in VALID_STATUSES


def test_corpus_findings_never_carry_an_engine_status():
    findings = assess_corpus(
        load_manifest().all(), AssessmentContext(regulated_entity_type="nbfc"), {}
    )
    for finding in findings:
        assert finding.status in REQUIREMENT_STATUSES
        assert finding.status != "WARNING"


# ===========================================================================
# The two layers describe different things and do not overlap
# ===========================================================================


def test_the_six_rules_and_the_19_documents_are_different_identifier_spaces():
    """A rule_id is not a document_id; the layers cannot be confused."""
    rule_ids = {r["rule_id"] for r in load_rules()}
    document_ids = set(load_manifest().document_ids())

    assert rule_ids & document_ids == set()


def test_corpus_layer_produces_no_conclusive_findings_yet():
    """With no source documents, the new layer asserts nothing about compliance."""
    findings = assess_corpus(
        load_manifest().all(), AssessmentContext(regulated_entity_type="nbfc"), {}
    )
    assert findings
    assert all(f.status == NOT_ASSESSED for f in findings)
    assert not any(f.is_conclusive() for f in findings)


def test_six_rule_engine_can_still_reach_PASS_while_corpus_cannot():
    """The layers are independent: one working does not imply the other does.

    This is the honest current state — the illustrative engine produces
    results, and the corpus layer correctly produces none.
    """
    findings = {"explainability": {"global_importance": {"a": 1.0}}}
    explainability_rule = next(
        r for r in load_rules() if r["rule_id"] == "RBI-EXPL-01"
    )
    assert evaluate_rule(explainability_rule, findings) == ENGINE_PASS

    corpus_findings = assess_corpus(
        load_manifest().all(), AssessmentContext(regulated_entity_type="nbfc"), findings
    )
    assert all(f.status == NOT_ASSESSED for f in corpus_findings)
