"""Tests for the manifest-driven RBI corpus, applicability and requirements.

NO NETWORK. Every test reads the committed manifest or builds local
fixtures. The suite must pass with rbi.org.in unreachable, because a test
that silently depends on a live regulator website fails for reasons that
have nothing to do with the code.

The theme throughout is that UNCERTAINTY MUST SURVIVE. The corpus is
currently empty of source text, and these tests exist to prove the system
says so — loudly, in every layer — instead of producing confident output
built on nothing.
"""

from __future__ import annotations

import json

import pytest

from app.rbi.applicability import (
    APPLICABILITY_UNCLEAR,
    APPLIES,
    DOES_NOT_APPLY,
    AssessmentContext,
    evaluate_applicability,
    filter_applicable,
    rank_by_authority,
)
from app.rbi.manifest import (
    MANIFEST_PATH,
    REQUIRED_DOCUMENT_KEYS,
    SOURCE_DOWNLOADED,
    TIER_0,
    TIER_1,
    TIER_2,
    VALID_REGULATORY_STATUS,
    VALID_SOURCE_STATUS,
    ManifestError,
    RBIManifest,
    load_manifest,
    validate_document_entry,
)
from app.rbi.requirements import (
    APPLICABILITY_UNCLEAR as REQ_UNCLEAR,
    CONCLUSIVE_STATUSES,
    EVIDENCE_MISSING,
    NOT_ASSESSED,
    PARTIAL,
    PASS,
    REQUIREMENT_STATUSES,
    EvidenceRequirement,
    RegulatoryRequirement,
    SourceLocation,
    assess_corpus,
    assess_requirement,
    requirements_for_document,
    summarise,
)


@pytest.fixture(scope="module")
def manifest() -> RBIManifest:
    return load_manifest()


# ===========================================================================
# Manifest structure
# ===========================================================================


# The corpus specification defined 19 source documents. Four have been added
# since, each a genuinely distinct instrument rather than an edit to an
# existing entry:
#
#   RBI-FS-OUTSOURCE-NBFC-2017     vs the 2006 bank circular, which excludes
#                                  NBFCs while this one is addressed to them
#   RBI-DIGITAL-LENDING-2025       vs the 2022 Guidelines, which it repeals
#   RBI-FRAUD-NBFC-2024            the NBFC one of three parallel Directions
#   RBI-NBFC-CREDIT-INFO-...-2025  vs the generic Jan-2025 instrument
#
# Separate entries in every case, so a citation can never attribute one
# document's requirements to the other.
EXPECTED_DOCUMENTS = 23

# The documents actually obtained. Nothing else may be cited from.
DOWNLOADED_DOCUMENT_IDS = {
    "RBI-IT-GOV-2023",
    "RBI-IT-OUTSOURCE-2023",
    "RBI-FS-OUTSOURCE-NBFC-2017",
    "RBI-DIGITAL-LENDING-2025",
    "RBI-FRAUD-NBFC-2024",
    "RBI-NBFC-CREDIT-INFO-REPORTING-2025",
}


def test_manifest_contains_the_expected_documents(manifest):
    assert len(manifest) == EXPECTED_DOCUMENTS


def test_manifest_documents_are_not_rules(manifest):
    """Guards the headline misreading this architecture exists to prevent.

    19 documents is not 19 rules. Each document yields many requirements,
    which is why requirement extraction is a separate layer keyed on
    document_id + section rather than a flat rule list.
    """
    for document in manifest.all():
        assert document.document_id
        # A document carries no rule fields; it is an identity record.
        assert not hasattr(document, "evaluation")
        assert not hasattr(document, "technical_finding_ref")


def test_every_manifest_document_has_required_metadata(manifest):
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in raw["documents"]:
        problems = validate_document_entry(entry)
        assert problems == [], f"{entry.get('document_id')}: {problems}"
        for key in REQUIRED_DOCUMENT_KEYS:
            assert key in entry, f"{entry['document_id']} missing {key}"


def test_manifest_tiers_match_the_specification(manifest):
    # tier_0 gained the 2025 Digital Lending Directions; tier_1 gained the
    # NBFC outsourcing, fraud-risk and credit-information instruments.
    assert len(manifest.by_tier(TIER_0)) == 9
    assert len(manifest.by_tier(TIER_1)) == 13
    assert len(manifest.by_tier(TIER_2)) == 1
    assert (
        len(manifest.by_tier(TIER_0))
        + len(manifest.by_tier(TIER_1))
        + len(manifest.by_tier(TIER_2))
        == EXPECTED_DOCUMENTS
    ), "every document must sit in exactly one tier"


def test_manifest_statuses_are_from_the_declared_vocabularies(manifest):
    for document in manifest.all():
        assert document.regulatory_status in VALID_REGULATORY_STATUS
        assert document.source_status in VALID_SOURCE_STATUS


def test_document_ids_are_unique(manifest):
    ids = manifest.document_ids()
    assert len(ids) == len(set(ids))


def test_duplicate_document_id_is_rejected():
    from app.rbi.manifest import RBIDocument

    def make(document_id):
        return RBIDocument(
            document_id=document_id, source="RBI", title="t", circular_number=None,
            issued_date=None, effective_date=None, last_updated=None,
            priority=TIER_0, regulatory_status="current", source_url=None,
            source_status="not_downloaded", local_path=None, storage_dir=None,
            applies_to=(), excluded=(), applicability_verified=False,
            applicability_note=None, requires_conditions={}, domain=(),
            supersedes=(), superseded_by=(), related_documents=(),
            project_mappings=(), retrieval_concepts=(), provisional=False,
            provisional_note=None, scope_limit=None,
        )

    with pytest.raises(ManifestError, match="duplicate document_id"):
        RBIManifest([make("X"), make("X")])


# ===========================================================================
# No fabrication: absent sources stay absent
# ===========================================================================


def test_only_the_obtained_documents_are_marked_downloaded(manifest):
    """Exactly the six obtained documents are citable -- and no others.

    The point of the assertion is not the number. It is that citability is
    driven by whether the file was actually obtained, so a document cannot
    become quotable by having its identity recorded more confidently.
    """
    assert {d.document_id for d in manifest.citable()} == DOWNLOADED_DOCUMENT_IDS
    assert len(manifest.unresolved()) == EXPECTED_DOCUMENTS - len(
        DOWNLOADED_DOCUMENT_IDS
    )


def test_every_downloaded_document_has_a_local_file_that_exists(manifest):
    """'downloaded' must mean the bytes are on disk, not that we meant to."""
    repo_root = MANIFEST_PATH.parents[2]
    for document in manifest.citable():
        assert document.local_path, document.document_id
        path = repo_root / document.local_path
        assert path.is_file(), f"{document.document_id}: missing {path}"
        assert path.stat().st_size > 0


def test_no_document_that_was_not_obtained_claims_a_source_url(manifest):
    """An invented URL looks resolvable and is not.

    A URL is allowed only where the document was actually obtained and the
    URL verified against it. For everything else the honest value is null --
    a plausible-looking link could send an ingestion run at the wrong
    instrument, which is worse than having no link at all.
    """
    for document in manifest.unresolved():
        assert document.source_url is None, document.document_id


def test_the_two_outsourcing_instruments_are_kept_distinct(manifest):
    """The bank and NBFC outsourcing circulars must never be merged.

    They are separate instruments eleven years apart, and their entity scope
    is not merely different but opposite: the 2006 one explicitly excludes
    NBFCs, the 2017 one is addressed to them. Collapsing them would let a
    finding against an NBFC cite a document that says it does not apply to
    NBFCs -- and it would look perfectly well-sourced.
    """
    banks = manifest.get("RBI-FINANCIAL-SERVICES-OUTSOURCE")
    nbfcs = manifest.get("RBI-FS-OUTSOURCE-NBFC-2017")

    assert banks is not None and nbfcs is not None
    assert banks.issued_date != nbfcs.issued_date
    assert "nbfc" in banks.excluded
    assert "nbfc" in nbfcs.applies_to
    assert "nbfc" not in banks.applies_to
    # Only the one actually obtained may be cited.
    assert nbfcs.is_citable() is True
    assert banks.is_citable() is False


def test_the_nbfc_outsourcing_circular_is_not_claimed_to_be_current(manifest):
    """A 2017 circular is not current merely because we have the file.

    Whether it is still the operative NBFC outsourcing instrument has not
    been verified, so it is recorded non-operative in both vocabularies. The
    manifest has no "unverified" status, and of the values it does have,
    only a non-current one cannot outrank a genuinely current source.
    """
    document = manifest.get("RBI-FS-OUTSOURCE-NBFC-2017")

    assert document.is_operative() is False
    assert document.raw["is_current"] is False
    assert document.raw.get("regulatory_status_note"), (
        "a conservatively-assigned status must say it was assigned "
        "conservatively, or a later reader will take it as established"
    )


def test_documents_with_unresolved_exact_urls_are_flagged(manifest):
    """The three the specification flags must be distinguishable."""
    flagged = {
        d.document_id
        for d in manifest.all()
        if d.source_status in ("exact_url_unresolved", "current_version_unresolved")
    }
    assert flagged == {
        "RBI-DLG-2023",
        "RBI-FINANCIAL-SERVICES-OUTSOURCE",
        "RBI-PRUDENTIAL-ADVANCES",
    }


def test_prudential_advances_has_no_invented_issue_date(manifest):
    """Its date is unresolved; a plausible one must not be supplied."""
    document = manifest.get("RBI-PRUDENTIAL-ADVANCES")
    assert document.issued_date is None
    assert document.source_status == "current_version_unresolved"
    assert "unresolved" in (document.raw.get("issued_date_note") or "").lower()


def test_unresolved_documents_are_not_citable(manifest):
    for document in manifest.unresolved():
        assert document.is_citable() is False
        assert document.blocking_reason()


def test_coverage_report_does_not_claim_readiness(manifest):
    """Three of twenty is progress, and still nowhere near ready.

    Three tier_0 documents are now citable, which is exactly the kind of
    partial progress that could be misreported as coverage. The report must
    keep saying the corpus is not ready, because six tier_0 documents are
    still missing.
    """
    report = manifest.coverage_report()
    assert report["total_documents"] == EXPECTED_DOCUMENTS
    assert report["citable_documents"] == len(DOWNLOADED_DOCUMENT_IDS)
    assert report["tier_0_citable"] == 3
    assert report["tier_0_total"] == 9
    assert report["corpus_ready_for_compliance"] is False


def test_draft_source_is_marked_provisional(manifest):
    mrm = manifest.get("RBI-MRM-DRAFT-2026")
    assert mrm.regulatory_status == "draft"
    assert mrm.is_provisional() is True
    assert mrm.is_operative() is False
    assert "draft" in mrm.provisional_note.lower()


def test_committee_report_is_not_binding(manifest):
    free_ai = manifest.get("RBI-FREE-AI-REPORT-2025")
    assert free_ai.regulatory_status == "committee_report"
    assert free_ai.is_operative() is False
    assert free_ai.is_provisional() is True


def test_historical_sources_are_marked_historical(manifest):
    for document_id in ("RBI-FPC-LENDERS", "RBI-FINANCIAL-SERVICES-OUTSOURCE"):
        assert manifest.get(document_id).regulatory_status == "historical"
        assert manifest.get(document_id).is_operative() is False

    assert manifest.get("RBI-CREDIT-RISK-GUIDANCE-2002").regulatory_status == "to_be_superseded"
    assert manifest.get("RBI-MODEL-RISK-CREDIT-DRAFT-2024").regulatory_status == "superseded_in_scope"


def test_historical_source_does_not_override_current_source(manifest):
    """Authority ranking must place current sources above stale ones."""
    documents = [
        manifest.get("RBI-CREDIT-RISK-GUIDANCE-2002"),   # to_be_superseded
        manifest.get("RBI-MODEL-RISK-CREDIT-DRAFT-2024"),  # superseded_in_scope
        manifest.get("RBI-IT-GOV-2023"),                  # current
        manifest.get("RBI-MRM-DRAFT-2026"),               # draft
    ]
    ranked = rank_by_authority(documents)

    assert ranked[0].document_id == "RBI-IT-GOV-2023", "a current source must rank first"
    stale = [d.document_id for d in ranked]
    assert stale.index("RBI-IT-GOV-2023") < stale.index("RBI-CREDIT-RISK-GUIDANCE-2002")
    assert stale.index("RBI-IT-GOV-2023") < stale.index("RBI-MODEL-RISK-CREDIT-DRAFT-2024")


def test_supersession_is_recorded_both_ways(manifest):
    mrm = manifest.get("RBI-MRM-DRAFT-2026")
    assert "RBI-MODEL-RISK-CREDIT-DRAFT-2024" in mrm.supersedes
    assert "RBI-MRM-DRAFT-2026" in manifest.get("RBI-MODEL-RISK-CREDIT-DRAFT-2024").superseded_by


# ===========================================================================
# Applicability engine
# ===========================================================================


def test_excluded_entity_does_not_receive_rule(manifest):
    """The specification's own example: base-layer NBFC vs IT-GOV-2023."""
    decision = evaluate_applicability(
        manifest.get("RBI-IT-GOV-2023"),
        AssessmentContext(regulated_entity_type="nbfc_base_layer"),
    )
    assert decision.applicability == DOES_NOT_APPLY
    assert "exclusion list" in decision.reasons[0]


def test_exclusion_beats_inclusion(manifest):
    """An excluded entity stays excluded even if it also matches applies_to."""
    document = manifest.get("RBI-IT-GOV-2023")
    assert "nbfc" in document.applies_to
    assert "nbfc_base_layer" in document.excluded

    assert (
        evaluate_applicability(
            document, AssessmentContext(regulated_entity_type="nbfc_base_layer")
        ).applicability
        == DOES_NOT_APPLY
    )


def test_digital_lending_rule_requires_digital_lending_applicability(manifest):
    """A physical-channel credit model must not attract digital-lending rules."""
    physical = AssessmentContext(
        regulated_entity_type="nbfc",
        model_use_case="credit_scoring",
        digital_vs_physical="physical",
    )
    decision = evaluate_applicability(manifest.get("RBI-DIGITAL-LENDING-2022"), physical)
    assert decision.applicability == DOES_NOT_APPLY
    assert "digital" in decision.reasons[-1]


def test_digital_lending_undeclared_channel_is_unclear_not_applied(manifest):
    """Not declaring the channel must not be read as 'digital'."""
    decision = evaluate_applicability(
        manifest.get("RBI-DIGITAL-LENDING-2022"),
        AssessmentContext(regulated_entity_type="nbfc"),
    )
    assert decision.applicability == APPLICABILITY_UNCLEAR


def test_third_party_rule_requires_third_party_context(manifest):
    """DLG needs an actual third-party/LSP arrangement."""
    decision = evaluate_applicability(
        manifest.get("RBI-DLG-2023"),
        AssessmentContext(
            regulated_entity_type="nbfc",
            digital_vs_physical="digital",
            third_party_dependency=False,
        ),
    )
    assert decision.applicability == DOES_NOT_APPLY
    assert "third_party_dependency" in decision.reasons[-1]


def test_bank_only_document_does_not_apply_to_nbfc(manifest):
    for document_id in (
        "RBI-COMPLIANCE-FUNCTION-2020",
        "RBI-IR-ADVANCES-2016",
        "RBI-OPERATIONAL-RISK-2023",
    ):
        decision = evaluate_applicability(
            manifest.get(document_id), AssessmentContext(regulated_entity_type="nbfc")
        )
        assert decision.applicability == DOES_NOT_APPLY, document_id


def test_unverified_applicability_is_not_treated_as_verified(manifest):
    """Matching transcribed metadata is not confirming it against the source."""
    document = manifest.get("RBI-MRM-DRAFT-2026")
    assert "nbfc" in document.applies_to
    assert document.applicability_verified is False

    decision = evaluate_applicability(
        document, AssessmentContext(regulated_entity_type="nbfc")
    )
    assert decision.applicability == APPLICABILITY_UNCLEAR
    assert "not verified" in decision.reasons[-1]
    assert decision.verified is False


def test_no_document_currently_yields_APPLIES(manifest):
    """Every document is unverified, so nothing can be asserted as applying.

    This is the honest state of the corpus today. It is asserted so that the
    day verification lands, this test fails and forces a review rather than
    the change passing unnoticed.
    """
    context = AssessmentContext(
        regulated_entity_type="nbfc",
        digital_vs_physical="digital",
        third_party_dependency=True,
        loan_type="personal_loan",
    )
    outcomes = {
        evaluate_applicability(d, context).applicability for d in manifest.all()
    }
    assert APPLIES not in outcomes


def test_undeclared_entity_type_is_unclear(manifest):
    decision = evaluate_applicability(manifest.get("RBI-IT-GOV-2023"), AssessmentContext())
    assert decision.applicability == APPLICABILITY_UNCLEAR


def test_applicability_filter_excludes_irrelevant_rules(manifest):
    """A physical NBFC credit model should not carry the whole corpus."""
    context = AssessmentContext(
        regulated_entity_type="nbfc",
        model_use_case="credit_scoring",
        digital_vs_physical="physical",
        third_party_dependency=False,
    )
    kept = filter_applicable(manifest.all(), context)
    kept_ids = {d.document_id for d in kept}

    assert len(kept) < 19, "filtering removed nothing"
    for excluded in (
        "RBI-DIGITAL-LENDING-2022",
        "RBI-DLG-2023",
        "RBI-COMPLIANCE-FUNCTION-2020",
        "RBI-OPERATIONAL-RISK-2023",
    ):
        assert excluded not in kept_ids, f"{excluded} should have been filtered out"


def test_effective_date_in_the_future_does_not_apply(manifest):
    """A direction not yet in force does not bind."""
    document = manifest.get("RBI-IT-OUTSOURCE-2023")
    assert document.effective_date == "2023-10-01"

    decision = evaluate_applicability(
        document,
        AssessmentContext(regulated_entity_type="nbfc", as_of_date="2023-05-01"),
    )
    assert decision.applicability == DOES_NOT_APPLY
    assert "takes effect" in decision.reasons[-1]


def test_applicability_decision_carries_its_reasoning(manifest):
    decision = evaluate_applicability(
        manifest.get("RBI-IT-GOV-2023"),
        AssessmentContext(regulated_entity_type="nbfc_base_layer"),
    )
    payload = decision.to_dict()
    assert payload["reasons"]
    assert payload["applicability"] == DOES_NOT_APPLY
    assert payload["document_id"] == "RBI-IT-GOV-2023"


# ===========================================================================
# Requirement model + deterministic evidence matching
# ===========================================================================


def test_requirement_statuses_are_the_six_specified():
    assert set(REQUIREMENT_STATUSES) == {
        "PASS",
        "FAIL",
        "PARTIAL",
        "NOT_ASSESSED",
        "APPLICABILITY_UNCLEAR",
        "EVIDENCE_MISSING",
    }


def test_requirement_vocabulary_does_not_touch_the_six_rule_engine():
    """The existing engine's vocabulary must be untouched by this work."""
    from app.config.thresholds import VALID_STATUSES

    assert set(VALID_STATUSES) == {"PASS", "WARNING", "FAIL", "PENDING"}
    assert "WARNING" not in REQUIREMENT_STATUSES
    assert "EVIDENCE_MISSING" not in VALID_STATUSES


def test_no_requirements_can_be_extracted_from_absent_sources(manifest):
    """Requirement text belongs to the document; no document is present."""
    for document in manifest.all():
        assert requirements_for_document(document) == []


def test_absent_source_produces_visible_not_assessed_not_silence(manifest):
    """A skipped regulation looks identical to a passing one. So none is skipped."""
    context = AssessmentContext(regulated_entity_type="nbfc")
    findings = assess_corpus(manifest.all(), context, {})

    assert findings, "the corpus produced no findings at all"
    assert all(f.status == NOT_ASSESSED for f in findings)
    for finding in findings:
        assert finding.reason
        assert finding.citable is False
        assert finding.document_id


def test_missing_evidence_is_not_pass():
    """The single most important rule in the whole architecture."""
    from app.rbi.manifest import RBIDocument

    document = RBIDocument(
        document_id="TEST-DOC", source="RBI", title="Fixture", circular_number=None,
        issued_date=None, effective_date=None, last_updated=None, priority=TIER_0,
        regulatory_status="current", source_url=None, source_status=SOURCE_DOWNLOADED,
        local_path="fixture.txt", storage_dir=None, applies_to=("nbfc",), excluded=(),
        applicability_verified=True, applicability_note=None, requires_conditions={},
        domain=("model_risk_management",), supersedes=(), superseded_by=(),
        related_documents=(), project_mappings=(), retrieval_concepts=(),
        provisional=False, provisional_note=None, scope_limit=None,
    )
    requirement = RegulatoryRequirement(
        requirement_id="TEST-DOC:1.1",
        document_id="TEST-DOC",
        requirement="The model must be independently validated.",
        source=SourceLocation(document_id="TEST-DOC", section_id="1.1", page=4),
        evidence_required=(
            EvidenceRequirement(
                evidence_type="independent_validation_record",
                description="A validation record for this model.",
                finding_ref="validation.record",
            ),
        ),
    )

    finding = assess_requirement(
        requirement, document, AssessmentContext(regulated_entity_type="nbfc"), {}
    )
    assert finding.status == EVIDENCE_MISSING
    assert finding.status != PASS
    assert "not compliance" in finding.reason


def test_partial_when_some_evidence_is_present():
    from app.rbi.manifest import RBIDocument

    document = RBIDocument(
        document_id="TEST-DOC", source="RBI", title="Fixture", circular_number=None,
        issued_date=None, effective_date=None, last_updated=None, priority=TIER_0,
        regulatory_status="current", source_url=None, source_status=SOURCE_DOWNLOADED,
        local_path="fixture.txt", storage_dir=None, applies_to=("nbfc",), excluded=(),
        applicability_verified=True, applicability_note=None, requires_conditions={},
        domain=(), supersedes=(), superseded_by=(), related_documents=(),
        project_mappings=(), retrieval_concepts=(), provisional=False,
        provisional_note=None, scope_limit=None,
    )
    requirement = RegulatoryRequirement(
        requirement_id="TEST-DOC:2.1",
        document_id="TEST-DOC",
        requirement="Monitoring and validation must both be performed.",
        source=SourceLocation(document_id="TEST-DOC", section_id="2.1", page=9),
        evidence_required=(
            EvidenceRequirement("monitoring_result", "monitoring", "monitoring.status"),
            EvidenceRequirement("validation_record", "validation", "validation.record"),
        ),
    )
    context = AssessmentContext(regulated_entity_type="nbfc")

    partial = assess_requirement(requirement, document, context, {"monitoring": {"status": "PASS"}})
    assert partial.status == PARTIAL
    assert "still missing" in partial.reason

    complete = assess_requirement(
        requirement,
        document,
        context,
        {"monitoring": {"status": "PASS"}, "validation": {"record": "VAL-1"}},
    )
    assert complete.status == PASS
    assert len(complete.evidence_found) == 2


def test_compliance_finding_contains_source_citation(manifest):
    context = AssessmentContext(regulated_entity_type="nbfc")
    findings = assess_corpus(manifest.all(), context, {})
    for finding in findings:
        assert finding.source.document_id, "every finding must name its document"


def test_compliance_finding_contains_document_id(manifest):
    findings = assess_corpus(
        manifest.all(), AssessmentContext(regulated_entity_type="nbfc"), {}
    )
    for finding in findings:
        payload = finding.to_dict()
        assert payload["document_id"]
        assert payload["source"]["document_id"] == payload["document_id"]


def test_source_location_without_page_or_section_is_not_locatable():
    """An un-anchored citation must admit it cannot be checked."""
    assert SourceLocation(document_id="D").is_locatable() is False
    assert SourceLocation(document_id="D", page=3).is_locatable() is True
    assert SourceLocation(document_id="D", section_id="4.2").is_locatable() is True


def test_findings_carry_run_identity(manifest):
    findings = assess_corpus(
        manifest.all(),
        AssessmentContext(regulated_entity_type="nbfc"),
        {},
        model_id="german-credit-random-forest",
        assurance_run_id="run-abc",
    )
    for finding in findings:
        assert finding.model_id == "german-credit-random-forest"
        assert finding.assurance_run_id == "run-abc"


def test_summary_reports_zero_assessed_for_an_empty_corpus(manifest):
    """A corpus with no sources must score 0.0, not look healthy."""
    findings = assess_corpus(
        manifest.all(), AssessmentContext(regulated_entity_type="nbfc"), {}
    )
    summary = summarise(findings)

    assert summary["assessed_fraction"] == 0.0
    assert summary["conclusive"] == 0
    assert summary["citable_findings"] == 0
    assert all(summary["by_status"][s] == 0 for s in CONCLUSIVE_STATUSES)


def test_inapplicable_document_is_not_counted_as_a_failure(manifest):
    """DOES_NOT_APPLY is not FAIL; an irrelevant rule is not a breach."""
    context = AssessmentContext(
        regulated_entity_type="nbfc", digital_vs_physical="physical"
    )
    findings = assess_corpus(manifest.all(), context, {})
    document_ids = {f.document_id for f in findings}

    assert "RBI-DIGITAL-LENDING-2022" not in document_ids
    assert not any(f.status == "FAIL" for f in findings)
