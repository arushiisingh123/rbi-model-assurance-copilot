"""The verified RBI demonstration corpus.

The risk this register carries is not a wrong calculation -- it performs none.
It is that a requirement gets attributed to a document nobody read, that an
advisory becomes a binding obligation, that a superseded source is cited as
current, or that absent organisational evidence quietly becomes a PASS.

These tests pin exactly those properties.
"""
import pytest

from app.rbi.applicability import APPLICABILITY_UNCLEAR, APPLIES
from app.rbi.requirements import (
    CONCLUSIVE_STATUSES,
    EVIDENCE_MISSING,
    NOT_ASSESSED,
    REQUIREMENT_STATUSES,
)
from app.rbi.verified_requirements import (
    ASSESSMENT_MODES,
    ATTESTATION,
    INSTRUMENT_BINDING,
    NOT_APPLICABLE,
    VERIFIED_REQUIREMENTS,
    VERIFIED_SOURCES,
    EntityProfile,
    applicable_requirements,
    assess_verified_requirements,
    requirement_by_id,
    source_by_id,
)

# A fully declared NBFC that does digital lending through an external vendor.
FULL_PROFILE = EntityProfile(
    entity_type="NBFC",
    nbfc_layer="Middle",
    digital_lending=True,
    microfinance=False,
    uses_external_model_vendor=True,
)

# A Base Layer NBFC doing neither digital lending nor vendor-hosted modelling.
NARROW_PROFILE = EntityProfile(
    entity_type="NBFC",
    nbfc_layer="Base",
    digital_lending=False,
    microfinance=False,
    uses_external_model_vendor=False,
)


# ---------------------------------------------------------------------------
# 1-2. Every live requirement is traceable to a verified source
# ---------------------------------------------------------------------------


def test_the_register_is_small_and_curated():
    """A demonstration corpus, not a coverage claim."""
    assert 1 <= len(VERIFIED_REQUIREMENTS) <= 20


@pytest.mark.parametrize("requirement", VERIFIED_REQUIREMENTS, ids=lambda r: r.requirement_id)
def test_every_requirement_carries_full_source_identity(requirement):
    """No requirement may exist without somewhere a reviewer can open."""
    assert requirement.source_url.startswith("https://")
    assert "rbi.org.in" in requirement.source_url
    assert requirement.clause, "a requirement must name its clause"
    assert requirement.requirement.source.document_title
    assert requirement.requirement.source.section_id == requirement.clause
    assert requirement.regulatory_status
    assert requirement.instrument_type in (INSTRUMENT_BINDING, "advisory")
    assert requirement.assessment_mode in ASSESSMENT_MODES


@pytest.mark.parametrize("requirement", VERIFIED_REQUIREMENTS, ids=lambda r: r.requirement_id)
def test_every_requirement_resolves_to_a_declared_source(requirement):
    source = source_by_id(requirement.document_id)

    assert source is not None, f"{requirement.document_id} is not a declared source"
    assert source.source_url == requirement.source_url
    assert source.clause_text_available, (
        "a requirement may not come from a source whose clause text was never read"
    )


@pytest.mark.parametrize("requirement", VERIFIED_REQUIREMENTS, ids=lambda r: r.requirement_id)
def test_every_requirement_carries_the_verbatim_clause_it_paraphrases(requirement):
    """The paraphrase is checkable because the quote travels with it."""
    assert requirement.quote
    assert len(requirement.quote) > 40


def test_no_requirement_comes_from_a_source_whose_text_could_not_be_read():
    """The SBR Directions are identity-verified but unreadable, so they
    contribute nothing. Encoding NBFC layer definitions from general knowledge
    would have been inventing regulation."""
    sbr = source_by_id("RBI-SBR-NBFC-2023")

    assert sbr is not None
    assert sbr.clause_text_available is False
    assert not [r for r in VERIFIED_REQUIREMENTS if r.document_id == sbr.document_id]


# ---------------------------------------------------------------------------
# 3. Missing evidence never becomes PASS
# ---------------------------------------------------------------------------


def test_no_requirement_can_return_pass():
    """Every requirement here is attestation-mode: only the entity can
    evidence it, so the platform must never conclude compliance."""
    findings = assess_verified_requirements(FULL_PROFILE)

    assert findings
    assert all(f["status"] != "PASS" for f in findings)
    assert all(f["status"] not in CONCLUSIVE_STATUSES for f in findings)


def test_an_applicable_requirement_reports_evidence_missing_not_a_verdict():
    findings = {f["requirement_id"]: f for f in assess_verified_requirements(FULL_PROFILE)}

    applicable = [f for f in findings.values() if f["applicability"] == APPLIES]
    assert applicable
    for finding in applicable:
        assert finding["status"] == EVIDENCE_MISSING
        assert finding["status"] != "FAIL"


def test_every_status_is_from_the_existing_vocabulary():
    """No second status vocabulary is introduced."""
    for finding in assess_verified_requirements(FULL_PROFILE):
        assert finding["status"] in REQUIREMENT_STATUSES


def test_every_requirement_is_attestation_and_says_why():
    for requirement in VERIFIED_REQUIREMENTS:
        assert requirement.assessment_mode == ATTESTATION
        assert requirement.limitation, "an unautomatable requirement must say why"


def test_the_creditworthiness_clause_is_not_automated_from_model_features():
    """The specific leap the spec forbids: a feature list is not proof that a
    lender obtained a borrower's economic profile."""
    requirement = requirement_by_id("RBI-DL-2025-7.i")

    assert requirement is not None
    assert requirement.assessment_mode == ATTESTATION
    assert "feature list is not proof" in requirement.limitation


# ---------------------------------------------------------------------------
# 3a. Suggestion: a next step, keyed by applicability, never a verdict
# ---------------------------------------------------------------------------


def test_every_finding_carries_a_suggestion():
    for finding in assess_verified_requirements(FULL_PROFILE):
        assert finding["suggestion"], "every finding must say what to do next"


def test_suggestion_is_keyed_by_applicability_not_status():
    """APPLICABILITY_UNCLEAR and NOT_APPLICABLE share one status
    (NOT_ASSESSED) but need different next steps -- keying on status alone
    would give them the same instruction."""
    unclear = assess_verified_requirements(EntityProfile())
    assert unclear
    assert all(f["applicability"] == APPLICABILITY_UNCLEAR for f in unclear)

    narrow = assess_verified_requirements(NARROW_PROFILE)
    not_applicable = [f for f in narrow if f["applicability"] == NOT_APPLICABLE]
    assert not_applicable

    # Both sets share status NOT_ASSESSED...
    assert all(f["status"] == NOT_ASSESSED for f in unclear)
    assert all(f["status"] == NOT_ASSESSED for f in not_applicable)
    # ...but must not share the same suggestion text.
    assert {f["suggestion"] for f in unclear} != {f["suggestion"] for f in not_applicable}


def test_applies_suggestion_routes_to_attestation():
    findings = assess_verified_requirements(FULL_PROFILE)
    applicable = [f for f in findings if f["applicability"] == APPLIES]
    assert applicable
    for finding in applicable:
        assert finding["status"] == EVIDENCE_MISSING
        assert "attestation" in finding["suggestion"].lower()


def test_unclear_suggestion_points_at_completing_the_profile():
    for finding in assess_verified_requirements(EntityProfile()):
        assert finding["applicability"] == APPLICABILITY_UNCLEAR
        assert "profile" in finding["suggestion"].lower()


def test_not_applicable_suggestion_names_no_action_and_a_reassessment_trigger():
    findings = assess_verified_requirements(NARROW_PROFILE)
    not_applicable = [f for f in findings if f["applicability"] == NOT_APPLICABLE]
    assert not_applicable
    for finding in not_applicable:
        assert "no action" in finding["suggestion"].lower()
        assert "re-assess" in finding["suggestion"].lower()


def test_no_suggestion_asserts_a_compliance_outcome():
    """A next step is not a verdict: none of the three templates may claim
    the entity does or does not comply."""
    forbidden = ("complies", "compliant", "non-compliant", "violat", "pass", "fail")
    seen_suggestions = set()
    for profile in (FULL_PROFILE, NARROW_PROFILE, EntityProfile()):
        for finding in assess_verified_requirements(profile):
            seen_suggestions.add(finding["suggestion"])
    assert seen_suggestions, "expected at least one suggestion across all profiles"
    for suggestion in seen_suggestions:
        lowered = suggestion.lower()
        for word in forbidden:
            assert word not in lowered, f"{word!r} in suggestion: {suggestion!r}"


def test_suggestion_is_a_fixed_template_not_per_requirement_text():
    """Hardcoded per outcome state, not derived from the requirement's own
    text -- every APPLIES finding in one profile must share identical
    wording, regardless of which clause it is."""
    applicable = [
        f for f in assess_verified_requirements(FULL_PROFILE) if f["applicability"] == APPLIES
    ]
    assert len(applicable) > 1, "need more than one APPLIES finding to prove this"
    assert len({f["suggestion"] for f in applicable}) == 1


# ---------------------------------------------------------------------------
# 4-5. Binding vs advisory; current vs superseded
# ---------------------------------------------------------------------------


def test_instrument_character_is_recorded_for_every_source():
    for source in VERIFIED_SOURCES:
        assert source.instrument_type in (INSTRUMENT_BINDING, "advisory")
        assert source.regulatory_status in (
            "current",
            "superseded",
            "historical",
            "draft",
            "committee_report",
        )


def test_a_superseded_source_is_distinguishable_and_carries_no_requirements():
    superseded = source_by_id("RBI-MONITORING-FRAUDS-NBFC-2016")

    assert superseded is not None
    assert superseded.regulatory_status == "superseded"
    assert not [
        r for r in VERIFIED_REQUIREMENTS if r.document_id == superseded.document_id
    ]


def test_the_supersession_relationship_is_recorded_on_the_current_source():
    current = source_by_id("RBI-FRAUD-NBFC-2024")

    assert current is not None
    assert current.regulatory_status == "current"
    assert "RBI-MONITORING-FRAUDS-NBFC-2016" in current.supersedes


def test_the_current_digital_lending_source_is_the_2025_directions():
    """The 2022 Guidelines must not be presented as the current source."""
    source = source_by_id("RBI-DIGITAL-LENDING-2025")

    assert source is not None
    assert "2025" in source.title
    assert source.regulatory_status == "current"
    assert source.issued_date == "2025-05-08"


# ---------------------------------------------------------------------------
# 6-8. Applicability
# ---------------------------------------------------------------------------


def test_requirements_are_never_universally_applied():
    """The register must demonstrate applicability, not blanket application."""
    decisions = {
        req.requirement_id: applicability
        for req, applicability, _ in applicable_requirements(NARROW_PROFILE)
    }

    assert NOT_APPLICABLE in decisions.values()
    assert APPLIES in decisions.values()


def test_digital_lending_requirements_do_not_apply_when_digital_lending_is_false():
    decisions = {
        req.requirement_id: applicability
        for req, applicability, _ in applicable_requirements(NARROW_PROFILE)
    }

    for requirement_id in ("RBI-DL-2025-5.ii", "RBI-DL-2025-5.iii", "RBI-DL-2025-7.i"):
        assert decisions[requirement_id] == NOT_APPLICABLE


def test_vendor_requirements_apply_only_when_vendor_usage_is_declared():
    vendor_ids = ("RBI-ITO-2023-4.1", "RBI-ITO-2023-13.1", "RBI-ITO-2023-16.13")

    with_vendor = {
        r.requirement_id: a for r, a, _ in applicable_requirements(FULL_PROFILE)
    }
    without_vendor = {
        r.requirement_id: a for r, a, _ in applicable_requirements(NARROW_PROFILE)
    }

    for requirement_id in vendor_ids:
        assert with_vendor[requirement_id] == APPLIES
        assert without_vendor[requirement_id] == NOT_APPLICABLE


def test_a_layer_scoped_clause_binds_only_the_layers_it_names():
    """3.1.3 names Upper and Middle Layer NBFCs, so a Base Layer NBFC is
    genuinely outside its scope -- source-backed applicability, not a guess."""
    decisions = {
        r.requirement_id: a for r, a, _ in applicable_requirements(NARROW_PROFILE)
    }
    upper = EntityProfile(entity_type="NBFC", nbfc_layer="Upper")
    upper_decisions = {
        r.requirement_id: a for r, a, _ in applicable_requirements(upper)
    }

    assert decisions["RBI-FRAUD-2024-3.1.3"] == NOT_APPLICABLE
    assert upper_decisions["RBI-FRAUD-2024-3.1.3"] == APPLIES


def test_an_undeclared_dimension_is_unclear_never_assumed():
    """"We were not told" and "it does not apply" are different answers."""
    decisions = [a for _, a, _ in applicable_requirements(EntityProfile())]

    assert set(decisions) == {APPLICABILITY_UNCLEAR}
    assert APPLIES not in decisions
    assert NOT_APPLICABLE not in decisions


def test_a_non_applicable_requirement_is_reported_not_dropped():
    """A silently skipped regulation looks exactly like one that passed."""
    findings = assess_verified_requirements(NARROW_PROFILE)

    assert len(findings) == len(VERIFIED_REQUIREMENTS)
    not_applicable = [f for f in findings if f["applicability"] == NOT_APPLICABLE]
    assert not_applicable
    for finding in not_applicable:
        assert finding["status"] == NOT_ASSESSED
        assert finding["applicability_reasons"], "a refusal must say why"


def test_nothing_is_inferred_from_a_model_or_dataset():
    """Entity characteristics are declared by the caller, never derived.

    Checked on IMPORTS rather than prose: the module legitimately discusses
    prediction drift in order to say it does NOT satisfy a fraud clause, and a
    substring scan would flag that explanation as a violation.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path("app/rbi/verified_requirements.py").read_text(encoding="utf-8")
    )

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    # It may reach the RBI layer and typing/dataclasses -- nothing else.
    for module in imported:
        assert not module.startswith("app.models"), module
        assert not module.startswith("app.drift"), module
        assert not module.startswith("app.explainability"), module
        assert not module.startswith("app.report"), module
        assert not module.startswith("app.rag"), module


# ---------------------------------------------------------------------------
# 9-11. Evidence identity and isolation
# ---------------------------------------------------------------------------


def test_findings_carry_model_and_run_identity_when_supplied():
    findings = assess_verified_requirements(
        FULL_PROFILE,
        model_id="german-credit-random-forest",
        model_version="0.1.0",
        assurance_run_id="run-abc",
    )

    for finding in findings:
        assert finding["model_id"] == "german-credit-random-forest"
        assert finding["model_version"] == "0.1.0"
        assert finding["assurance_run_id"] == "run-abc"


def test_identity_is_omitted_rather_than_nulled_when_not_supplied():
    """Matches the convention every other evidence producer follows."""
    for finding in assess_verified_requirements(FULL_PROFILE):
        assert "model_id" not in finding
        assert "assurance_run_id" not in finding


def test_two_models_regulatory_findings_stay_separable_when_pooled():
    lr = assess_verified_requirements(FULL_PROFILE, model_id="lr", assurance_run_id="r1")
    rf = assess_verified_requirements(FULL_PROFILE, model_id="rf", assurance_run_id="r2")

    pooled = lr + rf
    assert {f["model_id"] for f in pooled} == {"lr", "rf"}
    for finding in pooled:
        if finding["model_id"] == "lr":
            assert finding["assurance_run_id"] == "r1"


def test_every_finding_declares_its_own_evidence_type_and_provenance():
    for finding in assess_verified_requirements(FULL_PROFILE):
        assert finding["evidence_type"] == "rbi_verified_requirement"
        assert finding["is_mock"] is False
        assert finding["verified_on"]
        assert finding["source_url"].startswith("https://")


# ---------------------------------------------------------------------------
# 12-13. The register decides nothing by narrative
# ---------------------------------------------------------------------------


def test_the_register_never_calls_an_llm():
    import inspect

    from app.rbi import verified_requirements

    source = inspect.getsource(verified_requirements).lower()
    for forbidden in ("groq", "openai", "chat.completions", "generate_report"):
        assert forbidden not in source


def test_status_is_deterministic_across_repeated_assessment():
    first = assess_verified_requirements(FULL_PROFILE)
    second = assess_verified_requirements(FULL_PROFILE)

    assert [f["status"] for f in first] == [f["status"] for f in second]
    assert [f["applicability"] for f in first] == [f["applicability"] for f in second]
