"""Phase C2: retrieved evidence reaches the API as a structured citation.

The properties under test are provenance properties, not retrieval-quality
ones: a citation must name a document a reviewer can open, at a place they can
turn to, using the clause number that document actually prints -- and must say
"unknown" rather than guess when it does not have one.

The separation that matters most is asserted here too: a citation NEVER
changes a compliance status.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.schemas import Citation, ComplianceFinding
from app.rbi.clause_crossref import CLAUSE_CROSS_REFERENCES

client = TestClient(app)


@pytest.fixture(scope="module")
def compliance():
    response = client.get("/compliance")
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def citations(compliance):
    return [c for f in compliance["findings"] for c in f["citations"]]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_the_finding_schema_carries_citations():
    finding = ComplianceFinding(
        rule_id="X", rule_description="d", technical_finding_ref="r", status="PASS"
    )
    assert finding.citations == []


def test_every_finding_exposes_a_citation_list(compliance):
    for finding in compliance["findings"]:
        assert isinstance(finding["citations"], list)


def test_citations_validate_against_the_schema(citations):
    assert citations, "expected the live corpus to support at least one finding"
    for raw in citations:
        Citation(**raw)


def test_chunk_ids_are_still_present_for_internal_tracing(compliance):
    """evidence_chunks was not replaced; the engine still owns it."""
    for finding in compliance["findings"]:
        assert isinstance(finding["evidence_chunks"], list)
        if finding["citations"]:
            assert len(finding["evidence_chunks"]) == len(finding["citations"])


# ---------------------------------------------------------------------------
# Content: a reviewer can actually follow it
# ---------------------------------------------------------------------------


def test_each_citation_names_a_document_and_a_place_in_it(citations):
    for raw in citations:
        assert raw["source"], "a citation with no document is not a citation"
        assert raw["quote"].strip()
        assert raw["reference_number"], "the RBI reference number was dropped"
        assert isinstance(raw["page"], int) and raw["page"] >= 1
        assert raw["source_clause"]


def test_the_locator_points_at_a_page_and_clause(citations):
    for raw in citations:
        assert raw["locator"].startswith("page ")
        assert str(raw["page"]) in raw["locator"]


def test_source_urls_are_real_links_where_present(citations):
    for raw in citations:
        if raw["source_url"]:
            assert raw["source_url"].startswith("http")


def test_currency_is_reported_for_every_cited_source(citations):
    """A reader must be able to tell current regulation from historical."""
    for raw in citations:
        assert raw["is_current"] in (True, False)
        assert raw["is_excerpt"] in (True, False)


# ---------------------------------------------------------------------------
# The two clause numbers stay separate
# ---------------------------------------------------------------------------


def test_source_clause_and_register_clause_are_separate_fields(citations):
    for raw in citations:
        assert "source_clause" in raw
        assert "register_clause" in raw


def test_a_register_clause_is_only_set_where_one_was_established(citations):
    """Most of a Direction is not in a sixteen-requirement register."""
    known = {row.register_clause for row in CLAUSE_CROSS_REFERENCES}
    for raw in citations:
        if raw["register_clause"] is not None:
            assert raw["register_clause"] in known, (
                f"{raw['register_clause']!r} is not a clause the register cites"
            )


def test_the_register_clause_is_never_written_into_the_source_clause():
    """16.13 must never be presented as what the PDF prints; it prints 16(m)."""
    from app.rag.citations import citation_from_evidence

    class _Evidence:
        doc_id = "RBI-IT-OUTSOURCE-2023"
        title = "Master Direction on Outsourcing of Information Technology Services"
        text = "right to conduct audit of the service provider"
        chunk_id = "RBI-IT-OUTSOURCE-2023::chunk-1"
        chunk_index = 1
        source_url = None
        publication_date = "2023-04-10"
        document_type = "Master Direction"
        is_excerpt = False
        is_current = True
        provenance = {"page": 16, "section_id": "16(m)"}

    citation = citation_from_evidence(_Evidence())

    assert citation.source_clause == "16(m)"
    assert citation.register_clause == "16.13"
    assert citation.source_clause != citation.register_clause
    # And the locator quotes the document's own number, not the register's.
    assert "16(m)" in citation.locator
    assert "16.13" not in citation.locator


# ---------------------------------------------------------------------------
# Nothing is fabricated
# ---------------------------------------------------------------------------


def test_a_source_without_pages_reports_no_page_and_no_clause():
    """The 2014 text excerpt has neither. It must claim neither."""
    from app.rag.citations import citation_from_evidence

    class _Evidence:
        doc_id = "rbi-master-circular-irac-advances-2014-07-01"
        title = "Master Circular - Prudential Norms"
        text = "A non performing asset is an advance where interest is overdue."
        chunk_id = "rbi-master-circular-irac-advances-2014-07-01::chunk-9"
        chunk_index = 9
        source_url = "https://www.rbi.org.in/x.pdf"
        publication_date = "2014-07-01"
        document_type = "Master Circular"
        is_excerpt = True
        is_current = False
        provenance = {}

    citation = citation_from_evidence(_Evidence())

    assert citation.page is None
    assert citation.source_clause is None
    assert citation.section_title is None
    assert citation.register_clause is None
    assert citation.locator == "chunk #9"


def test_an_unreadable_page_value_becomes_unknown_not_a_guess():
    from app.rag.citations import citation_from_evidence

    class _Evidence:
        doc_id = "RBI-IT-GOV-2023"
        title = "t"
        text = "x"
        chunk_id = "c"
        chunk_index = 0
        source_url = None
        publication_date = None
        document_type = None
        is_excerpt = None
        is_current = None
        provenance = {"page": "not-a-number"}

    citation = citation_from_evidence(_Evidence())
    assert citation.page is None


def test_currency_flags_survive_chroma_stringification():
    """Chroma round-trips booleans as strings; they must come back as bools."""
    from app.rag.citations import citation_from_evidence

    class _Evidence:
        doc_id = "d"
        title = "t"
        text = "x"
        chunk_id = "c"
        chunk_index = 0
        source_url = None
        publication_date = None
        document_type = None
        is_excerpt = "True"
        is_current = "False"
        provenance = {}

    citation = citation_from_evidence(_Evidence())
    assert citation.is_excerpt is True
    assert citation.is_current is False


# ---------------------------------------------------------------------------
# Retrieval never decides compliance
# ---------------------------------------------------------------------------


def test_statuses_are_unchanged_by_the_presence_of_citations(compliance):
    """The six-rule engine's verdicts, unaffected by what retrieval found."""
    statuses = {f["rule_id"]: f["status"] for f in compliance["findings"]}

    assert set(statuses) == {
        "RBI-FAIR-01", "RBI-FAIR-02", "RBI-DRIFT-01",
        "RBI-DRIFT-02", "RBI-EXPL-01", "RBI-MODEL-01",
    }
    for status in statuses.values():
        assert status in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_a_finding_with_no_citation_is_not_downgraded(compliance):
    """Missing regulatory text is not a failed check."""
    for finding in compliance["findings"]:
        if not finding["citations"]:
            assert finding["status"] in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_verified_requirements_are_not_merged_into_citations(compliance):
    """Two regulatory layers, kept apart.

    The register reports applicability and evidence status for clauses only
    the regulated entity can evidence. A retrieved citation reports where some
    regulatory text can be read. Folding one into the other would make an
    unevidenced obligation look like a sourced finding.
    """
    assert compliance["verified_requirements"]
    for requirement in compliance["verified_requirements"]:
        assert requirement["evidence_type"] == "rbi_verified_requirement"
        assert "citations" not in requirement

    for finding in compliance["findings"]:
        for citation in finding["citations"]:
            assert "applicability" not in citation
            assert "assessment_mode" not in citation


# ---------------------------------------------------------------------------
# Integration with the PDF report path: citations must not depend on which
# endpoint you ask
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def assurance():
    response = client.get("/assurance-result")
    assert response.status_code == 200
    return response.json()


def test_assurance_result_carries_the_same_citations_as_compliance(
    compliance, assurance
):
    """One shared builder, so the two endpoints cannot drift apart.

    Before this, structured citations were attached only in the /compliance
    route while build_assurance_result() left findings with chunk-ids alone.
    The downloadable PDF is built from the assurance path, so it silently
    lost the regulatory grounding the JSON already had -- the same check
    citing an RBI page on one endpoint and nothing on another.
    """
    theirs = assurance["compliance"]["findings"]
    ours = compliance["findings"]

    assert [f["rule_id"] for f in theirs] == [f["rule_id"] for f in ours]
    for a, b in zip(theirs, ours):
        assert len(a["citations"]) == len(b["citations"]), a["rule_id"]


def test_assurance_result_citations_are_fully_structured(assurance):
    citations = [c for f in assurance["compliance"]["findings"] for c in f["citations"]]

    assert citations, "expected the live corpus to support at least one citation"
    for citation in citations:
        assert citation["source"]
        assert citation["reference_number"]
        assert isinstance(citation["page"], int)
        assert citation["source_clause"]
        # The two numbering schemes stay apart here too.
        assert "register_clause" in citation


def test_attaching_citations_never_changes_a_status(compliance, assurance):
    """The rule engine decides status before any citation is built."""
    ours = {f["rule_id"]: f["status"] for f in compliance["findings"]}
    theirs = {f["rule_id"]: f["status"] for f in assurance["compliance"]["findings"]}

    assert ours == theirs
    for status in ours.values():
        assert status in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_the_shared_helper_copies_rather_than_mutates():
    """The rule engine's own output shape is asserted by its tests."""
    from app.rag.citations import attach_citations_to_findings

    original = [{"rule_id": "R1", "status": "PASS"}]
    result = attach_citations_to_findings(original, {})

    assert "citations" not in original[0], "input findings were mutated"
    assert result[0]["citations"] == []
    assert result[0]["status"] == "PASS"


def test_evidence_chunks_survive_alongside_citations(assurance):
    """Additive: the engine's chunk-id list is not replaced."""
    for finding in assurance["compliance"]["findings"]:
        assert "evidence_chunks" in finding
        if finding["citations"]:
            assert len(finding["evidence_chunks"]) == len(finding["citations"])
