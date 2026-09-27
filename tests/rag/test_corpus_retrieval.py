"""Phase B acceptance: can retrieval actually find known RBI requirements?

Parsing a PDF is not retrieval working. These tests ask the question that
matters -- given a requirement stated in business language, does the existing
retriever return the right document, at the right page, with the document's
own clause number attached?

Measured, not assumed. The thresholds below are the levels actually achieved
by the current lexical retriever over the real corpus; they exist so a change
that degrades retrieval fails here rather than being discovered in a report.
"""
import pytest

from app.rag.corpus import corpus_from_manifest
from app.rag.retrieval import (
    EVIDENCE_RETRIEVED,
    RBIRetriever,
    build_default_retriever,
    clear_index_cache,
    default_corpus,
)
from app.rag.chunking import chunk_documents
from app.rag.ingestion import load_corpus
from app.rag.vector_store import ChunkVectorStore
from app.rbi.clause_crossref import CLAUSE_CROSS_REFERENCES, crossref_for
from app.rbi.verified_requirements import VERIFIED_REQUIREMENTS

# verified_requirements and the manifest chose different ids for the same two
# documents. Mapped here rather than in either module, so neither has to know
# about the other.
MANIFEST_ID = {
    "RBI-IT-GOVERNANCE-2023": "RBI-IT-GOV-2023",
    "RBI-IT-OUTSOURCING-2023": "RBI-IT-OUTSOURCE-2023",
}

TOP_K = 5


@pytest.fixture(scope="module")
def retriever():
    documents = load_corpus(corpus_from_manifest())
    store = ChunkVectorStore.in_memory()
    store.add_chunks(chunk_documents(documents))
    return RBIRetriever(store, default_top_k=TOP_K)


def _rows(result):
    return result.get("results") or []


def _recall(retriever, *, want_page: bool):
    hits = total = 0
    for requirement in VERIFIED_REQUIREMENTS:
        xref = crossref_for(requirement.requirement.requirement_id)
        if xref is None:
            continue
        total += 1
        wanted = MANIFEST_ID.get(xref.document_id, xref.document_id)
        result = retriever.retrieve(requirement.requirement.requirement, top_k=TOP_K)
        for row in _rows(result):
            provenance = row.get("provenance") or row
            if (row.get("doc_id") or provenance.get("doc_id")) != wanted:
                continue
            if not want_page or provenance.get("page") == xref.page:
                hits += 1
                break
    return hits, total


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


def test_every_known_requirement_retrieves_something(retriever):
    for requirement in VERIFIED_REQUIREMENTS:
        if crossref_for(requirement.requirement.requirement_id) is None:
            continue
        result = retriever(query=requirement.requirement.requirement)
        assert result["evidence_status"] == EVIDENCE_RETRIEVED, (
            requirement.requirement.requirement_id
        )


def test_correct_document_is_retrieved_for_every_known_requirement(retriever):
    hits, total = _recall(retriever, want_page=False)
    assert hits == total, f"correct document for only {hits}/{total}"


def test_correct_page_is_retrieved_for_almost_every_known_requirement(retriever):
    """Measured at 15/16. Asserted as a floor, so a regression is visible.

    Not asserted as 16/16: one clause's wording is close enough to a
    neighbouring provision that a lexical retriever prefers the neighbour, and
    pretending otherwise would make this test a fiction.
    """
    hits, total = _recall(retriever, want_page=True)
    assert hits >= total - 1, f"correct page for only {hits}/{total}"


# ---------------------------------------------------------------------------
# Provenance quality of what comes back
# ---------------------------------------------------------------------------


def test_retrieved_evidence_carries_document_page_and_clause(retriever):
    """The three things a citation needs, on a real retrieval.

    The clause asserted is the PARENT (16), not the exact sub-item. Clause 16
    runs to thirteen lettered sub-items across one page, and which of them a
    fixed-width chunk starts in depends on chunk boundaries -- the top hit for
    the audit-rights wording lands in 16(k) rather than 16(m). Document, page
    and parent clause are what retrieval can actually guarantee here; pinning
    the sub-item would assert a chunking artefact as if it were a property of
    the document.
    """
    result = retriever(
        query="right to conduct audit of the service provider including sub-contractors"
    )

    assert result["evidence_status"] == EVIDENCE_RETRIEVED
    provenance = result["provenance"]
    assert provenance["doc_id"] == "RBI-IT-OUTSOURCE-2023"
    assert provenance["page"] == 16
    assert provenance["section_id"].startswith("16(")
    assert provenance["title"]
    assert provenance["reference_number"]


def test_retrieved_provenance_never_invents_a_page(retriever):
    for requirement in VERIFIED_REQUIREMENTS[:6]:
        result = retriever(query=requirement.requirement.requirement)
        for row in _rows(result):
            provenance = row.get("provenance") or row
            page = provenance.get("page")
            if page is not None:
                assert isinstance(page, int) and page >= 1


def test_the_historical_excerpt_is_no_longer_in_the_live_corpus():
    """Phase C removed it, once the report stopped depending on it.

    It is a partial 2014 excerpt of a superseded circular about asset
    classification. While the report's section queries were still written
    around it, it had to stay indexed or report coverage fell to zero; now
    that those queries ask about the obligations the current Directions
    impose, keeping it retrievable only creates the chance of citing it.

    It is NOT deleted: still on disk, still in APPROVED_CORPUS, still the
    fixture the ingestion and retrieval tests run against.
    """
    from app.rag.corpus import APPROVED_CORPUS, RBI_IRAC_ADVANCES_2014

    assert RBI_IRAC_ADVANCES_2014.doc_id not in default_corpus()
    assert RBI_IRAC_ADVANCES_2014.doc_id in APPROVED_CORPUS

    # And retrieving it explicitly still labels it honestly.
    retriever = build_default_retriever(APPROVED_CORPUS)
    result = retriever(query="non performing asset prudential norms advances")
    assert result["evidence_status"] == EVIDENCE_RETRIEVED
    assert result["is_current"] in (False, "False")
    assert result["is_excerpt"] in (True, "True")


# ---------------------------------------------------------------------------
# What the default corpus is
# ---------------------------------------------------------------------------


def test_default_corpus_is_exactly_the_manifest_documents():
    """The manifest is the canonical registry; nothing is added beside it."""
    corpus = default_corpus()
    manifest_ids = set(corpus_from_manifest().doc_ids())

    assert set(corpus.doc_ids()) == manifest_ids
    assert "rbi-master-circular-irac-advances-2014-07-01" not in corpus


def test_default_corpus_contains_no_document_that_was_not_obtained():
    from app.rbi.manifest import load_manifest

    obtained = {d.document_id for d in load_manifest().citable()}
    for source in default_corpus():
        if source.doc_id.startswith("RBI-"):
            assert source.doc_id in obtained


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_the_index_is_built_once_and_reused():
    clear_index_cache()
    first = build_default_retriever()
    second = build_default_retriever()

    # Same underlying index object -- the expensive part is not repeated.
    assert first._store is second._store or first.store is second.store


def test_a_cached_index_does_not_leak_state_between_retrievers():
    """Sharing a READ-ONLY index must not make two retrievers interact."""
    clear_index_cache()
    query = "right to conduct audit of the service provider"
    first = build_default_retriever()
    hit = first(query=query)

    second = build_default_retriever()
    miss = second(query="quantum chromodynamics basketball tournament schedule")

    assert hit["evidence_status"] == EVIDENCE_RETRIEVED
    assert miss["evidence_status"] != EVIDENCE_RETRIEVED
    # The first retriever's answers are unchanged by the second's existence.
    assert first(query=query) == hit


def test_clearing_the_cache_rebuilds(retriever):
    clear_index_cache()
    rebuilt = build_default_retriever()
    assert rebuilt(query="right to conduct audit of the service provider")[
        "evidence_status"
    ] == EVIDENCE_RETRIEVED


# ---------------------------------------------------------------------------
# The cross-reference table is evidence, not decoration
# ---------------------------------------------------------------------------


def test_every_verified_requirement_has_a_cross_reference():
    ids = {row.requirement_id for row in CLAUSE_CROSS_REFERENCES}
    assert ids == {r.requirement.requirement_id for r in VERIFIED_REQUIREMENTS}


def test_the_cross_reference_records_where_numbering_diverges():
    """10 agree, 6 do not. The split is data, not a rounding error."""
    matching = [row for row in CLAUSE_CROSS_REFERENCES if row.numbering_matches]
    diverging = [row for row in CLAUSE_CROSS_REFERENCES if not row.numbering_matches]

    assert matching and diverging
    assert len(matching) + len(diverging) == len(VERIFIED_REQUIREMENTS)
    # The known example, kept explicit so it cannot quietly change.
    audit = next(r for r in CLAUSE_CROSS_REFERENCES if r.register_clause == "16.13")
    assert audit.source_clause == "16(m)"
    assert audit.page == 16
    assert audit.numbering_matches is False


# ---------------------------------------------------------------------------
# Phase D1: the 2014 excerpt is a fixture, and cannot leak into production
# ---------------------------------------------------------------------------


def test_the_2014_excerpt_is_preserved_as_a_fixture():
    """Kept on disk and registered -- it proves the non-PDF path still works."""
    from app.rag.corpus import APPROVED_CORPUS, RBI_IRAC_ADVANCES_2014

    assert RBI_IRAC_ADVANCES_2014.doc_id in APPROVED_CORPUS
    assert RBI_IRAC_ADVANCES_2014.resolved_path().is_file()


def test_no_production_retrieval_can_return_the_2014_excerpt():
    """Its subject is the strongest possible probe: nothing else mentions NPAs.

    If the excerpt were reachable from the production corpus, a query about
    non-performing assets would find it. That it finds nothing is the proof
    that a superseded 2014 excerpt cannot be cited as current regulation.
    """
    from app.rag.corpus import RBI_IRAC_ADVANCES_2014

    clear_index_cache()
    retriever = build_default_retriever()
    result = retriever(query="non performing asset prudential norms advances npa")

    if result["evidence_status"] == EVIDENCE_RETRIEVED:
        assert result["doc_id"] != RBI_IRAC_ADVANCES_2014.doc_id, (
            "the historical excerpt is retrievable from the production corpus"
        )


def test_the_excerpt_keeps_its_labels_when_retrieved_deliberately():
    """Explicitly retrieving the fixture must still mark it historical."""
    from app.rag.corpus import APPROVED_CORPUS

    retriever = build_default_retriever(APPROVED_CORPUS)
    result = retriever(query="non performing asset prudential norms advances")

    assert result["evidence_status"] == EVIDENCE_RETRIEVED
    assert result["is_current"] in (False, "False")
    assert result["is_excerpt"] in (True, "True")
