"""Tests for hybrid retrieval: BM25 + vector + metadata filter + rerank.

All fixtures are local. No RBI text appears here — the chunk bodies below
are obviously synthetic test strings, not quotations, because quoting a
document this project has never obtained would be fabricating regulation.

The properties under test are the ones that make retrieval safe for
compliance use:
  - an inapplicable document cannot be retrieved at all, however well it
    matches the query text
  - a historical/draft chunk cannot outrank a current one on equal text
  - ranking is reproducible
"""

from __future__ import annotations

import pytest

from app.rag.hybrid import (
    AUTHORITY_MULTIPLIERS,
    BM25Index,
    RetrievalCandidate,
    apply_metadata_filter,
    authority_multiplier,
    hybrid_search,
    tokenize,
)


def candidate(chunk_id, text, **metadata):
    metadata.setdefault("document_id", chunk_id.split("::")[0])
    return RetrievalCandidate(chunk_id=chunk_id, text=text, metadata=metadata)


@pytest.fixture
def corpus():
    """Synthetic chunks. Wording is invented for the test, not quoted."""
    return [
        candidate(
            "DOC-CURRENT::chunk-0",
            "model validation independent review governance framework",
            regulatory_status="current",
            domain="model_risk_management",
            page=4,
            section_id="3.1",
        ),
        candidate(
            "DOC-HISTORICAL::chunk-0",
            "model validation independent review governance framework",
            regulatory_status="historical",
            domain="model_risk_management",
            page=11,
            section_id="2.2",
        ),
        candidate(
            "DOC-DRAFT::chunk-0",
            "model validation independent review governance framework",
            regulatory_status="draft",
            domain="model_risk_management",
        ),
        candidate(
            "DOC-DIGITAL::chunk-0",
            "digital lending platform consent grievance redressal borrower",
            regulatory_status="current",
            domain="digital_lending",
        ),
        candidate(
            "DOC-OPS::chunk-0",
            "operational risk capital measurement business indicator",
            regulatory_status="current",
            domain="operational_risk",
        ),
    ]


# ===========================================================================
# BM25 (the lexical leg)
# ===========================================================================


def test_bm25_ranks_the_matching_document_highest():
    index = BM25Index([
        tokenize("model validation independent review"),
        tokenize("operational risk capital measurement"),
    ])
    scores = index.scores(tokenize("independent validation"))

    assert scores[0] > scores[1]


def test_bm25_weights_rare_terms_above_common_ones():
    """The reason BM25 is here rather than raw term counting.

    'model' appears in every document and carries no discriminating power;
    'decommissioning' appears in one and should dominate the ranking.
    """
    index = BM25Index([
        tokenize("model model model model decommissioning"),
        tokenize("model model model model model"),
        tokenize("model model model model model"),
    ])
    rare = index.scores(tokenize("decommissioning"))
    common = index.scores(tokenize("model"))

    assert rare[0] > 0
    assert rare[1] == 0
    # 'model' is in every doc, so its IDF contribution is near-zero.
    assert max(rare) > max(common)


def test_bm25_scores_zero_for_an_unmatched_query():
    index = BM25Index([tokenize("alpha beta gamma")])
    assert index.scores(tokenize("nothing matches here")) == [0.0]


def test_tokenize_is_case_and_punctuation_insensitive():
    assert tokenize("Model-Risk, MANAGEMENT.") == ["model", "risk", "management"]


# ===========================================================================
# Metadata filtering happens BEFORE ranking
# ===========================================================================


def test_metadata_filter_excludes_inapplicable_documents(corpus):
    kept, rejected = apply_metadata_filter(
        corpus, exclude_document_ids={"DOC-DIGITAL", "DOC-OPS"}
    )
    kept_ids = {c.document_id for c in kept}

    assert "DOC-DIGITAL" not in kept_ids
    assert "DOC-OPS" not in kept_ids
    assert len(rejected) == 2


def test_rejections_are_returned_with_a_reason_not_silently_dropped(corpus):
    """An invisible exclusion is indistinguishable from an oversight."""
    _, rejected = apply_metadata_filter(corpus, exclude_document_ids={"DOC-OPS"})

    assert len(rejected) == 1
    _, reason = rejected[0]
    assert "excluded by applicability" in reason


def test_allowed_set_restricts_to_applicable_documents(corpus):
    kept, _ = apply_metadata_filter(corpus, allowed_document_ids={"DOC-CURRENT"})
    assert {c.document_id for c in kept} == {"DOC-CURRENT"}


def test_domain_filter_narrows_retrieval(corpus):
    kept, _ = apply_metadata_filter(corpus, domains={"digital_lending"})
    assert {c.document_id for c in kept} == {"DOC-DIGITAL"}


def test_an_excluded_document_cannot_be_retrieved_however_well_it_matches(corpus):
    """The central safety property of this pipeline.

    A strong text match must not be able to surface a regulation that does
    not apply — filtering is a gate, not a tiebreaker.
    """
    result = hybrid_search(
        "digital lending platform consent grievance redressal borrower",
        corpus,
        exclude_document_ids={"DOC-DIGITAL"},
    )
    returned = {hit["document_id"] for hit in result["hits"]}

    assert "DOC-DIGITAL" not in returned
    assert any(r["document_id"] == "DOC-DIGITAL" for r in result["audit"]["rejected"])


# ===========================================================================
# Reranking by regulatory authority
# ===========================================================================


def test_current_source_outranks_historical_on_identical_text(corpus):
    """Equal text, unequal authority — current must win.

    The three model-validation chunks are byte-identical, so text ranking
    alone would order them arbitrarily. Only the authority rerank makes the
    current source reliably first.
    """
    result = hybrid_search(
        "model validation independent review governance framework",
        corpus,
        top_k=5,
    )
    order = [hit["document_id"] for hit in result["hits"]]

    assert order[0] == "DOC-CURRENT"
    assert order.index("DOC-CURRENT") < order.index("DOC-HISTORICAL")
    assert order.index("DOC-CURRENT") < order.index("DOC-DRAFT")


def test_authority_multiplier_never_exceeds_current(corpus):
    for status, multiplier in AUTHORITY_MULTIPLIERS.items():
        assert multiplier <= 1.0, status
    assert AUTHORITY_MULTIPLIERS["current"] == 1.0
    assert AUTHORITY_MULTIPLIERS["historical"] < AUTHORITY_MULTIPLIERS["current"]
    assert AUTHORITY_MULTIPLIERS["draft"] < AUTHORITY_MULTIPLIERS["current"]


def test_excerpt_is_demoted_against_a_full_document():
    assert authority_multiplier({"regulatory_status": "current", "is_excerpt": True}) < 1.0
    assert authority_multiplier({"regulatory_status": "current"}) == 1.0


def test_unknown_status_is_not_boosted():
    assert authority_multiplier({"regulatory_status": "something_new"}) <= 1.0


# ===========================================================================
# Fusion, determinism and the audit trail
# ===========================================================================


def test_hybrid_runs_without_an_embedder(corpus):
    """Degrades to BM25 + filter + rerank, which is still a real pipeline."""
    result = hybrid_search("model validation", corpus)

    assert result["hits"]
    assert "disabled" in result["audit"]["vector_leg"]
    assert all(hit["scores"]["vector"] == 0.0 for hit in result["hits"])


def test_vector_leg_is_used_when_an_embedder_is_supplied(corpus):
    from app.rag.vector_store import DeterministicHashEmbedding

    embedder = DeterministicHashEmbedding()
    result = hybrid_search(
        "model validation independent review",
        corpus,
        embed=embedder.embed_text,
    )

    assert "enabled" in result["audit"]["vector_leg"]
    assert any(hit["scores"]["vector"] > 0 for hit in result["hits"])


def test_ranking_is_reproducible(corpus):
    """A report must not reorder between identical runs."""
    first = hybrid_search("model validation governance", corpus)
    second = hybrid_search("model validation governance", corpus)

    assert [h["chunk_id"] for h in first["hits"]] == [h["chunk_id"] for h in second["hits"]]


def test_hits_preserve_chunk_provenance(corpus):
    """Page/section must survive retrieval, or a citation cannot be anchored."""
    result = hybrid_search("model validation independent review", corpus)
    top = result["hits"][0]

    assert top["document_id"] == "DOC-CURRENT"
    assert top["metadata"]["page"] == 4
    assert top["metadata"]["section_id"] == "3.1"
    assert top["chunk_id"] == "DOC-CURRENT::chunk-0"


def test_top_k_is_respected(corpus):
    assert len(hybrid_search("model", corpus, top_k=2)["hits"]) <= 2


def test_empty_candidate_set_returns_no_hits_not_an_error():
    result = hybrid_search("anything", [])
    assert result["hits"] == []
    assert "no candidate" in result["audit"]["reason"]


def test_everything_filtered_out_returns_no_hits(corpus):
    result = hybrid_search("model validation", corpus, allowed_document_ids={"NOT-PRESENT"})
    assert result["hits"] == []
    assert result["audit"]["candidates_after_filter"] == 0


def test_audit_trail_records_what_was_considered(corpus):
    result = hybrid_search("model validation", corpus, exclude_document_ids={"DOC-OPS"})
    audit = result["audit"]

    assert audit["candidates_in"] == len(corpus)
    assert audit["candidates_after_filter"] == len(corpus) - 1
    assert audit["query"] == "model validation"
    assert audit["returned"] <= audit["ranked"]
