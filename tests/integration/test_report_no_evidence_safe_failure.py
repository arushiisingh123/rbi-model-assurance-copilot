"""Exercises the NOT_FOUND safe-failure path through the real pipeline + real IsolatedRAGRetriever, not hand-crafted fixtures.

Scope note: this file covers only the NOT_FOUND branch -- that an unsupported
regulatory claim is stripped when no evidence backs it. That the real retriever
can reach RETRIEVED at all is asserted in ``tests/report/test_generate_report.py``
(the C1 regression tests). Both halves are needed: before the C1 fix, real
retrieval raised TypeError into the broad fallback, so all five sections became
NOT_FOUND and this file passed for the wrong reason.
"""
from app.api.schemas import ReportResult
from app.report.generate import (
    SAFE_FALLBACK_TEXT,
    generate_report,
)
from tests.report.test_generate_report import FakeGroqClient


def _finds_nothing(*, query):
    """A retriever that always returns text no section's relevance bar accepts."""
    return {
        "query": query,
        "retrieved_text": "General administrative correspondence of no regulatory content.",
        "source": "irrelevant-source",
        "chunk_index": 0,
    }


def test_not_found_safe_failure_end_to_end(real_pipeline, monkeypatch):
    """A section that retrieved nothing must strip the model's regulatory claim.

    Driven by a retriever that finds nothing, rather than by the live corpus.
    This previously relied on the corpus being a single 2014 IRAC excerpt, in
    which ML governance vocabulary does not appear, so four of five sections
    were NOT_FOUND as a side effect. The corpus is now six current RBI
    Directions and the section queries ask about the obligations they impose,
    so all five sections retrieve -- and this property stopped being exercised
    at exactly the moment retrieval got better.

    The property itself is unchanged and is the one that matters: when there
    is no evidence, a hallucinated regulatory claim must not survive.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    hallucinated = {
        k: "This finding means the model must comply with RBI circular clause 2.1 and the governing regulation mandates action."
        for k in ["model", "explainability", "fairness", "drift", "compliance"]
    }

    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=FakeGroqClient(canned_response_dict=hallucinated),
        retrieval_fn=_finds_nothing,
    )

    validated = ReportResult(**out)
    assert validated is not None

    not_found_sections = [
        s for s in out["sections"]
        if s["retrieved_evidence"]["evidence_status"] == "NOT_FOUND"
    ]
    assert len(not_found_sections) == 5, "every section should have found nothing here"

    for s in not_found_sections:
        ev = s["retrieved_evidence"]
        interp = s["llm_interpretation"]
        assert ev["evidence_status"] == "NOT_FOUND"
        assert ev["citations"] == []
        assert interp["regulatory_basis"] == "none"
        # The hallucinated regulatory claim must have been replaced with SAFE_FALLBACK_TEXT
        assert interp["text"] == SAFE_FALLBACK_TEXT
        assert "RBI" not in interp["text"]
        assert "§" not in interp["text"]

    coverage = out["evidence_coverage"]
    assert coverage["not_found"] == 5
    assert coverage["retrieved"] + coverage["not_found"] == coverage["total"] == 5

    # No section may have "cited_evidence" (the regulatory corpus is curated,
    # not complete, so a narrative is never presented as regulatory authority)
    for s in out["sections"]:
        assert s["llm_interpretation"]["regulatory_basis"] != "cited_evidence"


def test_the_live_corpus_now_grounds_the_report(real_pipeline, monkeypatch):
    """The C1 counterpart: against the real corpus, sections DO retrieve.

    Guards the symptom the previous version of the test above was written for
    -- every section NOT_FOUND because retrieval was broken -- now that a
    healthy result is no longer "four of five NOT_FOUND".
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=FakeGroqClient(),
        retrieval_fn=None,
    )

    assert out["evidence_coverage"]["retrieved"] >= 3, (
        "the report should reach the current RBI Directions for most sections"
    )
    for section in out["sections"]:
        for citation in section["retrieved_evidence"]["citations"]:
            # Structured, and pointing at a place in a real document.
            assert citation["page"] is not None
            assert citation["source_clause"]
            assert citation["reference_number"]
        # Retrieval still never becomes regulatory authority.
        assert section["llm_interpretation"]["regulatory_basis"] != "cited_evidence"


def test_retrieval_exception_fails_safe(real_pipeline):
    """Ensure that if the retrieval backend raises an exception, generate_report fails safely."""
    def boom(query: str):
        raise RuntimeError("retrieval backend down")

    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=FakeGroqClient(),
        retrieval_fn=boom,
    )

    validated = ReportResult(**out)
    assert validated is not None
    assert len(out["sections"]) == 5

    for s in out["sections"]:
        assert s["retrieved_evidence"]["evidence_status"] == "NOT_FOUND"
        assert s["llm_interpretation"]["regulatory_basis"] == "none"
