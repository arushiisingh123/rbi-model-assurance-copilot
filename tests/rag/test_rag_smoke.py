"""Phase 0 test for the RAG smoke test pipeline (owner: Nidhi).

This checks the mechanics (chunk -> embed -> index -> retrieve ->
attribute) work end to end against the real, officially-hosted RBI
source approved in docs/decisions.md ("Phase 0 RAG smoke-test source
selected"). It does NOT verify regulatory accuracy or completeness of
the source beyond the verbatim excerpt stored in
data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt --
see that file's own header for scope limitations.
"""
from app.rag.smoke_test import chunk_text, run_smoke_test


def test_chunk_text_splits_into_multiple_chunks():
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, chunk_size_words=40)
    assert len(chunks) == 3


def test_run_smoke_test_returns_attributed_result():
    result = run_smoke_test()
    assert result["retrieved_text"]
    assert result["source"] == "RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt"
    assert result["num_chunks_indexed"] >= 1
    # The retrieval query asks about non-performing assets, which the
    # real source defines in section 2.1 -- confirm the retrieved chunk
    # is actually relevant, not just present.
    assert "non performing" in result["retrieved_text"].lower()
