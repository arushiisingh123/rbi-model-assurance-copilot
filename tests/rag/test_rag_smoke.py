"""Phase 0 test for the RAG smoke test pipeline (owner: Nidhi).

This checks the mechanics (chunk -> embed -> index -> retrieve ->
attribute) work end to end. It does NOT verify regulatory accuracy of
the placeholder text -- see data/rbi_sources/PLACEHOLDER_NOT_REAL_RBI_TEXT.txt
for why a human must swap in a real, approved RBI document.
"""
from app.rag.smoke_test import chunk_text, run_smoke_test


def test_chunk_text_splits_into_multiple_chunks():
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, chunk_size_words=40)
    assert len(chunks) == 3


def test_run_smoke_test_returns_attributed_result():
    result = run_smoke_test()
    assert result["retrieved_text"]
    assert result["source"] == "PLACEHOLDER_NOT_REAL_RBI_TEXT.txt"
    assert result["num_chunks_indexed"] >= 1
