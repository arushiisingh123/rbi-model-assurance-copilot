"""Phase 0 RAG smoke test (owner: Nidhi).

Approved scope (see docs/decisions.md, "RAG scope for Phase 0 and
Phase 1"): prove that chunk -> embed -> index -> retrieve -> attribute
works end to end against ONE document. This is a proof of concept
only, not full RAG (no multi-document corpus, no retrieval tuning, no
LLM integration).

IMPORTANT: DOCUMENT_PATH currently points at a placeholder file that
is explicitly NOT real RBI text (see the warning inside that file).
Nidhi must replace it with a real, chosen RBI document before this
smoke test's output can be treated as real regulatory evidence.

This is a standalone script, not wired into the API yet (per TASK.md
Phase 0 scope). Run it directly:

    python -m app.rag.smoke_test
"""
import hashlib
from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

DOCUMENT_PATH = Path("data/rbi_sources/PLACEHOLDER_NOT_REAL_RBI_TEXT.txt")
CHUNK_SIZE_WORDS = 40
TEST_QUERY = "How should a credit-scoring model be validated?"


class HashingEmbeddingFunction(EmbeddingFunction):
    """Deterministic, dependency-light stand-in for a real embedding
    model. Good enough to prove the chunk/index/retrieve pipeline
    works, without requiring internet access or heavy ML libraries
    (e.g. torch) just for a Phase 0 proof of concept. Phase 3 may
    swap in a real embedding model.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "phase0-hashing-embedding"

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = [0.0] * self.dim
            for word in text.lower().split():
                digest = hashlib.md5(word.encode("utf-8")).hexdigest()
                vector[int(digest, 16) % self.dim] += 1.0
            vectors.append(vector)
        return vectors


def chunk_text(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS) -> list[str]:
    """Split text into fixed-size word chunks (simple, beginner-friendly
    chunking strategy for the Phase 0 proof of concept)."""
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size_words])
        for i in range(0, len(words), chunk_size_words)
    ]


def run_smoke_test(document_path: Path = DOCUMENT_PATH, query: str = TEST_QUERY) -> dict:
    """Run the chunk -> embed -> index -> retrieve pipeline once and
    return the top result with its source attribution."""
    text = document_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="rbi_smoke_test",
        embedding_function=HashingEmbeddingFunction(),
    )
    collection.add(
        documents=chunks,
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        metadatas=[{"source": document_path.name, "chunk_index": i} for i in range(len(chunks))],
    )

    result = collection.query(query_texts=[query], n_results=1)

    return {
        "query": query,
        "retrieved_text": result["documents"][0][0],
        "source": result["metadatas"][0][0]["source"],
        "chunk_index": result["metadatas"][0][0]["chunk_index"],
        "num_chunks_indexed": len(chunks),
    }


if __name__ == "__main__":
    outcome = run_smoke_test()
    print("RAG smoke test result:")
    print(f"  Query: {outcome['query']}")
    print(f"  Source document: {outcome['source']} (chunk #{outcome['chunk_index']})")
    print(f"  Retrieved text: {outcome['retrieved_text']}")
    print(f"  Total chunks indexed: {outcome['num_chunks_indexed']}")
