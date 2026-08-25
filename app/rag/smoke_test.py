"""Phase 0 RAG smoke test (owner: Nidhi).

Approved scope (see docs/decisions.md, "RAG scope for Phase 0 and
Phase 1"): prove that chunk -> embed -> index -> retrieve -> attribute
works end to end against ONE document. This is a proof of concept
only, not full RAG (no multi-document corpus, no retrieval tuning, no
LLM integration).

DOCUMENT_PATH points at a real, officially-hosted RBI document (see
docs/decisions.md, "Phase 0 RAG smoke-test source selected"): the RBI
Master Circular - Prudential Norms on Income Recognition, Asset
Classification and Provisioning pertaining to Advances (RBI/2014-15/74,
July 1, 2014), downloaded directly from www.rbi.org.in. Only a limited,
verbatim excerpt is stored locally -- see the source file's own header
for the official source URL, circular reference, and scope limitations.

This is a standalone script, not wired into the API yet (per TASK.md
Phase 0 scope). Run it directly:

    python -m app.rag.smoke_test
"""
import hashlib
import math
from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

DOCUMENT_PATH = Path("data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt")
CHUNK_SIZE_WORDS = 40
TEST_QUERY = "What is a non performing asset?"

EXCERPT_START_MARKER = "VERBATIM EXCERPT BEGINS BELOW"
EXCERPT_END_MARKER = "VERBATIM EXCERPT ENDS"


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
            # L2-normalize: raw word-count magnitude varies a lot between
            # chunks and swamps true overlap under plain distance, so
            # normalize to unit vectors and compare with cosine distance
            # (the standard approach for bag-of-words vectors) instead.
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


def load_document_text(document_path: Path) -> str:
    """Read the source file and return only the verbatim source excerpt,
    excluding the provenance/metadata header at the top of the file, so
    only real source text gets chunked, embedded, and indexed. Falls
    back to the full file contents if the markers aren't present."""
    full_text = document_path.read_text(encoding="utf-8")
    start = full_text.find(EXCERPT_START_MARKER)
    end = full_text.find(EXCERPT_END_MARKER)
    if start == -1 or end == -1:
        return full_text
    return full_text[start + len(EXCERPT_START_MARKER) : end]


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
    text = load_document_text(document_path)
    chunks = chunk_text(text)

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="rbi_smoke_test",
        embedding_function=HashingEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
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
