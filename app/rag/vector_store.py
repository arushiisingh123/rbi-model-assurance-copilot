"""RBI chunk embeddings + ChromaDB vector store (owner: Nidhi). Phase 3A, Task 4.

WHAT THIS MODULE PROVIDES
    - ``DeterministicHashEmbedding`` -- turns chunk text into a fixed-width
      vector, fully offline and deterministic: no model download, no
      network call, no heavy ML dependency. This matches the convention
      set by ``app/rag/smoke_test.py``'s Phase 0 stand-in. It is a lexical
      (bag-of-words) embedding, not a semantic model; swapping in a real
      embedding model later is an internal change to this class.
    - ``ChunkVectorStore`` -- a thin wrapper over ONE ChromaDB collection
      that stores ``DocumentChunk`` text + embedding + provenance metadata,
      keeps many RBI documents in one index, and can rebuild itself from
      an approved corpus (ingest -> chunk -> index).

WHAT THIS MODULE DOES NOT DO
    - No retrieval API (ranking, citations, evidence-status, no-evidence
      handling). ``query()`` here is a low-level ChromaDB passthrough only,
      enough to confirm the index works and to give the later retrieval
      module a hook. That richer interface is the next task.
    - No LLM, no report generation.
    - No regulatory text is hard-coded anywhere. The store indexes only the
      chunks it is handed; ``rebuild_from_corpus()`` indexes only text
      ingested from approved source files.

This module does not import ``app/rag/smoke_test.py`` and does not change it.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any, Iterable, Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from app.rag.chunking import DEFAULT_CHUNK_SIZE_WORDS, DocumentChunk, chunk_documents
from app.rag.corpus import APPROVED_CORPUS, RBICorpus
from app.rag.ingestion import load_corpus


class DeterministicHashEmbedding(EmbeddingFunction):
    """Offline, deterministic lexical embedding for RBI chunk text.

    Each lowercased whitespace-separated word is hashed (SHA-256) into one
    bucket of a fixed-width vector; the vector is then L2-normalised, so
    cosine distance tracks lexical overlap. No network, no model files.

    Not a semantic model -- it cannot tell that "NPA" and "non performing
    asset" mean the same thing. It exists so the whole pipeline is testable
    end to end without pulling in ``sentence-transformers`` / ``torch`` or
    calling an embeddings API (which would be a "major technology" needing
    team approval -- CLAUDE.md section 3).
    """

    DEFAULT_DIM = 256

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def name() -> str:
        return "rbi-rag-deterministic-hash-v1"

    def get_config(self) -> dict:
        return {"dim": self._dim}

    @staticmethod
    def build_from_config(config: dict) -> "DeterministicHashEmbedding":
        return DeterministicHashEmbedding(dim=int(config["dim"]))

    def embed_text(self, text: str) -> list[float]:
        """The embedding vector for one string (unit-norm, or all-zero if
        the string has no words)."""
        vector = [0.0] * self._dim
        for word in text.lower().split():
            bucket = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % self._dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def __call__(self, input: Documents) -> Embeddings:
        return [self.embed_text(text) for text in input]


# Chroma collection metadata values must be str / int / float / bool -- no
# None, and the dict must be non-empty.
_METADATA_SCALARS = (str, int, float, bool)


def chunk_metadata(chunk: DocumentChunk) -> dict:
    """Flat, Chroma-safe metadata for one chunk.

    All provenance fields (``doc_id``, ``title``, ``source_url``,
    ``document_type``, ``publication_date``, ``effective_date``,
    ``reference_number``, ``is_excerpt``, ``is_current``, ``coverage_note``,
    ``scope_note``, ...) plus ``chunk_index`` and ``chunk_id``.

    A provenance value that is ``None`` on the source (for example
    ``effective_date`` on the 2014 excerpt) is **omitted** -- Chroma
    metadata holds no nulls, and an absent key is an honest representation
    of an absent value rather than an invented empty string. ``doc_id``,
    ``chunk_index`` and ``chunk_id`` are always present, so the dict is
    never empty. Full-fidelity provenance stays reachable via
    ``app.rag.corpus.get_source(doc_id)``.
    """
    raw: dict[str, Any] = dict(chunk.provenance)
    raw["chunk_index"] = chunk.chunk_index
    raw["chunk_id"] = chunk.chunk_id

    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if not isinstance(value, _METADATA_SCALARS):
            value = str(value)
        clean[key] = value
    return clean


class ChunkVectorStore:
    """A ChromaDB-backed index of RBI document chunks.

    Wraps a single collection. One index holds chunks from many RBI
    documents; they stay distinguishable by ``doc_id`` / ``chunk_id`` and
    can be filtered with a ``where={"doc_id": ...}`` clause. The
    historical-excerpt vs. verified-current distinction rides along in each
    chunk's ``is_excerpt`` / ``is_current`` metadata.
    """

    DEFAULT_COLLECTION_NAME = "rbi_regulatory_corpus"
    _COLLECTION_METADATA = {"hnsw:space": "cosine"}

    def __init__(
        self,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Optional[Any] = None,
        embedding: Optional[EmbeddingFunction] = None,
    ) -> None:
        self.embedding: EmbeddingFunction = embedding or DeterministicHashEmbedding()
        self._client = client or chromadb.EphemeralClient()
        self.collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding,
            metadata=self._COLLECTION_METADATA,
        )

    @classmethod
    def in_memory(
        cls,
        *,
        name_prefix: str = "rbi-rag",
        embedding: Optional[EmbeddingFunction] = None,
    ) -> "ChunkVectorStore":
        """A fresh, isolated in-memory store with a unique collection name.

        Every call gets its own ``EphemeralClient`` and a UUID-suffixed
        collection, so tests -- and concurrent report-generation runs --
        never collide over a shared collection name.
        """
        return cls(
            collection_name=f"{name_prefix}-{uuid.uuid4().hex}",
            client=chromadb.EphemeralClient(),
            embedding=embedding,
        )

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        """Index (upsert) chunks. Returns how many were written.

        Idempotent by ``chunk_id``: re-adding the same chunk overwrites it
        and does not grow the index.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunk_list],
            documents=[c.text for c in chunk_list],
            metadatas=[chunk_metadata(c) for c in chunk_list],
        )
        return len(chunk_list)

    def rebuild_from_corpus(
        self,
        corpus: RBICorpus = APPROVED_CORPUS,
        *,
        repo_root: Optional[Any] = None,
        chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    ) -> dict:
        """Clear the index and rebuild it from an approved corpus.

        For every registered source: ingest its stored text, chunk it, and
        index the chunks. Returns a summary::

            {"documents": <n>, "chunks": <m>, "doc_ids": [<doc_id>, ...]}

        Repeatable -- a second call clears and rebuilds to the same state.
        """
        documents = load_corpus(corpus, repo_root=repo_root)
        chunks = chunk_documents(documents, chunk_size_words=chunk_size_words)
        self.clear()
        self.add_chunks(chunks)
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "doc_ids": [doc.doc_id for doc in documents],
        }

    # ------------------------------------------------------------------ #
    # Maintenance / inspection
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Drop and recreate the collection -- the store stays usable, empty."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding,
            metadata=self._COLLECTION_METADATA,
        )

    def count(self) -> int:
        return self._collection.count()

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Stored ``{"chunk_id", "text", "metadata"}`` for one id, or ``None``."""
        got = self._collection.get(ids=[chunk_id])
        if not got["ids"]:
            return None
        return {
            "chunk_id": got["ids"][0],
            "text": got["documents"][0],
            "metadata": got["metadatas"][0],
        }

    # ------------------------------------------------------------------ #
    # Low-level read -- NOT the retrieval API (that is the next task)
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_texts: Any,
        *,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Low-level ChromaDB similarity query passthrough.

        Returns Chroma's raw result dict (``ids`` / ``documents`` /
        ``metadatas`` / ``distances``). Deliberately unopinionated: no
        ranking policy, no relevance gate, no citation shaping, no
        "insufficient evidence" handling -- those belong to the retrieval
        module built on top of this.
        """
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        total = self.count()
        if total == 0:
            return {"ids": [[] for _ in query_texts], "documents": [[] for _ in query_texts],
                    "metadatas": [[] for _ in query_texts], "distances": [[] for _ in query_texts]}
        kwargs: dict[str, Any] = {
            "query_texts": list(query_texts),
            "n_results": max(1, min(n_results, total)),
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)


__all__ = [
    "DeterministicHashEmbedding",
    "ChunkVectorStore",
    "chunk_metadata",
]
