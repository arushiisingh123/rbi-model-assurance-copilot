"""RBI vector retrieval: query the chunk index for relevant source text.

Owner: Nidhi (RBI Compliance / RAG). Phase 3A, Task 5.

WHAT THIS MODULE PROVIDES
    - ``RBIRetriever`` -- runs a natural-language query against a
      ``ChunkVectorStore`` (Task 4), applies a clear relevance gate, and
      returns either the matching RBI source text with full provenance, or
      an explicit "no verified evidence" result.
    - ``build_default_retriever()`` -- one call to get a retriever over the
      approved RBI corpus (in-memory index, offline, deterministic).
    - The retriever is callable as ``retriever(query=...) -> dict`` so it
      drops straight into the ``retrieval_fn(query=...)`` slot that
      ``app/report/generate.py`` already expects. (This module does not
      modify ``generate.py``.)

DETERMINISM
    ChromaDB's approximate-nearest-neighbour (HNSW) query is built with
    per-process randomness and does not guarantee full recall, so its
    ordering -- and even which chunks it returns -- can vary between runs
    and under index load. This module therefore does not use the ANN query
    path at all: it enumerates the COMPLETE collection with an exact scan
    (``ChunkVectorStore.all_records`` -> ChromaDB ``get()`` of text +
    metadata), re-embeds each chunk's text with the store's own
    deterministic embedding (which reproduces exactly the vector stored for
    it), scores every chunk with an exact cosine distance, and breaks ties
    by ``chunk_id``. Ranking is identical for identical inputs, run to run,
    regardless of load. (The chunk vectors are re-derived rather than read
    back via ``all_records(include_embeddings=True)`` because that
    ``get(include=["embeddings"])`` path is not reliable under heavy
    concurrent ChromaDB use; for a deterministic embedding the two are
    numerically identical anyway.)

RELEVANCE GATE -- a vector match is NOT automatically evidence
    A candidate chunk is only returned as evidence if it clears BOTH:
      1. exact cosine distance <= ``max_distance`` (default 0.78), and
      2. it shares at least ``min_lexical_overlap`` (default 2) content
         words (length >= 4, tokenised on word characters) with the query.
    The second gate exists because the offline lexical embedding can score
    a stopword-only query ("the") as a close vector match; requiring real
    shared vocabulary stops that from becoming "evidence".

WHAT THIS MODULE DOES NOT DO
    - No LLM, no report generation, no compliance logic, no API/dashboard,
      no orchestration, no citation formatting beyond passing through the
      provenance already stored on each chunk.
    - It never invents, paraphrases, or infers a regulatory requirement,
      citation, URL, title, date, or clause. Every value in a result comes
      from the retrieved chunk text or the stored Chroma metadata.
    - Retrieving an excerpt does NOT make it current/binding: the stored
      ``is_excerpt`` / ``is_current`` flags pass through unchanged so the
      caller can still tell historical excerpts from verified current text.

This module does not import ``app/rag/smoke_test.py`` and does not change it.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from app.rag.corpus import APPROVED_CORPUS, RBICorpus
from app.rag.vector_store import ChunkVectorStore

# --- relevance gate defaults ------------------------------------------------
# Calibrated for the offline DeterministicHashEmbedding: for the approved
# corpus, genuinely relevant queries score <= ~0.74 and clearly unrelated
# queries score >= ~0.80. A real semantic embedding model would need this
# recalibrated -- it is a plain constructor argument.
DEFAULT_MAX_DISTANCE = 0.78
DEFAULT_MIN_LEXICAL_OVERLAP = 2
DEFAULT_CONTENT_WORD_MIN_LENGTH = 4
DEFAULT_TOP_K = 3

# --- evidence status vocabulary -------------------------------------------
EVIDENCE_RETRIEVED = "EVIDENCE_RETRIEVED"
NO_VERIFIED_EVIDENCE = "NO_VERIFIED_EVIDENCE"
EVIDENCE_STATUSES = (EVIDENCE_RETRIEVED, NO_VERIFIED_EVIDENCE)

# Attribution/provenance keys mirrored to the top level of every result, so
# the result shape is identical whether or not evidence was found.
_ATTRIBUTION_KEYS = (
    "doc_id",
    "chunk_id",
    "chunk_index",
    "source_url",
    "title",
    "publication_date",
    "document_type",
    "is_excerpt",
    "is_current",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity. Returns 1.0 (max) if either vector is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


class RBIRetriever:
    """Relevance-gated retrieval over an indexed RBI chunk corpus.

    Parameters
    ----------
    store
        A ``ChunkVectorStore`` that already has chunks indexed.
    max_distance
        Exact-cosine-distance ceiling for a candidate to count as evidence.
    min_lexical_overlap
        Minimum number of shared content words (length >=
        ``content_word_min_length``) between query and chunk.
    content_word_min_length
        Words shorter than this are ignored when measuring overlap (a
        crude, deliberate stopword filter).
    default_top_k
        Default number of results returned by ``retrieve`` / ``__call__``.
    """

    def __init__(
        self,
        store: ChunkVectorStore,
        *,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        min_lexical_overlap: int = DEFAULT_MIN_LEXICAL_OVERLAP,
        content_word_min_length: int = DEFAULT_CONTENT_WORD_MIN_LENGTH,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if not isinstance(store, ChunkVectorStore):
            raise TypeError(
                f"RBIRetriever needs a ChunkVectorStore, got {type(store).__name__}"
            )
        if default_top_k < 1:
            raise ValueError("default_top_k must be >= 1")
        self._store = store
        self.max_distance = float(max_distance)
        self.min_lexical_overlap = int(min_lexical_overlap)
        self.content_word_min_length = int(content_word_min_length)
        self.default_top_k = int(default_top_k)

    # -- public API -------------------------------------------------------

    def __call__(self, *, query: str) -> dict:
        """The ``retrieval_fn(query=...)`` entry point. Uses ``default_top_k``."""
        return self.retrieve(query)

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> dict:
        """Retrieve up to ``top_k`` relevant RBI chunks for ``query``.

        Returns a stable-shaped dict (see the module docstring / the two
        builders below). ``evidence_status`` is ``EVIDENCE_RETRIEVED`` when
        at least one chunk cleared the relevance gate, otherwise
        ``NO_VERIFIED_EVIDENCE`` -- never an arbitrary nearest chunk.
        """
        k = self.default_top_k if top_k is None else int(top_k)
        if k < 1:
            raise ValueError("top_k must be >= 1")

        query = query or ""
        if not query.strip():
            return self._no_evidence(query, "query is empty")

        if self._store.count() == 0:
            return self._no_evidence(query, "the vector index is empty")

        # Enumerate the COMPLETE indexed collection with an exact scan
        # (``all_records``, backed by ChromaDB ``get()``), then score every
        # chunk with an exact cosine distance. This deliberately does not use
        # the approximate-nearest-neighbour ``query()`` path: its recall can
        # dip below 100% under index load, which would make a borderline
        # query non-deterministic.
        records = self._store.all_records()

        # The embedding is deterministic, so re-embedding a chunk's text
        # reproduces exactly the vector stored for it. Embedding the query
        # and every chunk in one call keeps this to a single pass.
        texts = [record["text"] for record in records]
        vectors = self._store.embedding([query] + texts)
        query_vector = [float(x) for x in vectors[0]]
        chunk_vectors = [[float(x) for x in v] for v in vectors[1:]]

        query_terms = self._content_words(query)

        passed: list[tuple[float, str, str, dict]] = []
        for record, chunk_vector in zip(records, chunk_vectors):
            distance = _cosine_distance(query_vector, chunk_vector)
            if distance > self.max_distance:
                continue
            if len(query_terms & self._content_words(record["text"])) < self.min_lexical_overlap:
                continue
            passed.append(
                (distance, record["chunk_id"], record["text"], dict(record["metadata"]))
            )

        if not passed:
            return self._no_evidence(
                query, "no indexed chunk met the relevance threshold"
            )

        # Deterministic order: closest first, then chunk_id as a stable tie-break.
        passed.sort(key=lambda item: (item[0], item[1]))
        hits = [
            self._hit(text=text, meta=meta, distance=distance)
            for (distance, _chunk_id, text, meta) in passed[:k]
        ]
        return self._evidence(query, hits)

    # -- result builders -------------------------------------------------

    def _hit(self, *, text: str, meta: dict, distance: float) -> dict:
        """One retrieved chunk: exact text + attribution pulled from stored
        metadata (never reconstructed)."""
        hit = {"text": text, "distance": distance, "provenance": dict(meta)}
        for key in _ATTRIBUTION_KEYS:
            hit[key] = meta.get(key)
        return hit

    def _evidence(self, query: str, hits: list[dict]) -> dict:
        top = hits[0]
        result = {
            "query": query,
            "evidence_status": EVIDENCE_RETRIEVED,
            "reason": None,
            "retrieved_text": top["text"],
            # `source` is the required minimum key; it mirrors `title`.
            "source": top["title"],
            "distance": top["distance"],
            "provenance": top["provenance"],
            "results": hits,
        }
        for key in _ATTRIBUTION_KEYS:
            result[key] = top[key]
        return result

    def _no_evidence(self, query: str, reason: str) -> dict:
        result = {
            "query": query,
            "evidence_status": NO_VERIFIED_EVIDENCE,
            "reason": reason,
            "retrieved_text": "",
            "source": None,
            "distance": None,
            "provenance": None,
            "results": [],
        }
        for key in _ATTRIBUTION_KEYS:
            result[key] = None
        return result

    # -- helpers --------------------------------------------------------

    def _content_words(self, text: str) -> set[str]:
        """Distinct lowercase words of at least ``content_word_min_length``
        letters/digits. Deliberately crude -- no stemming, no stopword list
        beyond the length cut-off."""
        return {
            word
            for word in _WORD_RE.findall(text.lower())
            if len(word) >= self.content_word_min_length
        }


def build_default_retriever(
    corpus: RBICorpus = APPROVED_CORPUS,
    *,
    repo_root: Optional[Any] = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    min_lexical_overlap: int = DEFAULT_MIN_LEXICAL_OVERLAP,
    content_word_min_length: int = DEFAULT_CONTENT_WORD_MIN_LENGTH,
    default_top_k: int = DEFAULT_TOP_K,
) -> RBIRetriever:
    """Build an in-memory index over ``corpus`` and return a retriever for it.

    Convenience for callers that just want "retrieval over the approved RBI
    corpus" without wiring the store themselves. Offline and deterministic
    (uses ``ChunkVectorStore.in_memory()`` and the deterministic embedding).
    """
    store = ChunkVectorStore.in_memory()
    store.rebuild_from_corpus(corpus, repo_root=repo_root)
    return RBIRetriever(
        store,
        max_distance=max_distance,
        min_lexical_overlap=min_lexical_overlap,
        content_word_min_length=content_word_min_length,
        default_top_k=default_top_k,
    )


__all__ = [
    "RBIRetriever",
    "build_default_retriever",
    "EVIDENCE_RETRIEVED",
    "NO_VERIFIED_EVIDENCE",
    "EVIDENCE_STATUSES",
    "DEFAULT_MAX_DISTANCE",
    "DEFAULT_MIN_LEXICAL_OVERLAP",
    "DEFAULT_TOP_K",
]
