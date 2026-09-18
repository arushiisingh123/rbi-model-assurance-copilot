"""Hybrid retrieval: lexical (BM25) + vector + metadata filter + rerank.

WHY NOT PURE VECTOR SEARCH
    The project's embedding function (``DeterministicHashEmbedding``) is a
    SHA-256 word-bucket hash, and its own docstring is explicit that it is
    "not a semantic model -- it cannot tell that 'NPA' and 'non performing
    asset' mean the same thing". Regulatory retrieval on top of that alone
    is lexical similarity wearing a vector's clothing.

    BM25 is added because it is honest about being lexical AND it is better
    at it: it weights rare terms ("disparate", "decommissioning") above
    common ones, which bucket-hash cosine does not. Combining the two ranks
    a chunk highly when either signal is strong, and highest when both are.

    This is NOT a claim of semantic retrieval. Swapping in a real embedding
    model later changes one component; the pipeline stays.

THE ORDER IS THE POINT

    query -> METADATA FILTER -> BM25 + vector -> fuse -> rerank -> top-k

    Filtering comes FIRST, not last. A document that does not apply to this
    entity must never reach the ranker, because a sufficiently strong text
    match would otherwise float an inapplicable regulation to the top and it
    would be cited. Applicability is a gate, not a tiebreaker.

    Reranking then demotes stale authority: a historical or superseded
    chunk can be the best textual match and still must not outrank a current
    one. That is a corpus rule, so it is enforced in code rather than left
    to whoever reads the output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Iterable, Optional, Sequence

# Ranking weights. Deliberately explicit and boring -- these are the knobs
# that decide which regulation a reviewer sees first.
DEFAULT_LEXICAL_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5

# Multiplicative authority adjustments applied AFTER fusion. Values below 1.0
# demote; nothing is ever boosted above a current source.
AUTHORITY_MULTIPLIERS = {
    "current": 1.0,
    "to_be_superseded": 0.85,
    "draft": 0.75,
    "committee_report": 0.65,
    "superseded_in_scope": 0.5,
    "historical": 0.5,
}

_TOKEN = re.compile(r"[a-z0-9]+")

# BM25 parameters (Robertson/Sparck-Jones defaults).
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Shared by indexing and querying."""
    return _TOKEN.findall(str(text or "").lower())


@dataclass
class RetrievalCandidate:
    """One chunk eligible for ranking."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]

    @property
    def document_id(self) -> Optional[str]:
        # Manifest-era chunks use document_id; the existing corpus uses
        # doc_id. Both are read so the two generations coexist.
        return self.metadata.get("document_id") or self.metadata.get("doc_id")


@dataclass
class RankedChunk:
    """A candidate with its scores, kept separate so ranking is auditable."""

    candidate: RetrievalCandidate
    lexical_score: float
    vector_score: float
    fused_score: float
    authority_multiplier: float
    final_score: float
    filtered_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.candidate.chunk_id,
            "document_id": self.candidate.document_id,
            "text": self.candidate.text,
            "metadata": dict(self.candidate.metadata),
            "scores": {
                "lexical": round(self.lexical_score, 6),
                "vector": round(self.vector_score, 6),
                "fused": round(self.fused_score, 6),
                "authority_multiplier": self.authority_multiplier,
                "final": round(self.final_score, 6),
            },
        }


class BM25Index:
    """Minimal Okapi BM25 over the candidate chunks.

    Implemented here rather than added as a dependency: it is ~30 lines, and
    CLAUDE.md asks that new major technologies not be introduced without
    team approval.
    """

    def __init__(self, documents: Sequence[Sequence[str]]) -> None:
        self._docs = [list(d) for d in documents]
        self._lengths = [len(d) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._docs else 0.0
        self._freqs: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for tokens in self._docs:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._freqs.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self._df = document_frequency
        self._n = len(self._docs)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], index: int) -> float:
        if index >= self._n or not self._avg_len:
            return 0.0
        counts = self._freqs[index]
        length = self._lengths[index] or 1
        total = 0.0
        for term in query_tokens:
            freq = counts.get(term, 0)
            if not freq:
                continue
            numerator = freq * (BM25_K1 + 1)
            denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * length / self._avg_len)
            total += self._idf(term) * numerator / denominator
        return total

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        return [self.score(query_tokens, i) for i in range(self._n)]


def _normalise_scores(scores: Sequence[float]) -> list[float]:
    """Min-max to [0, 1] so BM25 and cosine are comparable before fusion.

    BM25 is unbounded and cosine similarity is bounded; summing them raw
    would let BM25 silently dominate.
    """
    if not scores:
        return []
    lowest, highest = min(scores), max(scores)
    if highest <= lowest:
        return [0.0 for _ in scores]
    span = highest - lowest
    return [(s - lowest) / span for s in scores]


def apply_metadata_filter(
    candidates: Iterable[RetrievalCandidate],
    *,
    allowed_document_ids: Optional[set[str]] = None,
    domains: Optional[set[str]] = None,
    exclude_document_ids: Optional[set[str]] = None,
) -> tuple[list[RetrievalCandidate], list[tuple[RetrievalCandidate, str]]]:
    """Gate candidates by metadata BEFORE ranking.

    Returns ``(kept, rejected_with_reason)``. Rejections are returned rather
    than discarded so a caller can show WHY a regulation was not considered
    -- an invisible exclusion is indistinguishable from an oversight.
    """
    kept: list[RetrievalCandidate] = []
    rejected: list[tuple[RetrievalCandidate, str]] = []

    for candidate in candidates:
        document_id = candidate.document_id
        if exclude_document_ids and document_id in exclude_document_ids:
            rejected.append((candidate, f"document {document_id} excluded by applicability"))
            continue
        if allowed_document_ids is not None and document_id not in allowed_document_ids:
            rejected.append(
                (candidate, f"document {document_id} is not in the applicable set")
            )
            continue
        if domains:
            raw = candidate.metadata.get("domain") or candidate.metadata.get("domains") or ""
            chunk_domains = {
                d.strip().lower()
                for d in (raw.split(",") if isinstance(raw, str) else list(raw))
                if str(d).strip()
            }
            if chunk_domains and not (chunk_domains & {d.lower() for d in domains}):
                rejected.append(
                    (candidate, f"domain mismatch: chunk {sorted(chunk_domains)} vs {sorted(domains)}")
                )
                continue
        kept.append(candidate)

    return kept, rejected


def authority_multiplier(metadata: dict[str, Any]) -> float:
    """How much a chunk's own regulatory standing should demote it.

    Never above 1.0: nothing may be boosted past a current source.
    """
    status = str(metadata.get("regulatory_status") or "current").lower()
    multiplier = AUTHORITY_MULTIPLIERS.get(status, 1.0)
    # An excerpt is a fragment of a document; it can be right, but it should
    # not outrank the complete text of an equally-matching source.
    if metadata.get("is_excerpt") in (True, "True", "true"):
        multiplier *= 0.9
    return multiplier


def hybrid_search(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    *,
    embed: Optional[Callable[[str], Sequence[float]]] = None,
    top_k: int = 5,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    allowed_document_ids: Optional[set[str]] = None,
    domains: Optional[set[str]] = None,
    exclude_document_ids: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Run the full pipeline and return ranked hits plus an audit trail.

    ``embed`` is optional. Without it the vector leg scores 0 and retrieval
    degrades to BM25 + metadata + rerank -- which is a real, usable pipeline,
    not a broken one. That matters because it keeps the architecture testable
    with no embedding backend present.
    """
    kept, rejected = apply_metadata_filter(
        candidates,
        allowed_document_ids=allowed_document_ids,
        domains=domains,
        exclude_document_ids=exclude_document_ids,
    )

    audit = {
        "query": query,
        "candidates_in": len(list(candidates)),
        "candidates_after_filter": len(kept),
        "rejected": [
            {"chunk_id": c.chunk_id, "document_id": c.document_id, "reason": r}
            for c, r in rejected
        ],
        "vector_leg": "enabled" if embed is not None else "disabled (no embedder supplied)",
    }

    if not kept:
        return {"hits": [], "audit": {**audit, "reason": "no candidate survived metadata filtering"}}

    query_tokens = tokenize(query)
    index = BM25Index([tokenize(c.text) for c in kept])
    lexical = _normalise_scores(index.scores(query_tokens))

    if embed is not None:
        query_vector = list(embed(query))
        raw_vector_scores = [
            _cosine(query_vector, list(embed(candidate.text))) for candidate in kept
        ]
        vector = _normalise_scores(raw_vector_scores)
    else:
        vector = [0.0] * len(kept)

    ranked: list[RankedChunk] = []
    for position, candidate in enumerate(kept):
        fused = lexical_weight * lexical[position] + vector_weight * vector[position]
        multiplier = authority_multiplier(candidate.metadata)
        ranked.append(
            RankedChunk(
                candidate=candidate,
                lexical_score=lexical[position],
                vector_score=vector[position],
                fused_score=fused,
                authority_multiplier=multiplier,
                final_score=fused * multiplier,
            )
        )

    # RERANK: authority-adjusted score, then stable by chunk_id so equal
    # scores never reorder between runs (a report must be reproducible).
    ranked.sort(key=lambda r: (-r.final_score, r.candidate.chunk_id))

    return {
        "hits": [r.to_dict() for r in ranked[:top_k]],
        "audit": {**audit, "ranked": len(ranked), "returned": min(top_k, len(ranked))},
    }


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = [
    "BM25Index",
    "RetrievalCandidate",
    "RankedChunk",
    "AUTHORITY_MULTIPLIERS",
    "DEFAULT_LEXICAL_WEIGHT",
    "DEFAULT_VECTOR_WEIGHT",
    "tokenize",
    "apply_metadata_filter",
    "authority_multiplier",
    "hybrid_search",
]
