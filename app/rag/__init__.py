"""RBI regulatory RAG package (owner: Nidhi).

Phase 0: ``smoke_test`` -- a one-document chunk -> embed -> retrieve proof
of concept (unchanged).

Phase 3A, Task 1: ``corpus`` -- metadata for approved RBI source documents
and a simple registry. Source metadata only; no ingestion, embeddings,
retrieval, or regulatory interpretation yet.
"""
from app.rag.corpus import (
    APPROVED_CORPUS,
    RBI_IRAC_ADVANCES_2014,
    RBICorpus,
    RBISourceMetadata,
    get_source,
    list_sources,
    validate_source_metadata,
)

__all__ = [
    "RBISourceMetadata",
    "RBICorpus",
    "validate_source_metadata",
    "RBI_IRAC_ADVANCES_2014",
    "APPROVED_CORPUS",
    "list_sources",
    "get_source",
]
