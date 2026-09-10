"""RBI regulatory RAG package (owner: Nidhi).

Phase 0: ``smoke_test`` -- a one-document chunk -> embed -> retrieve proof
of concept (unchanged).

Phase 3A, Task 1: ``corpus`` -- metadata for approved RBI source documents
and a simple registry. Source metadata only.

Phase 3, Task 2: ``ingestion`` -- load an approved source's stored text
from disk, paired with its provenance metadata. No chunking, embeddings,
retrieval, or interpretation yet.
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
from app.rag.ingestion import (
    IngestionError,
    LoadedDocument,
    SourceDocumentNotFoundError,
    load_by_id,
    load_corpus,
    load_source,
    provenance,
)

__all__ = [
    # corpus
    "RBISourceMetadata",
    "RBICorpus",
    "validate_source_metadata",
    "RBI_IRAC_ADVANCES_2014",
    "APPROVED_CORPUS",
    "list_sources",
    "get_source",
    # ingestion
    "LoadedDocument",
    "IngestionError",
    "SourceDocumentNotFoundError",
    "load_source",
    "load_by_id",
    "load_corpus",
    "provenance",
]
