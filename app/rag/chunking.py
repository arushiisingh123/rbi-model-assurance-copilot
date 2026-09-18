"""RBI document chunking: split an ingested document into ordered chunks.

Owner: Nidhi (RBI Compliance / RAG). Phase 3A, Task 3.

WHAT THIS MODULE DOES
    Takes a ``LoadedDocument`` (from ``app/rag/ingestion.py``) and splits
    its text into an ordered list of ``DocumentChunk`` records. Each chunk
    carries the document's full source metadata, so a chunk pulled out of a
    vector store later still knows exactly which document -- and which kind
    of document (historical excerpt vs. verified current) -- it came from.

CHUNKING STRATEGY (deliberately simple, matches the Phase 0 smoke test)
    Fixed-size windows of whitespace-separated words, no overlap. The
    document's words are split on whitespace and grouped into windows of
    ``chunk_size_words``; each window's words are rejoined with single
    spaces. No word is added, removed, or reordered -- only runs of
    whitespace *between* words are collapsed (exactly as
    ``app/rag/smoke_test.chunk_text`` already does). This does not invent,
    paraphrase, or interpret any regulatory text.

WHAT THIS MODULE DOES NOT DO
    No embeddings, vector database, retrieval, LLM, or report generation --
    those are later, separate tasks. This module does not import
    ``app/rag/smoke_test.py`` and does not change it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.rag.corpus import RBISourceMetadata
from app.rag.ingestion import LoadedDocument
from app.rag.ingestion import provenance as _source_provenance

# Words per chunk. 40 mirrors ``app/rag/smoke_test.CHUNK_SIZE_WORDS`` so
# Phase 0 and Phase 3 chunk the same way by default; kept as an independent
# constant here rather than imported, so the smoke-test PoC and the real
# pipeline stay decoupled.
DEFAULT_CHUNK_SIZE_WORDS = 40


@dataclass(frozen=True)
class DocumentChunk:
    """One ordered piece of an ingested RBI document.

    Attributes
    ----------
    text
        The chunk's words, rejoined with single spaces. Always non-empty.
    chunk_index
        0-based position of this chunk within its own document, in reading
        order. Deterministic and contiguous (0, 1, 2, ... N-1).
    source_metadata
        The exact ``RBISourceMetadata`` record for the document this chunk
        came from -- the same object, so no provenance field is lost.

    page, section_id, section_title, paragraph
        WHERE IN THE DOCUMENT this chunk came from. All Optional and all
        default to None, because they are only knowable from a document that
        has structure to extract (a paginated PDF, a numbered circular). The
        existing plain-text excerpt has none, so its chunks legitimately
        carry None.

        They are NEVER filled with a placeholder. A citation pointing at
        "page 1" because no page was known is worse than one that admits it
        has no page: it looks checkable, and checking it finds the wrong
        thing. ``is_locatable()`` is the honest test of whether this chunk
        can anchor a citation.
    """

    text: str
    chunk_index: int
    source_metadata: RBISourceMetadata
    page: int | None = None
    section_id: str | None = None
    section_title: str | None = None
    paragraph: int | None = None

    def is_locatable(self) -> bool:
        """Whether this chunk points at a place a reviewer could open.

        A compliance finding that cites a chunk should require this: without
        a page or a section, "as per the RBI circular" is unverifiable.
        """
        return self.page is not None or bool(self.section_id)

    @property
    def location(self) -> dict:
        """Page/section provenance as a dict, omitting what is unknown."""
        raw = {
            "page": self.page,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "paragraph": self.paragraph,
        }
        return {k: v for k, v in raw.items() if v is not None}

    @property
    def doc_id(self) -> str:
        return self.source_metadata.doc_id

    @property
    def chunk_id(self) -> str:
        """Corpus-unique handle for this chunk: ``"<doc_id>::chunk-<index>"``.

        Derived from ``doc_id`` + ``chunk_index`` (both stable), so two
        chunks from different documents can never collide.
        """
        return f"{self.source_metadata.doc_id}::chunk-{self.chunk_index}"

    @property
    def provenance(self) -> dict:
        """All source provenance fields as a plain dict (see
        ``app/rag/ingestion.provenance``): ``doc_id``, ``title``,
        ``issuing_authority``, ``document_type``, ``publication_date``,
        ``effective_date``, ``reference_number``, ``source_url``,
        ``retrieved_date``, ``applicable_to``, ``is_excerpt``,
        ``is_current``, ``coverage_note``, ``scope_note``. Values are
        copied verbatim from the source record -- nothing invented.

        Page/section location is merged in when known, so every consumer
        downstream of this property (Chroma metadata, retrieval hits,
        report citations) inherits chunk-level provenance without needing
        its own change. Unknown location fields are absent, not empty.
        """
        return {**_source_provenance(self.source_metadata), **self.location}


def chunk_document(
    document: LoadedDocument,
    *,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
) -> list[DocumentChunk]:
    """Split one ingested document into ordered, non-empty chunks.

    Parameters
    ----------
    document
        A ``LoadedDocument`` from ``app/rag/ingestion.py``.
    chunk_size_words
        Maximum words per chunk (must be >= 1). The final chunk may be
        shorter.

    Returns
    -------
    list[DocumentChunk]
        Chunks in reading order, ``chunk_index`` 0..N-1. Never contains an
        empty chunk.

    Raises
    ------
    TypeError
        If ``document`` is not a ``LoadedDocument``.
    ValueError
        If ``chunk_size_words`` < 1, or the document has no words to chunk.
    """
    if not isinstance(document, LoadedDocument):
        raise TypeError(
            "chunk_document() expects a LoadedDocument, got "
            f"{type(document).__name__}"
        )
    if chunk_size_words < 1:
        raise ValueError(f"chunk_size_words must be >= 1, got {chunk_size_words}")

    words = document.text.split()
    if not words:
        raise ValueError(
            f"document {document.doc_id!r} has no words to chunk"
        )

    metadata = document.metadata
    chunks: list[DocumentChunk] = []
    for chunk_index, start in enumerate(range(0, len(words), chunk_size_words)):
        window = words[start : start + chunk_size_words]
        chunks.append(
            DocumentChunk(
                text=" ".join(window),
                chunk_index=chunk_index,
                source_metadata=metadata,
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[LoadedDocument],
    *,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
) -> list[DocumentChunk]:
    """Chunk several ingested documents into one flat, ordered list.

    Each document's chunks are indexed from 0 within that document (not
    globally), and stay distinguishable by ``doc_id`` / ``chunk_id``.
    Documents are processed in the order given; within each document,
    chunks stay in reading order.
    """
    result: list[DocumentChunk] = []
    for document in documents:
        result.extend(chunk_document(document, chunk_size_words=chunk_size_words))
    return result


__all__ = [
    "DocumentChunk",
    "DEFAULT_CHUNK_SIZE_WORDS",
    "chunk_document",
    "chunk_documents",
]
