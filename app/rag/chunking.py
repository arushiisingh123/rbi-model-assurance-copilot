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

PAGINATED SOURCES ARE CHUNKED PAGE BY PAGE (Phase B)
    When a ``LoadedDocument`` carries extractor structure (a PDF), chunking
    restarts at every page boundary, so no chunk spans two pages. A chunk
    that straddled pages 15 and 16 could honestly claim neither, and would
    have to claim one -- which is precisely the kind of citation that looks
    checkable and resolves to the wrong place.

    Each chunk then takes the page it sits on, and the clause that was in
    force where it starts, from ``app/rag/pdf_extract.py``. Those values are
    READ, never derived here: this module does not parse clause numbers and
    does not know what a clause looks like.

    Front matter -- covering letters and contents pages, everything before
    the document's first substantive clause -- carries ``section_id = None``
    rather than inheriting a neighbouring clause's number.

    A text source has no pages and no clauses, so its chunks keep the
    ``None`` they have always had. That path is unchanged.

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

    if document.structure is not None:
        return _chunk_paginated(document, chunk_size_words=chunk_size_words)

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


def _chunk_paginated(
    document: LoadedDocument,
    *,
    chunk_size_words: int,
) -> list[DocumentChunk]:
    """Chunk a PDF page by page, attaching page and clause provenance.

    Chunks never span a page boundary, so ``page`` is always exactly true
    rather than "mostly". Within a page, words are windowed exactly as the
    text path windows them, so the two produce the same kind of chunk.

    ``section_id`` is whatever clause ``pdf_extract`` reports as in force at
    the chunk's starting offset, and ``None`` before the first clause. This
    module never parses, completes or normalises a clause identifier.

    KNOWN LIMITATION: section_id describes where a chunk STARTS
        A window that begins in the tail of clause 16 and runs on into
        clause 17 is labelled 16. The page is always exact; the clause is the
        one the chunk opens in. 40% of chunks span a boundary this way.

        The label is imprecise but never WRONG in the dangerous direction: a
        chunk can only ever be labelled with a clause at or before it, never
        one that begins later. tests/rag/test_chunk_page_provenance.py pins
        that invariant, which is what stops a citation pointing a reviewer
        past the provision they were shown.

        FOUR ALTERNATIVES WERE IMPLEMENTED AND MEASURED. ALL WERE REJECTED.
        Clause-aware segmentation, with short segments merged forward into
        their neighbour at several thresholds:

            strategy        chunks  <8 words  crossing  coverage  page@5
            ------------------------------------------------------------
            THIS MODULE      1264        36     40.0%       5/5   15/16
            pure split       1724       215      0.0%       3/5   15/16
            merge <15w       1511        89     22.8%       2/5   14/16
            merge <25w       1459       104     31.5%       4/5   14/16
            merge <35w       1442       118     44.7%         -       -

        Not one improves on the current behaviour. Higher merge thresholds
        are worse still: they re-create the spanning they were meant to
        remove.

        WHY -- AND A CORRECTION TO THE EARLIER EXPLANATION
        An earlier version of this note blamed retrieval scoring: it claimed
        short chunks outranked longer passages, and that length-normalising
        the ranking would make clause splitting viable. That was a
        hypothesis, and measuring it showed it to be WRONG.

        Nine ranking strategies were benchmarked against this corpus --
        plain cosine, pivoted length normalisation (Singhal et al.) at three
        slopes, a bounded short-chunk penalty at three settings, and two
        query-coverage blends -- each under this chunking and under clause
        splitting. Two findings killed the hypothesis:

          * Under THIS chunking, not one selected chunk is under eight words.
            There is no short-chunk problem for a scorer to fix, which is why
            the short-chunk penalty changed nothing at all.
          * Under clause splitting, the sections that lose coverage retrieve
            chunks of 40 and 30 words -- not short ones -- while the single
            five-word chunk that does win actually PASSES the relevance gate.

        The real mechanism is simpler. Splitting redistributes text across
        boundaries, so a different passage ranks first. The new winner is
        usually a topically adjacent clause that does not happen to contain
        the multi-word phrase app/report/generate.py's relevance gate looks
        for -- "regular monitoring and assessment" where the gate wants
        "periodic assessment". Retrieval is not failing: doc@5 stays 16/16
        and page@5 stays 15/16 throughout. The section gate is what fails.

        So this is not a scoring problem and not a chunk-length problem. It
        is the interaction between chunk size and a lexical relevance gate,
        and no ranking change can fix it. The simpler windowing stands,
        because a slightly imprecise clause label is a far better failure
        than losing two report sections.
    """
    structure = document.structure
    metadata = document.metadata
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page_number, page_start, page_end in structure.page_spans:
        # A contents page is navigation, not regulatory text. It lists every
        # heading in the document, which makes it match almost any topical
        # query -- and a citation pointing at a table of contents tells a
        # reviewer nothing. Indexing it actively crowds out the real clause.
        if structure.is_contents_page(page_number):
            continue

        page_text = structure.text[page_start:page_end]

        # Walk the page's words while tracking each one's offset in the full
        # document text, so a chunk's clause is looked up at the position the
        # chunk actually begins -- not at the start of the page, which would
        # attribute a whole page to its first clause.
        offsets: list[int] = []
        words: list[str] = []
        cursor = 0
        for token in page_text.split():
            found = page_text.find(token, cursor)
            if found < 0:  # pragma: no cover - split() guarantees presence
                found = cursor
            offsets.append(page_start + found)
            words.append(token)
            cursor = found + len(token)

        if not words:
            # A genuinely blank page (separator sheets do occur). It
            # contributes no chunk rather than an empty one.
            continue

        for start in range(0, len(words), chunk_size_words):
            window = words[start : start + chunk_size_words]
            mark = structure.section_for_offset(offsets[start])
            chunks.append(
                DocumentChunk(
                    text=" ".join(window),
                    chunk_index=chunk_index,
                    source_metadata=metadata,
                    page=page_number,
                    section_id=mark.section_id if mark else None,
                    section_title=mark.section_title if mark else None,
                )
            )
            chunk_index += 1

    if not chunks:
        raise ValueError(f"document {document.doc_id!r} has no words to chunk")

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
