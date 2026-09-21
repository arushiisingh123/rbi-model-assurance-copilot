"""RBI document ingestion: load an approved source's stored text + metadata.

Owner: Nidhi (RBI Compliance / RAG). Phase 3, Task 2.

WHAT THIS MODULE DOES
    Given an approved source (from ``app/rag/corpus.py``), resolve its local
    file, read the stored text exactly as it is on disk, and return that
    text paired with the source's provenance metadata.

TWO FILE FORMATS, ONE RETURN TYPE
    ``.txt`` sources are read verbatim, exactly as they always have been.
    ``.pdf`` sources go through ``app/rag/pdf_extract.py``, which returns
    the same kind of thing plus the page and clause structure a citation
    needs. Either way the caller gets a ``LoadedDocument``; the PDF case
    additionally populates ``LoadedDocument.structure``.

    The text path is deliberately untouched by the PDF work. The stored 2014
    excerpt must keep loading byte-for-byte identically -- see
    ``tests/rag/test_ingestion.py`` for the regression that holds this.

WHAT THIS MODULE DOES NOT DO
    - No chunking, embeddings, vector store, retrieval, LLM, or report
      generation -- those are later, separate tasks. In particular this
      module does not decide which page or clause a chunk belongs to; it
      only makes that answerable, via ``structure``.
    - No cleaning, normalising, summarising, trimming, or interpreting the
      text. For a ``.txt`` source ``LoadedDocument.text`` is a verbatim copy
      of the file. Deciding which spans of a file to index (for example
      excluding the provenance header) belongs to the chunking task.
    - No metadata is invented. Every provenance value comes straight from
      the ``RBISourceMetadata`` record; a value the source never stated
      stays ``None``.

This module does not import ``app/rag/smoke_test.py`` (the Phase 0 proof of
concept) and does not change it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.rag.corpus import APPROVED_CORPUS, RBICorpus, RBISourceMetadata
from app.rag.pdf_extract import ExtractedDocument, PDFExtractionError, extract_pdf

# Suffixes routed through the PDF extractor. Everything else is read as UTF-8
# text, which is what every pre-PDF source was.
PDF_SUFFIXES = (".pdf",)

# The provenance fields ingestion carries through, in a stable order. Kept
# here as an explicit statement of what ingestion promises to preserve;
# ``tests/rag/test_ingestion.py`` asserts this list stays in sync with the
# ``RBISourceMetadata`` dataclass fields.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "doc_id",
    "title",
    "issuing_authority",
    "document_type",
    "publication_date",
    "effective_date",
    "reference_number",
    "source_url",
    "retrieved_date",
    "applicable_to",
    "is_excerpt",
    "is_current",
    "coverage_note",
    "scope_note",
)


class IngestionError(Exception):
    """Base class for RBI document ingestion failures."""


class SourceDocumentNotFoundError(IngestionError, FileNotFoundError):
    """An approved source's ``local_path`` does not resolve to an existing file.

    Subclasses both ``IngestionError`` and ``FileNotFoundError`` so callers
    can catch it either way.
    """


def provenance(source: RBISourceMetadata) -> dict:
    """Return every provenance field of ``source`` as a plain dict.

    Stable key order (``PROVENANCE_FIELDS``). Values are copied verbatim
    from the metadata record -- nothing is added, removed, or defaulted.
    Later steps (chunking, retrieval, citation) attach this to each unit
    they produce so a retrieved fragment always carries where it came from.
    """
    return {name: getattr(source, name) for name in PROVENANCE_FIELDS}


@dataclass(frozen=True)
class LoadedDocument:
    """One approved RBI source document, loaded from disk.

    Attributes
    ----------
    metadata
        The exact ``RBISourceMetadata`` record this text was loaded for --
        the same object, so provenance is preserved rather than copied or
        rebuilt.
    text
        The document's content. For a text source, the file read verbatim as
        UTF-8. For a PDF, the extracted page text joined in page order, with
        line breaks preserved. Not chunked, cleaned, or interpreted.
    path
        The resolved absolute path the text was read from.
    structure
        For a PDF, the ``ExtractedDocument`` carrying page spans and clause
        marks, so a later stage can ask which page and clause any offset in
        ``text`` belongs to. ``None`` for a text source, which genuinely has
        no page or clause structure -- not an omission, an absence.
    """

    metadata: RBISourceMetadata
    text: str
    path: Path
    structure: Optional[ExtractedDocument] = None

    @property
    def doc_id(self) -> str:
        return self.metadata.doc_id

    @property
    def is_paginated(self) -> bool:
        """Whether this document can anchor a page-level citation."""
        return self.structure is not None


def load_source(
    source: RBISourceMetadata,
    *,
    repo_root: Path | None = None,
) -> LoadedDocument:
    """Load one approved source's stored text.

    Parameters
    ----------
    source
        An ``RBISourceMetadata`` record, typically from ``APPROVED_CORPUS``.
    repo_root
        Optional root to resolve ``source.local_path`` against. Defaults to
        the repository root (handled by
        ``RBISourceMetadata.resolved_path``).

    Raises
    ------
    TypeError
        If ``source`` is not an ``RBISourceMetadata``.
    SourceDocumentNotFoundError
        If the resolved path does not exist or is not a file.
    IngestionError
        If the file exists but cannot be read as UTF-8 text, or is empty
        (only whitespace).
    """
    if not isinstance(source, RBISourceMetadata):
        raise TypeError(
            "load_source() expects an RBISourceMetadata, got "
            f"{type(source).__name__}"
        )

    path = source.resolved_path(repo_root)

    if not path.is_file():
        raise SourceDocumentNotFoundError(
            f"source document for {source.doc_id!r} not found: expected a file "
            f"at {path} (from local_path {source.local_path!r})"
        )

    structure: Optional[ExtractedDocument] = None

    if path.suffix.lower() in PDF_SUFFIXES:
        try:
            structure = extract_pdf(path)
        except PDFExtractionError as exc:
            raise IngestionError(
                f"could not read source document for {source.doc_id!r} at "
                f"{path}: {exc}"
            ) from exc
        text = structure.text
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestionError(
                f"could not read source document for {source.doc_id!r} at {path}: "
                f"{exc}"
            ) from exc

    if not text.strip():
        raise IngestionError(
            f"source document for {source.doc_id!r} at {path} is empty"
        )

    return LoadedDocument(
        metadata=source, text=text, path=path, structure=structure
    )


def load_by_id(
    doc_id: str,
    *,
    corpus: RBICorpus = APPROVED_CORPUS,
    repo_root: Path | None = None,
) -> LoadedDocument:
    """Load one approved source by its registry ``doc_id``.

    Raises ``IngestionError`` if no source with that id is registered in
    ``corpus``; otherwise behaves exactly like ``load_source``.
    """
    source = corpus.get(doc_id)
    if source is None:
        raise IngestionError(
            f"no approved RBI source registered with doc_id {doc_id!r} "
            f"(registered: {corpus.doc_ids()})"
        )
    return load_source(source, repo_root=repo_root)


def load_corpus(
    corpus: RBICorpus = APPROVED_CORPUS,
    *,
    repo_root: Path | None = None,
) -> list[LoadedDocument]:
    """Load the stored text for every source registered in ``corpus``.

    Order matches ``corpus.all()`` (insertion order). If any single source
    is unreadable, that error propagates -- ingestion does not silently
    skip a source it was asked to load.
    """
    return [load_source(source, repo_root=repo_root) for source in corpus.all()]


__all__ = [
    "LoadedDocument",
    "IngestionError",
    "SourceDocumentNotFoundError",
    "PROVENANCE_FIELDS",
    "PDF_SUFFIXES",
    "provenance",
    "load_source",
    "load_by_id",
    "load_corpus",
]
