"""RBI regulatory source corpus: document metadata + a simple registry.

Owner: Nidhi (RBI Compliance / RAG). Phase 3A, Task 1.

WHAT THIS MODULE IS
    A representation of *which* RBI source documents the project has
    approved for regulatory retrieval, and *what is known about each one*
    -- bibliographic metadata (title, issuing authority, dates, reference
    number, source URL) plus provenance/scope (is this a full document or
    an excerpt? current or historical?).

WHAT THIS MODULE IS NOT
    - Not regulatory text, clauses, requirements, or interpretation. It
      records facts *about* a document, never what the document says.
    - Not ingestion, chunking, embeddings, vector search, retrieval, LLM
      integration, or report generation. Those are later Phase 3 tasks in
      separate modules.

Nothing here is invented. Every value in the seeded record below comes
verbatim from the header of the stored source file
(``data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt``)
or from ``docs/decisions.md`` ("Phase 0 RAG smoke-test source selected").
Where the source does not state something (for example a separate
effective date), the field is left ``None`` rather than guessed.

Adding another approved RBI document later means constructing another
``RBISourceMetadata`` and adding it to a corpus -- the interface does not
change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repo root, so a stored copy's ``local_path`` (recorded relative to the
# repository root) can be resolved from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Fields every source metadata record must carry as a non-empty string.
REQUIRED_FIELDS: tuple[str, ...] = (
    "doc_id",
    "title",
    "issuing_authority",
    "document_type",
    "local_path",
)


def validate_source_metadata(meta: "RBISourceMetadata") -> list[str]:
    """Return a list of problems with ``meta`` (empty list == valid).

    Advisory, mirroring ``app/rbi/schema.validate_rule``: it reports rather
    than raising, so tests and a future file-based loader can call it
    directly. ``RBISourceMetadata.__post_init__`` turns a non-empty result
    into a ``ValueError`` so a malformed record can never enter a corpus
    unnoticed.
    """
    problems: list[str] = []

    for name in REQUIRED_FIELDS:
        value = getattr(meta, name, None)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{name} is required and must be a non-empty string")

    for name in ("is_excerpt", "is_current"):
        if not isinstance(getattr(meta, name), bool):
            problems.append(f"{name} must be a bool (True or False)")

    # An excerpt must say what part of the document it covers, so a caller
    # can never mistake a fragment for the whole instrument.
    if getattr(meta, "is_excerpt", None) is True:
        coverage = meta.coverage_note
        if not isinstance(coverage, str) or not coverage.strip():
            problems.append(
                "is_excerpt=True requires a non-empty coverage_note describing "
                "what part of the document the local copy contains"
            )

    return problems


@dataclass(frozen=True)
class RBISourceMetadata:
    """Bibliographic and provenance metadata for one approved RBI source.

    Required
    --------
    doc_id
        Stable registry identifier -- **not** a regulatory citation. Like
        ``rule_id`` in ``app/rbi/rules``: a key this project chooses so the
        document can be looked up. Example:
        ``"rbi-master-circular-irac-advances-2014-07-01"``.
    title
        Document title, exactly as printed in the source.
    issuing_authority
        For example ``"Reserve Bank of India (RBI)"``.
    document_type
        For example ``"Master Circular"``, ``"Master Direction"``,
        ``"Notification"``, ``"Circular"``.
    local_path
        Path to the locally stored text, relative to the repository root.

    Optional (kept ``None`` when the source does not state them -- never
    guessed)
    -------------------------------------------------------------------
    publication_date
        ISO date (``YYYY-MM-DD``) the document was issued / published.
    effective_date
        ISO date the document takes effect, only if the source states one
        *separately* from the publication date.
    reference_number
        Circular / notification reference string, for example
        ``"RBI/2014-15/74; DBOD.No.BP.BC.9/21.04.048/2014-15"``.
    source_url
        Official URL the local copy was obtained from.
    retrieved_date
        ISO date the local copy was downloaded.
    applicable_to
        Stated applicability, for example
        ``"All Commercial Banks (excluding Regional Rural Banks)"``.

    Provenance / scope -- keeps "what the file is" separate from "what it
    says"
    -----------------------------------------------------------------------
    is_excerpt
        ``True`` if ``local_path`` holds only a partial excerpt of the
        document rather than its full text.
    is_current
        ``True`` only if this has been confirmed by the team to be
        current, binding regulatory text. Defaults to ``False``: a stored
        copy is treated as historical unless explicitly verified
        otherwise.
    coverage_note
        For an excerpt, which portion of the document the local copy
        contains. Required when ``is_excerpt`` is ``True``.
    scope_note
        Free-text caveat: why this copy must (or must not) be relied on,
        what later instruments supersede it, and so on.
    """

    doc_id: str
    title: str
    issuing_authority: str
    document_type: str
    local_path: str

    publication_date: str | None = None
    effective_date: str | None = None
    reference_number: str | None = None
    source_url: str | None = None
    retrieved_date: str | None = None
    applicable_to: str | None = None

    is_excerpt: bool = False
    is_current: bool = False
    coverage_note: str | None = None
    scope_note: str | None = None

    def __post_init__(self) -> None:
        problems = validate_source_metadata(self)
        if problems:
            raise ValueError(
                "Invalid RBISourceMetadata"
                + (f" ({self.doc_id!r})" if isinstance(self.doc_id, str) else "")
                + ":\n  - "
                + "\n  - ".join(problems)
            )

    @property
    def is_verified_current_regulation(self) -> bool:
        """``True`` only for a full document confirmed as current/binding.

        Any excerpt, or anything not explicitly confirmed current, is not
        verified current regulation. Callers check this before presenting
        a source's text as a binding requirement.
        """
        return self.is_current and not self.is_excerpt

    def resolved_path(self, repo_root: Path | None = None) -> Path:
        """Absolute path to the locally stored copy.

        ``local_path`` is stored relative to the repository root; this
        resolves it so ingestion code (a later task) does not have to
        care about the working directory.
        """
        root = repo_root if repo_root is not None else _REPO_ROOT
        return (root / self.local_path).resolve()


class RBICorpus:
    """An ordered registry of approved RBI source documents.

    Holds ``RBISourceMetadata`` records, rejects duplicates by ``doc_id``,
    and offers simple lookup. Multiple records coexist; adding another
    approved document is just ``corpus.add(RBISourceMetadata(...))``.
    """

    def __init__(self, sources: list[RBISourceMetadata] | None = None) -> None:
        self._sources: dict[str, RBISourceMetadata] = {}
        for source in sources or []:
            self.add(source)

    def add(self, source: RBISourceMetadata) -> None:
        if not isinstance(source, RBISourceMetadata):
            raise TypeError(
                f"RBICorpus holds RBISourceMetadata records, got "
                f"{type(source).__name__}"
            )
        if source.doc_id in self._sources:
            raise ValueError(f"duplicate doc_id in corpus: {source.doc_id!r}")
        self._sources[source.doc_id] = source

    def get(self, doc_id: str) -> RBISourceMetadata | None:
        """One approved source by ``doc_id``, or ``None`` if not registered."""
        return self._sources.get(doc_id)

    def all(self) -> list[RBISourceMetadata]:
        """Every registered source, in insertion order."""
        return list(self._sources.values())

    def doc_ids(self) -> list[str]:
        """The registered ``doc_id`` values, in insertion order."""
        return list(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._sources

    def __iter__(self):
        return iter(self._sources.values())


# ---------------------------------------------------------------------------
# The 2014 regression fixture
#
# APPROVED_CORPUS holds exactly one source: the limited historical excerpt
# already in the repo. It is NOT the production corpus and has not been since
# the RBI Directions were obtained -- ``corpus_from_manifest()`` below reads
# the manifest, and ``app.rag.retrieval.default_corpus()`` uses that for live
# retrieval. This record is retained as the regression fixture that proves the
# non-PDF ingestion path still works.
#
# Every value below is taken from that file's own header -- see the file for
# the full scope/limitation statement.
# ---------------------------------------------------------------------------

RBI_IRAC_ADVANCES_2014 = RBISourceMetadata(
    doc_id="rbi-master-circular-irac-advances-2014-07-01",
    title=(
        "Master Circular - Prudential Norms on Income Recognition, Asset "
        "Classification and Provisioning pertaining to Advances"
    ),
    issuing_authority="Reserve Bank of India (RBI)",
    document_type="Master Circular",
    local_path="data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt",
    publication_date="2014-07-01",
    # The source states an issue date only; no separate effective date is
    # printed, so this stays None rather than being guessed.
    effective_date=None,
    reference_number="RBI/2014-15/74; DBOD.No.BP.BC.9/21.04.048/2014-15",
    source_url=(
        "https://www.rbi.org.in/commonman/Upload/English/Notification/PDFs/"
        "74MIR010714FL.pdf"
    ),
    retrieved_date="2026-08-25",
    applicable_to="All Commercial Banks (excluding Regional Rural Banks)",
    is_excerpt=True,
    is_current=False,
    coverage_note=(
        "Limited verbatim excerpt only: Part A, Section 1 'GENERAL', "
        "Section 2 'DEFINITIONS' (2.1-2.3), and the opening of Section 3 "
        "'INCOME RECOGNITION' (3.1). Not the full 120-page circular."
    ),
    scope_note=(
        "Historical excerpt retained for Phase 0 RAG smoke-test provenance. "
        "Dated July 1, 2014; the Reserve Bank of India has since issued "
        "later updates/consolidations on this subject (further Master "
        "Circulars, and subsequently Master Directions). This copy must not "
        "be treated as current or binding regulatory text. No RBI clauses "
        "or requirements have been invented, paraphrased, summarised, or "
        "interpreted here -- this record holds source metadata only. See "
        "the source file header and docs/decisions.md ('Phase 0 RAG "
        "smoke-test source selected')."
    ),
)

APPROVED_CORPUS = RBICorpus([RBI_IRAC_ADVANCES_2014])


def list_sources() -> list[RBISourceMetadata]:
    """Every approved RBI source document, as metadata records."""
    return APPROVED_CORPUS.all()


def get_source(doc_id: str) -> RBISourceMetadata | None:
    """One approved RBI source by registry ``doc_id``, or ``None``."""
    return APPROVED_CORPUS.get(doc_id)


# ---------------------------------------------------------------------------
# Manifest-backed corpus
#
# ``app/rbi/corpus_manifest.json`` is the canonical record of which documents
# this project has, and it is DATA -- adding a source is a manifest edit. The
# functions below read it, so no second hardcoded registry has to be kept in
# step with it.
#
# ``APPROVED_CORPUS`` above stays exactly as it was. It holds the 2014 text
# excerpt, which is not a manifest document: it is the regression fixture that
# proves the non-PDF ingestion path still works. Keeping it separate means a
# manifest edit can never silently break that proof.
# ---------------------------------------------------------------------------

# Manifest keys a document must carry to become an RBISourceMetadata. They are
# absent from entries for documents nobody has obtained, which is correct --
# such a document has no local file to describe.
MANIFEST_METADATA_KEYS: tuple[str, ...] = (
    "issuing_authority",
    "document_type",
)


class ManifestSourceError(Exception):
    """A downloaded manifest entry cannot be turned into source metadata."""


def source_from_manifest_document(document) -> RBISourceMetadata:
    """Build an ``RBISourceMetadata`` from one manifest document.

    Only meaningful for an entry that has actually been downloaded: the
    record describes a local file, and there is no local file otherwise.

    Every value is copied from the manifest. Nothing is derived, defaulted or
    inferred -- in particular ``is_current`` is read from the entry rather
    than computed from ``regulatory_status``, because "not superseded as far
    as anyone has checked" and "verified to be in force" are different claims
    and only the manifest author can make the second one.

    Raises
    ------
    ManifestSourceError
        If the entry is not downloaded, has no ``local_path``, or is missing
        a key ``RBISourceMetadata`` requires.
    """
    if not document.is_citable():
        raise ManifestSourceError(
            f"{document.document_id!r} has source_status "
            f"{document.source_status!r}; only a downloaded document has a "
            "local file to describe"
        )
    if not document.local_path:
        raise ManifestSourceError(
            f"{document.document_id!r} is marked downloaded but has no "
            "local_path, so nothing can be loaded for it"
        )

    raw = document.raw or {}
    missing = [key for key in MANIFEST_METADATA_KEYS if not raw.get(key)]
    if missing:
        raise ManifestSourceError(
            f"{document.document_id!r} is downloaded but its manifest entry "
            f"is missing {', '.join(missing)}, which RBISourceMetadata "
            "requires and which must be read from the document, not guessed"
        )

    applies_to = ", ".join(document.applies_to) or None

    return RBISourceMetadata(
        doc_id=document.document_id,
        title=document.title,
        issuing_authority=raw["issuing_authority"],
        document_type=raw["document_type"],
        local_path=document.local_path,
        publication_date=document.issued_date,
        effective_date=document.effective_date,
        reference_number=document.circular_number,
        source_url=document.source_url,
        retrieved_date=raw.get("retrieved_date"),
        applicable_to=applies_to,
        is_excerpt=bool(raw.get("is_excerpt", False)),
        is_current=bool(raw.get("is_current", False)),
        coverage_note=raw.get("coverage_note"),
        scope_note=raw.get("scope_note"),
    )


def corpus_from_manifest(manifest=None) -> RBICorpus:
    """Every DOWNLOADED manifest document, as a corpus ready for ingestion.

    Documents that have not been obtained are skipped, not stubbed: a corpus
    entry for a document nobody has is a citation waiting to happen.

    The 2014 text excerpt is deliberately NOT included -- it is not a manifest
    document. It remains available as ``APPROVED_CORPUS`` for the regression
    that proves text ingestion still works.
    """
    if manifest is None:
        from app.rbi.manifest import load_manifest

        manifest = load_manifest()

    return RBICorpus(
        [source_from_manifest_document(document) for document in manifest.citable()]
    )


__all__ = [
    "RBISourceMetadata",
    "RBICorpus",
    "validate_source_metadata",
    "REQUIRED_FIELDS",
    "RBI_IRAC_ADVANCES_2014",
    "APPROVED_CORPUS",
    "MANIFEST_METADATA_KEYS",
    "ManifestSourceError",
    "list_sources",
    "get_source",
    "source_from_manifest_document",
    "corpus_from_manifest",
]
