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
# Approved RBI source documents
#
# Phase 3A currently has exactly ONE approved source: the limited historical
# excerpt already in the repo. Every value below is taken from that file's
# own header -- see the file for the full scope/limitation statement.
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


__all__ = [
    "RBISourceMetadata",
    "RBICorpus",
    "validate_source_metadata",
    "REQUIRED_FIELDS",
    "RBI_IRAC_ADVANCES_2014",
    "APPROVED_CORPUS",
    "list_sources",
    "get_source",
]
