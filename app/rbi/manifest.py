"""Manifest-driven RBI corpus loader (owner: Nidhi / Manas, corpus phase).

WHAT THIS IS
    The loader for ``app/rbi/corpus_manifest.json``. The corpus is DATA:
    adding, retiring or re-tiering a source document is a manifest edit, not
    a code change. Nothing here hardcodes a document.

WHAT THIS IS NOT
    Not a downloader, and not a source of regulatory text. A manifest entry
    records a document's IDENTITY (title, dates, applicability, domains,
    status). The document's actual words live only in the source PDF, and
    none of the 19 sources is present.

THE DISTINCTION THIS MODULE EXISTS TO ENFORCE
    "We know this document exists and what it governs" is a completely
    different claim from "we have read this document and can cite it".
    Conflating them is how a compliance tool ends up citing a regulation
    nobody fetched. So every entry carries a ``source_status``, and
    ``is_citable()`` is False for every status except ``downloaded``.

    A requirement may therefore be DERIVED from a manifest entry only in the
    sense of "this document is in scope"; no clause, page, section or
    quotation can come from here, because none is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Optional

MANIFEST_PATH = Path(__file__).with_name("corpus_manifest.json")

# Tiers, in the order the corpus specification defines them.
TIER_0 = "tier_0"
TIER_1 = "tier_1"
TIER_2 = "tier_2"
VALID_TIERS = (TIER_0, TIER_1, TIER_2)

# Where a document stands as regulation.
STATUS_CURRENT = "current"
STATUS_DRAFT = "draft"
STATUS_COMMITTEE_REPORT = "committee_report"
STATUS_HISTORICAL = "historical"
STATUS_SUPERSEDED_IN_SCOPE = "superseded_in_scope"
STATUS_TO_BE_SUPERSEDED = "to_be_superseded"
VALID_REGULATORY_STATUS = (
    STATUS_CURRENT,
    STATUS_DRAFT,
    STATUS_COMMITTEE_REPORT,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED_IN_SCOPE,
    STATUS_TO_BE_SUPERSEDED,
)

# Statuses that are NOT the current operative rule. A finding raised against
# one of these must be labelled provisional/historical, never binding.
NON_OPERATIVE_STATUS = (
    STATUS_DRAFT,
    STATUS_COMMITTEE_REPORT,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED_IN_SCOPE,
)

# Whether we actually HAVE the document.
SOURCE_DOWNLOADED = "downloaded"
SOURCE_NOT_DOWNLOADED = "not_downloaded"
SOURCE_EXACT_URL_UNRESOLVED = "exact_url_unresolved"
SOURCE_CURRENT_VERSION_UNRESOLVED = "current_version_unresolved"
VALID_SOURCE_STATUS = (
    SOURCE_DOWNLOADED,
    SOURCE_NOT_DOWNLOADED,
    SOURCE_EXACT_URL_UNRESOLVED,
    SOURCE_CURRENT_VERSION_UNRESOLVED,
)

# Required on every manifest entry. Mirrors the metadata contract in the
# corpus specification. A key may be present-and-null (an honest "unknown");
# it may not be absent (which would be silence about whether it was ever
# considered).
REQUIRED_DOCUMENT_KEYS = (
    "document_id",
    "source",
    "title",
    "circular_number",
    "issued_date",
    "effective_date",
    "last_updated",
    "applicability",
    "domain",
    "priority",
    "source_url",
    "regulatory_status",
    "supersedes",
    "related_documents",
)

REQUIRED_APPLICABILITY_KEYS = (
    "applies_to",
    "excluded",
    "applicability_verified",
    "note",
)


class ManifestError(Exception):
    """The manifest is malformed or internally inconsistent."""


@dataclass(frozen=True)
class RBIDocument:
    """One source document's identity, as recorded in the manifest.

    Every field is transcribed from the corpus specification. Nothing is
    inferred, and an unknown value stays ``None`` rather than being filled
    with a plausible guess.
    """

    document_id: str
    source: str
    title: str
    circular_number: Optional[str]
    issued_date: Optional[str]
    effective_date: Optional[str]
    last_updated: Optional[str]
    priority: str
    regulatory_status: str
    source_url: Optional[str]
    source_status: str
    local_path: Optional[str]
    storage_dir: Optional[str]
    applies_to: tuple[str, ...]
    excluded: tuple[str, ...]
    applicability_verified: bool
    applicability_note: Optional[str]
    requires_conditions: dict[str, list[Any]]
    domain: tuple[str, ...]
    supersedes: tuple[str, ...]
    superseded_by: tuple[str, ...]
    related_documents: tuple[str, ...]
    project_mappings: tuple[str, ...]
    retrieval_concepts: tuple[str, ...]
    provisional: bool
    provisional_note: Optional[str]
    scope_limit: Optional[str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    # -- what may be claimed about this document ---------------------------

    def is_citable(self) -> bool:
        """Whether a finding may quote or cite this document.

        True ONLY when the source has actually been downloaded. Identity
        metadata is not evidence: citing a document nobody fetched would
        manufacture a regulatory reference, which is the single worst thing
        this system could do.
        """
        return self.source_status == SOURCE_DOWNLOADED

    def is_operative(self) -> bool:
        """Whether this is a CURRENT binding rule.

        False for drafts, committee reports, historical and superseded
        sources -- they may inform, but they must never be presented as the
        rule in force, and they must never outrank a current document.
        """
        return self.regulatory_status == STATUS_CURRENT

    def is_provisional(self) -> bool:
        """Draft / committee material, whose findings are consultative."""
        return bool(self.provisional) or self.regulatory_status in (
            STATUS_DRAFT,
            STATUS_COMMITTEE_REPORT,
        )

    def blocking_reason(self) -> Optional[str]:
        """Why this document cannot yet support a citable finding, if so."""
        if self.is_citable():
            return None
        note = self.raw.get("source_status_note")
        base = {
            SOURCE_NOT_DOWNLOADED: (
                f"The source document for {self.document_id} has not been "
                "obtained, so no clause, page or section of it can be cited."
            ),
            SOURCE_EXACT_URL_UNRESOLVED: (
                f"Only an RBI index/listing page is known for {self.document_id}; "
                "the exact document URL is unresolved and needs human action."
            ),
            SOURCE_CURRENT_VERSION_UNRESOLVED: (
                f"Which version of {self.document_id} is operative for the target "
                "entity type is unresolved, so no version of it can be treated "
                "as current."
            ),
        }.get(self.source_status, f"{self.document_id} is not citable.")
        return f"{base} {note}".strip() if note else base


def _require_keys(entry: dict, keys: Iterable[str], where: str) -> list[str]:
    return [f"{where}: missing required key '{k}'" for k in keys if k not in entry]


def validate_document_entry(entry: dict) -> list[str]:
    """Structural problems with one manifest entry. Empty list == valid."""
    where = entry.get("document_id") or "<entry with no document_id>"
    problems = _require_keys(entry, REQUIRED_DOCUMENT_KEYS, where)

    applicability = entry.get("applicability")
    if not isinstance(applicability, dict):
        problems.append(f"{where}: 'applicability' must be an object")
    else:
        problems.extend(
            _require_keys(applicability, REQUIRED_APPLICABILITY_KEYS, f"{where}.applicability")
        )
        for key in ("applies_to", "excluded"):
            if key in applicability and not isinstance(applicability[key], list):
                problems.append(f"{where}.applicability.{key} must be a list")
        verified = applicability.get("applicability_verified")
        if key in applicability and not isinstance(verified, bool):
            problems.append(f"{where}.applicability.applicability_verified must be a bool")

    tier = entry.get("priority")
    if tier not in VALID_TIERS:
        problems.append(f"{where}: priority {tier!r} not in {VALID_TIERS}")

    status = entry.get("regulatory_status")
    if status not in VALID_REGULATORY_STATUS:
        problems.append(f"{where}: regulatory_status {status!r} not in {VALID_REGULATORY_STATUS}")

    source_status = entry.get("source_status")
    if source_status not in VALID_SOURCE_STATUS:
        problems.append(f"{where}: source_status {source_status!r} not in {VALID_SOURCE_STATUS}")

    # A document claiming to be downloaded must say where it is -- otherwise
    # "downloaded" is an unbacked assertion.
    if source_status == SOURCE_DOWNLOADED and not entry.get("local_path"):
        problems.append(f"{where}: source_status 'downloaded' requires a local_path")

    if not isinstance(entry.get("domain"), list):
        problems.append(f"{where}: 'domain' must be a list")

    return problems


def _document_from_entry(entry: dict) -> RBIDocument:
    applicability = entry.get("applicability") or {}
    return RBIDocument(
        document_id=entry["document_id"],
        source=entry["source"],
        title=entry["title"],
        circular_number=entry.get("circular_number"),
        issued_date=entry.get("issued_date"),
        effective_date=entry.get("effective_date"),
        last_updated=entry.get("last_updated"),
        priority=entry["priority"],
        regulatory_status=entry["regulatory_status"],
        source_url=entry.get("source_url"),
        source_status=entry["source_status"],
        local_path=entry.get("local_path"),
        storage_dir=entry.get("storage_dir"),
        applies_to=tuple(applicability.get("applies_to") or ()),
        excluded=tuple(applicability.get("excluded") or ()),
        applicability_verified=bool(applicability.get("applicability_verified", False)),
        applicability_note=applicability.get("note"),
        requires_conditions=dict(applicability.get("requires_conditions") or {}),
        domain=tuple(entry.get("domain") or ()),
        supersedes=tuple(entry.get("supersedes") or ()),
        superseded_by=tuple(entry.get("superseded_by") or ()),
        related_documents=tuple(entry.get("related_documents") or ()),
        project_mappings=tuple(entry.get("project_mappings") or ()),
        retrieval_concepts=tuple(entry.get("retrieval_concepts") or ()),
        provisional=bool(entry.get("provisional", False)),
        provisional_note=entry.get("provisional_note"),
        scope_limit=entry.get("scope_limit"),
        raw=entry,
    )


class RBIManifest:
    """The loaded corpus manifest: documents keyed by ``document_id``."""

    def __init__(self, documents: Iterable[RBIDocument]) -> None:
        self._documents: dict[str, RBIDocument] = {}
        for document in documents:
            if document.document_id in self._documents:
                raise ManifestError(f"duplicate document_id: {document.document_id!r}")
            self._documents[document.document_id] = document

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, document_id: object) -> bool:
        return document_id in self._documents

    def get(self, document_id: str) -> Optional[RBIDocument]:
        return self._documents.get(document_id)

    def all(self) -> list[RBIDocument]:
        return list(self._documents.values())

    def document_ids(self) -> list[str]:
        return list(self._documents)

    def by_tier(self, tier: str) -> list[RBIDocument]:
        return [d for d in self._documents.values() if d.priority == tier]

    def by_domain(self, domain: str) -> list[RBIDocument]:
        return [d for d in self._documents.values() if domain in d.domain]

    def citable(self) -> list[RBIDocument]:
        """Documents that may actually be cited (i.e. downloaded)."""
        return [d for d in self._documents.values() if d.is_citable()]

    def unresolved(self) -> list[RBIDocument]:
        """Documents that cannot support a citable finding, and why."""
        return [d for d in self._documents.values() if not d.is_citable()]

    def coverage_report(self) -> dict[str, Any]:
        """Honest corpus state, for the ingestion CLI and the report layer.

        Deliberately reports what is MISSING as prominently as what is
        present: a corpus summary that only counted successes would let an
        empty corpus look healthy.
        """
        unresolved = self.unresolved()
        by_status: dict[str, list[str]] = {}
        for document in unresolved:
            by_status.setdefault(document.source_status, []).append(document.document_id)

        tier_0 = self.by_tier(TIER_0)
        return {
            "total_documents": len(self),
            "citable_documents": len(self.citable()),
            "unresolved_documents": len(unresolved),
            "unresolved_by_status": by_status,
            "tier_0_total": len(tier_0),
            "tier_0_citable": len([d for d in tier_0 if d.is_citable()]),
            "corpus_ready_for_compliance": bool(tier_0) and all(d.is_citable() for d in tier_0),
            "blocking_reasons": {
                d.document_id: d.blocking_reason() for d in unresolved
            },
        }


def load_manifest(path: Path | str = MANIFEST_PATH) -> RBIManifest:
    """Read, validate and return the corpus manifest.

    Raises ``ManifestError`` listing EVERY problem rather than the first: a
    manifest is edited by hand, and fixing one error at a time across 19
    documents is how entries get left half-corrected.
    """
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"corpus manifest not found at {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"corpus manifest is not valid JSON: {exc}") from exc

    entries = raw.get("documents")
    if not isinstance(entries, list):
        raise ManifestError("corpus manifest must contain a 'documents' list")

    problems: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append(f"document entry must be an object, got {type(entry).__name__}")
            continue
        problems.extend(validate_document_entry(entry))

    if problems:
        raise ManifestError(
            f"{len(problems)} problem(s) in {manifest_path}:\n  - "
            + "\n  - ".join(problems)
        )

    return RBIManifest(_document_from_entry(entry) for entry in entries)


__all__ = [
    "MANIFEST_PATH",
    "TIER_0",
    "TIER_1",
    "TIER_2",
    "VALID_TIERS",
    "STATUS_CURRENT",
    "STATUS_DRAFT",
    "STATUS_COMMITTEE_REPORT",
    "STATUS_HISTORICAL",
    "STATUS_SUPERSEDED_IN_SCOPE",
    "STATUS_TO_BE_SUPERSEDED",
    "VALID_REGULATORY_STATUS",
    "NON_OPERATIVE_STATUS",
    "SOURCE_DOWNLOADED",
    "SOURCE_NOT_DOWNLOADED",
    "SOURCE_EXACT_URL_UNRESOLVED",
    "SOURCE_CURRENT_VERSION_UNRESOLVED",
    "VALID_SOURCE_STATUS",
    "REQUIRED_DOCUMENT_KEYS",
    "ManifestError",
    "RBIDocument",
    "RBIManifest",
    "load_manifest",
    "validate_document_entry",
]
