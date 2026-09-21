"""Phase 3A, Task 1 tests: RBI source metadata + corpus registry (owner: Nidhi).

Covers only metadata/corpus representation. Ingestion, chunking, embeddings,
retrieval, and report generation are later tasks and are not exercised here.
The Phase 0 smoke-test tests (``test_rag_smoke.py``) are deliberately left
untouched.
"""
import dataclasses
from pathlib import Path

import pytest

from app.rag.corpus import (
    APPROVED_CORPUS,
    RBI_IRAC_ADVANCES_2014,
    RBICorpus,
    RBISourceMetadata,
    get_source,
    list_sources,
    validate_source_metadata,
)


# ---------------------------------------------------------------------------
# 1. The approved 2014 RBI source is represented
# ---------------------------------------------------------------------------


def test_approved_source_is_registered_and_retrievable():
    src = get_source("rbi-master-circular-irac-advances-2014-07-01")
    assert src is RBI_IRAC_ADVANCES_2014
    assert list_sources() == [RBI_IRAC_ADVANCES_2014]
    assert len(APPROVED_CORPUS) == 1


def test_approved_source_local_file_exists():
    resolved = RBI_IRAC_ADVANCES_2014.resolved_path()
    assert resolved.is_file()
    assert resolved.name == "RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt"


# ---------------------------------------------------------------------------
# 2. Required metadata values are preserved accurately
#    (every value below appears verbatim in the source file header)
# ---------------------------------------------------------------------------


def test_required_bibliographic_values_are_preserved():
    src = RBI_IRAC_ADVANCES_2014
    assert src.title == (
        "Master Circular - Prudential Norms on Income Recognition, Asset "
        "Classification and Provisioning pertaining to Advances"
    )
    assert src.issuing_authority == "Reserve Bank of India (RBI)"
    assert src.document_type == "Master Circular"
    assert src.local_path == (
        "data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt"
    )


def test_optional_bibliographic_values_are_preserved():
    src = RBI_IRAC_ADVANCES_2014
    assert src.publication_date == "2014-07-01"
    assert src.reference_number == "RBI/2014-15/74; DBOD.No.BP.BC.9/21.04.048/2014-15"
    assert src.source_url == (
        "https://www.rbi.org.in/commonman/Upload/English/Notification/PDFs/"
        "74MIR010714FL.pdf"
    )
    assert src.source_url.startswith("https://www.rbi.org.in/")
    assert src.retrieved_date == "2026-08-25"
    assert src.applicable_to == "All Commercial Banks (excluding Regional Rural Banks)"


def test_effective_date_is_none_because_the_source_states_no_separate_one():
    # The source header prints an issue date only. Nothing is invented.
    assert RBI_IRAC_ADVANCES_2014.effective_date is None


# ---------------------------------------------------------------------------
# 3. The source is clearly marked limited / historical, not current / binding
# ---------------------------------------------------------------------------


def test_source_is_marked_as_a_historical_excerpt():
    src = RBI_IRAC_ADVANCES_2014
    assert src.is_excerpt is True
    assert src.is_current is False
    assert src.is_verified_current_regulation is False


def test_scope_and_coverage_notes_state_the_limitation():
    src = RBI_IRAC_ADVANCES_2014
    assert src.coverage_note and "excerpt" in src.coverage_note.lower()
    assert src.scope_note
    lowered = src.scope_note.lower()
    assert "not be treated as current or binding" in lowered
    # provenance stays separate from interpretation
    assert "no rbi clauses or requirements have been invented" in lowered


def test_a_full_current_document_would_be_verified_current_regulation():
    full_current = dataclasses.replace(
        RBI_IRAC_ADVANCES_2014,
        doc_id="hypothetical-full-current",
        is_excerpt=False,
        is_current=True,
        coverage_note=None,
    )
    assert full_current.is_verified_current_regulation is True


# ---------------------------------------------------------------------------
# 4. Multiple document metadata records can coexist in the registry
# ---------------------------------------------------------------------------


def _minimal_source(doc_id: str) -> RBISourceMetadata:
    return RBISourceMetadata(
        doc_id=doc_id,
        title=f"Placeholder title for {doc_id}",
        issuing_authority="Reserve Bank of India (RBI)",
        document_type="Master Direction",
        local_path=f"data/rbi_sources/{doc_id}.txt",
    )


def test_corpus_holds_and_looks_up_multiple_records():
    a = _minimal_source("doc-a")
    b = _minimal_source("doc-b")
    corpus = RBICorpus([a, b])

    assert len(corpus) == 2
    assert corpus.get("doc-a") is a
    assert corpus.get("doc-b") is b
    assert corpus.get("missing") is None
    assert "doc-a" in corpus
    assert set(corpus.doc_ids()) == {"doc-a", "doc-b"}
    assert list(corpus) == [a, b]


def test_corpus_can_grow_without_changing_the_interface():
    corpus = RBICorpus([RBI_IRAC_ADVANCES_2014])
    corpus.add(_minimal_source("doc-later"))
    assert len(corpus) == 2
    assert corpus.get("doc-later").document_type == "Master Direction"


def test_corpus_rejects_duplicate_doc_ids():
    corpus = RBICorpus([_minimal_source("dup")])
    with pytest.raises(ValueError, match="duplicate doc_id"):
        corpus.add(_minimal_source("dup"))


def test_corpus_rejects_non_metadata_objects():
    corpus = RBICorpus()
    with pytest.raises(TypeError):
        corpus.add({"doc_id": "not-a-record"})


# ---------------------------------------------------------------------------
# 5. Invalid / incomplete metadata is handled clearly and predictably
# ---------------------------------------------------------------------------


def test_missing_required_field_raises_value_error():
    with pytest.raises(ValueError, match="title is required"):
        RBISourceMetadata(
            doc_id="x",
            title="   ",
            issuing_authority="RBI",
            document_type="Circular",
            local_path="data/rbi_sources/x.txt",
        )


def test_blank_doc_id_raises_value_error():
    with pytest.raises(ValueError, match="doc_id is required"):
        RBISourceMetadata(
            doc_id="",
            title="t",
            issuing_authority="RBI",
            document_type="Circular",
            local_path="data/rbi_sources/x.txt",
        )


def test_non_bool_flag_raises_value_error():
    with pytest.raises(ValueError, match="is_current must be a bool"):
        RBISourceMetadata(
            doc_id="x",
            title="t",
            issuing_authority="RBI",
            document_type="Circular",
            local_path="data/rbi_sources/x.txt",
            is_current="yes",  # type: ignore[arg-type]
        )


def test_excerpt_without_coverage_note_raises_value_error():
    with pytest.raises(ValueError, match="is_excerpt=True requires"):
        RBISourceMetadata(
            doc_id="x",
            title="t",
            issuing_authority="RBI",
            document_type="Circular",
            local_path="data/rbi_sources/x.txt",
            is_excerpt=True,
        )


def test_validate_source_metadata_reports_without_raising():
    # The advisory validator returns a problem list for the same bad input
    # that __post_init__ would reject -- but never raises itself.
    ok = _minimal_source("ok")
    assert validate_source_metadata(ok) == []

    # Force an invalid state past the frozen dataclass to exercise the
    # advisory path directly.
    broken = _minimal_source("broken")
    object.__setattr__(broken, "title", "")
    object.__setattr__(broken, "is_excerpt", True)
    problems = validate_source_metadata(broken)
    assert any("title is required" in p for p in problems)
    assert any("coverage_note" in p for p in problems)


# ---------------------------------------------------------------------------
# 6. The Phase 0 smoke test still lines up with the corpus (read-only check)
# ---------------------------------------------------------------------------


def test_corpus_path_matches_the_phase0_smoke_test_document():
    from app.rag.smoke_test import DOCUMENT_PATH

    assert RBI_IRAC_ADVANCES_2014.local_path == str(DOCUMENT_PATH).replace("\\", "/")
    assert RBI_IRAC_ADVANCES_2014.resolved_path() == (
        Path(__file__).resolve().parents[2] / DOCUMENT_PATH
    ).resolve()


# ---------------------------------------------------------------------------
# 7. Phase A: the manifest as the canonical corpus registry
# ---------------------------------------------------------------------------


def test_corpus_from_manifest_returns_only_downloaded_documents():
    """A document nobody obtained must not enter the corpus.

    A corpus entry for an absent document is a citation waiting to happen:
    everything downstream treats corpus membership as "we have this".
    """
    from app.rbi.manifest import load_manifest
    from app.rag.corpus import corpus_from_manifest

    manifest = load_manifest()
    corpus = corpus_from_manifest(manifest)

    assert set(corpus.doc_ids()) == {d.document_id for d in manifest.citable()}
    assert len(corpus) < len(manifest), "not every declared document is present"


def test_corpus_from_manifest_files_all_exist_on_disk():
    from app.rag.corpus import corpus_from_manifest

    for source in corpus_from_manifest():
        assert source.resolved_path().is_file(), source.doc_id


def test_manifest_corpus_excludes_the_2014_regression_fixture():
    """The fixture is not a manifest document and must not become one.

    It exists to prove the non-PDF ingestion path still works. Folding it
    into the manifest corpus would mean a manifest edit could silently
    remove that proof.
    """
    from app.rag.corpus import corpus_from_manifest

    assert RBI_IRAC_ADVANCES_2014.doc_id not in corpus_from_manifest()
    # ...and it is still reachable where it always was.
    assert APPROVED_CORPUS.get(RBI_IRAC_ADVANCES_2014.doc_id) is not None


def test_approved_corpus_is_unchanged_by_the_manifest_work():
    assert list_sources() == [RBI_IRAC_ADVANCES_2014]
    assert len(APPROVED_CORPUS) == 1


def test_manifest_metadata_is_copied_not_inferred():
    """Currency is read from the entry, never derived from its status.

    The NBFC outsourcing circular is recorded is_current=false while its
    regulatory_status is a manifest-vocabulary value. Deriving one from the
    other would silently promote an unverified document to "in force".
    """
    from app.rag.corpus import corpus_from_manifest

    source = corpus_from_manifest().get("RBI-FS-OUTSOURCE-NBFC-2017")

    assert source is not None
    assert source.is_current is False
    assert source.is_verified_current_regulation is False
    assert source.reference_number == "RBI/2017-18/87; DNBR.PD.CC.No.090/03.10.001/2017-18"
    assert source.issuing_authority == "Reserve Bank of India (RBI)"


def test_a_downloaded_entry_missing_required_metadata_is_rejected():
    """Silence about issuing authority must fail loudly, not default."""
    from app.rbi.manifest import RBIDocument
    from app.rag.corpus import ManifestSourceError, source_from_manifest_document

    document = RBIDocument(
        document_id="X", source="RBI", title="T", circular_number=None,
        issued_date=None, effective_date=None, last_updated=None,
        priority="tier_1", regulatory_status="current", source_url=None,
        source_status="downloaded", local_path="data/x.pdf", storage_dir=None,
        applies_to=(), excluded=(), applicability_verified=False,
        applicability_note=None, requires_conditions={}, domain=(),
        supersedes=(), superseded_by=(), related_documents=(),
        project_mappings=(), retrieval_concepts=(), provisional=False,
        provisional_note=None, scope_limit=None, raw={},
    )

    with pytest.raises(ManifestSourceError) as excinfo:
        source_from_manifest_document(document)
    assert "issuing_authority" in str(excinfo.value)


def test_an_undownloaded_entry_cannot_become_a_source():
    from app.rbi.manifest import load_manifest
    from app.rag.corpus import ManifestSourceError, source_from_manifest_document

    absent = load_manifest().unresolved()[0]

    with pytest.raises(ManifestSourceError):
        source_from_manifest_document(absent)
