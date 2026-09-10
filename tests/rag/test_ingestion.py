"""Phase 3, Task 2 tests: RBI document ingestion (owner: Nidhi).

Covers only loading an approved source's stored text + preserving its
provenance metadata. Chunking, embeddings, vector search, retrieval, and
report generation are later tasks and are not exercised here. The Phase 0
smoke-test files are left untouched.
"""
import dataclasses

import pytest

from app.rag.corpus import (
    APPROVED_CORPUS,
    RBI_IRAC_ADVANCES_2014,
    RBICorpus,
    RBISourceMetadata,
)
from app.rag.ingestion import (
    PROVENANCE_FIELDS,
    IngestionError,
    LoadedDocument,
    SourceDocumentNotFoundError,
    load_by_id,
    load_corpus,
    load_source,
    provenance,
)

APPROVED_2014_ID = "rbi-master-circular-irac-advances-2014-07-01"


# ---------------------------------------------------------------------------
# The approved 2014 source loads
# ---------------------------------------------------------------------------


def test_approved_2014_source_loads_successfully():
    loaded = load_by_id(APPROVED_2014_ID)
    assert isinstance(loaded, LoadedDocument)
    assert loaded.doc_id == APPROVED_2014_ID


def test_loaded_text_is_non_empty_and_is_the_real_excerpt():
    loaded = load_source(RBI_IRAC_ADVANCES_2014)
    assert loaded.text.strip()
    lowered = loaded.text.lower()
    # Distinctive strings from the stored verbatim excerpt.
    assert "non performing asset" in lowered
    assert "prudential norms" in lowered
    # Read verbatim -- the file's own provenance header travels through too.
    assert "VERBATIM EXCERPT BEGINS BELOW" in loaded.text


def test_loaded_text_matches_the_file_on_disk_byte_for_byte():
    loaded = load_source(RBI_IRAC_ADVANCES_2014)
    assert loaded.text == loaded.path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Local path resolution
# ---------------------------------------------------------------------------


def test_local_path_resolution_matches_corpus_metadata():
    loaded = load_source(RBI_IRAC_ADVANCES_2014)
    assert loaded.path == RBI_IRAC_ADVANCES_2014.resolved_path()
    assert loaded.path.is_absolute()
    assert loaded.path.is_file()
    assert loaded.path.name == "RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt"


def test_repo_root_override_is_honoured(tmp_path):
    (tmp_path / "data" / "rbi_sources").mkdir(parents=True)
    target = tmp_path / "data" / "rbi_sources" / "x.txt"
    target.write_text("some regulatory text", encoding="utf-8")

    src = RBISourceMetadata(
        doc_id="x",
        title="t",
        issuing_authority="Reserve Bank of India (RBI)",
        document_type="Circular",
        local_path="data/rbi_sources/x.txt",
    )
    loaded = load_source(src, repo_root=tmp_path)
    assert loaded.path == target.resolve()
    assert loaded.text == "some regulatory text"


# ---------------------------------------------------------------------------
# Metadata / provenance is preserved through ingestion
# ---------------------------------------------------------------------------


def test_metadata_object_is_preserved_by_identity():
    loaded = load_source(RBI_IRAC_ADVANCES_2014)
    assert loaded.metadata is RBI_IRAC_ADVANCES_2014


def test_all_provenance_fields_survive_ingestion():
    loaded = load_source(RBI_IRAC_ADVANCES_2014)
    prov = provenance(loaded.metadata)

    assert set(prov) == set(PROVENANCE_FIELDS)
    for name in PROVENANCE_FIELDS:
        assert prov[name] == getattr(RBI_IRAC_ADVANCES_2014, name)

    # Spot-check the exact stored values (nothing rewritten).
    assert prov["doc_id"] == APPROVED_2014_ID
    assert prov["title"].startswith("Master Circular - Prudential Norms")
    assert prov["issuing_authority"] == "Reserve Bank of India (RBI)"
    assert prov["document_type"] == "Master Circular"
    assert prov["publication_date"] == "2014-07-01"
    assert prov["reference_number"] == "RBI/2014-15/74; DBOD.No.BP.BC.9/21.04.048/2014-15"
    assert prov["source_url"].startswith("https://www.rbi.org.in/")
    assert prov["retrieved_date"] == "2026-08-25"
    assert prov["applicable_to"] == "All Commercial Banks (excluding Regional Rural Banks)"


def test_missing_metadata_is_not_invented():
    # The source states no separate effective date -> it stays None.
    prov = provenance(RBI_IRAC_ADVANCES_2014)
    assert prov["effective_date"] is None


def test_provenance_covers_every_metadata_field_except_the_local_path():
    dataclass_fields = {f.name for f in dataclasses.fields(RBISourceMetadata)}
    # local_path is where THIS checkout stores its copy -- a filesystem
    # detail, not a fact about the RBI document -- so it is intentionally
    # not part of provenance. The loaded document still exposes the resolved
    # path via LoadedDocument.path and the full record via .metadata.
    assert set(PROVENANCE_FIELDS) == dataclass_fields - {"local_path"}, (
        "PROVENANCE_FIELDS in app/rag/ingestion.py has drifted from "
        "RBISourceMetadata -- update it so ingestion keeps preserving every "
        "provenance field"
    )


# ---------------------------------------------------------------------------
# excerpt / current provenance survives ingestion
# ---------------------------------------------------------------------------


def test_excerpt_and_current_flags_survive_ingestion():
    loaded = load_source(RBI_IRAC_ADVANCES_2014)
    assert loaded.metadata.is_excerpt is True
    assert loaded.metadata.is_current is False
    assert loaded.metadata.is_verified_current_regulation is False

    prov = provenance(loaded.metadata)
    assert prov["is_excerpt"] is True
    assert prov["is_current"] is False
    assert prov["coverage_note"] and "excerpt" in prov["coverage_note"].lower()
    assert "not be treated as current or binding" in prov["scope_note"].lower()


# ---------------------------------------------------------------------------
# Failing clearly on bad input
# ---------------------------------------------------------------------------


def test_missing_file_fails_clearly():
    src = RBISourceMetadata(
        doc_id="ghost-doc",
        title="Nonexistent",
        issuing_authority="Reserve Bank of India (RBI)",
        document_type="Circular",
        local_path="data/rbi_sources/this_file_does_not_exist.txt",
    )
    with pytest.raises(SourceDocumentNotFoundError) as excinfo:
        load_source(src)

    msg = str(excinfo.value)
    assert "ghost-doc" in msg
    assert "this_file_does_not_exist.txt" in msg
    # Also catchable as the stdlib error and as the ingestion base error.
    assert isinstance(excinfo.value, FileNotFoundError)
    assert isinstance(excinfo.value, IngestionError)


def test_empty_file_fails_clearly(tmp_path):
    target = tmp_path / "empty.txt"
    target.write_text("   \n\t\n", encoding="utf-8")
    src = RBISourceMetadata(
        doc_id="empty-doc",
        title="Empty",
        issuing_authority="Reserve Bank of India (RBI)",
        document_type="Circular",
        local_path="empty.txt",
    )
    with pytest.raises(IngestionError, match="is empty"):
        load_source(src, repo_root=tmp_path)


def test_non_metadata_argument_is_rejected():
    with pytest.raises(TypeError):
        load_source({"doc_id": APPROVED_2014_ID})  # type: ignore[arg-type]


def test_load_by_id_unknown_id_fails_clearly():
    with pytest.raises(IngestionError, match="no approved RBI source"):
        load_by_id("no-such-doc")


# ---------------------------------------------------------------------------
# Multiple registered sources can be loaded
# ---------------------------------------------------------------------------


def test_load_corpus_loads_every_registered_source(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("Alpha regulatory text", encoding="utf-8")
    (tmp_path / "docs" / "b.txt").write_text("Bravo regulatory text", encoding="utf-8")

    a = RBISourceMetadata(
        doc_id="doc-a",
        title="Doc A",
        issuing_authority="Reserve Bank of India (RBI)",
        document_type="Master Direction",
        local_path="docs/a.txt",
    )
    b = RBISourceMetadata(
        doc_id="doc-b",
        title="Doc B",
        issuing_authority="Reserve Bank of India (RBI)",
        document_type="Notification",
        local_path="docs/b.txt",
    )
    corpus = RBICorpus([a, b])

    loaded = load_corpus(corpus, repo_root=tmp_path)
    assert [d.doc_id for d in loaded] == ["doc-a", "doc-b"]
    assert loaded[0].text == "Alpha regulatory text"
    assert loaded[1].text == "Bravo regulatory text"
    assert loaded[0].metadata is a
    assert loaded[1].metadata is b


def test_load_corpus_does_not_skip_an_unreadable_source(tmp_path):
    (tmp_path / "ok.txt").write_text("present", encoding="utf-8")
    ok = RBISourceMetadata(
        doc_id="ok",
        title="OK",
        issuing_authority="RBI",
        document_type="Circular",
        local_path="ok.txt",
    )
    missing = RBISourceMetadata(
        doc_id="missing",
        title="Missing",
        issuing_authority="RBI",
        document_type="Circular",
        local_path="gone.txt",
    )
    corpus = RBICorpus([ok, missing])
    with pytest.raises(SourceDocumentNotFoundError, match="missing"):
        load_corpus(corpus, repo_root=tmp_path)


def test_default_load_corpus_uses_the_approved_corpus():
    loaded = load_corpus()
    assert [d.doc_id for d in loaded] == [s.doc_id for s in APPROVED_CORPUS.all()]
    assert len(loaded) == 1
    assert loaded[0].doc_id == APPROVED_2014_ID
    assert loaded[0].text.strip()


# ---------------------------------------------------------------------------
# The Phase 0 smoke test is undisturbed
# ---------------------------------------------------------------------------


def test_ingestion_does_not_import_the_phase0_smoke_test():
    # The Phase 0 proof of concept still imports unchanged, and the new
    # ingestion layer does not depend on it. (The smoke test's own
    # behaviour is covered, untouched, by tests/rag/test_rag_smoke.py --
    # not re-run here, because run_smoke_test() creates a fixed-name
    # ChromaDB collection and a second call in one session would collide.)
    import ast
    import inspect

    import app.rag.ingestion as ingestion_module
    import app.rag.smoke_test  # noqa: F401 -- unchanged, must still import

    tree = ast.parse(inspect.getsource(ingestion_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert not any("smoke_test" in name for name in imported), (
        f"ingestion must not import the Phase 0 smoke test; imports: {sorted(imported)}"
    )
