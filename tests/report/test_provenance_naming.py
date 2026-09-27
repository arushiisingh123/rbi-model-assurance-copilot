"""Phase D2: the provenance literal names the corpus it actually describes.

``interim_single_document`` was accurate when the corpus was one 2014 excerpt.
It is now six RBI Directions, so the name was misleading -- a reader of an API
response would conclude the platform holds a single source.

This is a NAMING change only. The field means exactly what it meant: evidence
retrieved from the indexed corpus, carrying an interim provenance model rather
than a fully verified one. No status, no citation content and no retrieval
behaviour changed with it.
"""
import pathlib

import pytest

from app.api.schemas import Citation

OLD = "interim_single_document"
NEW = "interim_multi_document"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Everything that could emit or assert the literal. Scanned as text rather
# than imported, so a stale copy in a doc or a test is caught too.
SCANNED_DIRS = ("app", "tests", "docs", "frontend/src")


def _scan():
    for directory in SCANNED_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".md", ".js", ".jsx", ".json"}:
                continue
            try:
                yield path, path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue


def test_the_old_literal_is_gone_from_the_repository():
    """Not just from the API path -- from the source and the docs too.

    A stale name left in a doc is how the misleading claim survives a rename.
    """
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path, text in _scan()
        if OLD in text and path.name != pathlib.Path(__file__).name
    ]
    assert offenders == [], f"old provenance literal still present in: {offenders}"


def test_the_schema_accepts_the_new_literal_and_rejects_the_old():
    citation = Citation(source="s", locator="l", quote="q", provenance=NEW)
    assert citation.provenance == NEW

    with pytest.raises(Exception):
        Citation(source="s", locator="l", quote="q", provenance=OLD)


def test_the_other_provenance_values_are_untouched():
    """The rename must not have quietly changed the vocabulary's meaning."""
    for value in ("verified", "illustrative", NEW):
        assert Citation(
            source="s", locator="l", quote="q", provenance=value
        ).provenance == value


def test_no_citation_from_the_live_api_carries_the_old_literal():
    from fastapi.testclient import TestClient

    from app.api.main import app

    body = TestClient(app).get("/compliance").json()
    citations = [c for f in body["findings"] for c in f["citations"]]

    assert citations, "expected the live corpus to support at least one citation"
    for citation in citations:
        assert citation["provenance"] == NEW


def test_the_builder_defaults_to_the_new_literal():
    from app.rag.citations import citation_from_evidence

    class _Evidence:
        doc_id = "d"
        title = "t"
        text = "x"
        chunk_id = "c"
        chunk_index = 0
        source_url = None
        publication_date = None
        document_type = None
        is_excerpt = None
        is_current = None
        provenance = {}

    assert citation_from_evidence(_Evidence()).provenance == NEW
