"""Tests for app/report/status_meaning.py -- the PDF's status-label lookup.

Confirms this module reads the SAME file the frontend imports
(frontend/src/utils/statusMeaning.json) rather than a private copy of the
data, and that its lookup behaves the same way glossary.js's
describeStatus() does: known tokens resolve to their authored label/glyph/
tone, unknown tokens degrade honestly instead of raising or hiding.
"""
import json

from app.report.status_meaning import _STATUS_MEANING_PATH, describe_status


def test_reads_the_same_file_the_frontend_imports():
    assert _STATUS_MEANING_PATH.name == "statusMeaning.json"
    assert _STATUS_MEANING_PATH.parent.name == "utils"
    assert _STATUS_MEANING_PATH.parent.parent.name == "src"
    assert _STATUS_MEANING_PATH.exists()


def test_known_status_tokens_resolve_to_their_authored_label():
    with open(_STATUS_MEANING_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for token, entry in raw.items():
        resolved = describe_status(token)
        assert resolved["label"] == entry["label"]
        assert resolved["glyph"] == entry["glyph"]
        assert resolved["tone"] == entry["tone"]


def test_evidence_missing_label_matches_the_current_agreed_wording():
    assert describe_status("EVIDENCE_MISSING")["label"] == "Awaiting Organizational Evidence"


def test_unknown_token_degrades_honestly_rather_than_raising():
    resolved = describe_status("SOME_NEW_TOKEN_NOT_YET_CATALOGUED")
    assert resolved["label"] == "some new token not yet catalogued"
    assert resolved["tone"] == "neutral"


def test_empty_token_reports_unknown():
    resolved = describe_status("")
    assert resolved["label"] == "Unknown"
    assert resolved["tone"] == "neutral"
