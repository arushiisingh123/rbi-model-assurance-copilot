"""Status display labels -- read from the SAME source the frontend uses.

Single source of truth: frontend/src/utils/statusMeaning.json. That file is
authored once and imported directly by frontend/src/utils/glossary.js for the
web app. This module loads the identical file for the PDF assurance report,
so a status label (e.g. EVIDENCE_MISSING -> "Awaiting Organizational
Evidence") only ever needs to change in one place. Nothing here re-derives,
re-classifies or invents a label -- it is a lookup, exactly like
glossary.js's describeStatus().
"""
import json
from functools import lru_cache
from pathlib import Path

_STATUS_MEANING_PATH = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "utils"
    / "statusMeaning.json"
)


@lru_cache(maxsize=1)
def _status_meaning() -> dict:
    with open(_STATUS_MEANING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def describe_status(token: str) -> dict:
    """Label, glyph and tone for a status token, or an honest fallback.

    Mirrors glossary.js's describeStatus(): an unrecognised token is passed
    through with its raw text rather than hidden behind a generic label.
    """
    if not token:
        return {"label": "Unknown", "glyph": "?", "tone": "neutral"}
    entry = _status_meaning().get(token) or _status_meaning().get(str(token).upper())
    if not entry:
        return {"label": str(token).replace("_", " ").lower(), "glyph": "•", "tone": "neutral"}
    return {"label": entry["label"], "glyph": entry["glyph"], "tone": entry["tone"]}
