"""Phase 4 tests: assurance report presentation panel (owner: Nidhi).

Covers presentation only. This panel does not call the LLM, does not call
RAG retrieval, and does not compute or reclassify any technical finding --
these tests verify it renders whatever ``ReportResult``-shaped dict it is
given, faithfully, keeping the four report layers separate and rendering
evidence-honesty rules (NOT_FOUND wording, is_excerpt/is_current, missing
citation fields as "not stated").

Uses the project's existing ``streamlit.testing.v1.AppTest`` convention,
applied to the panel's render function via ``AppTest.from_function``.
"""
import ast
import inspect

from streamlit.testing.v1 import AppTest

from app.api.mock_data import MOCK_REPORT_RESULT


def _run(data: dict, source: str = "api") -> AppTest:
    """Render the panel through AppTest and return the finished app."""

    def _wrapper(data, source):
        from dashboard.panels.report_panel import render_report_panel
        render_report_panel(data=data, source=source)

    at = AppTest.from_function(_wrapper, kwargs={"data": data, "source": source})
    at.run(timeout=15)
    return at


def _all_text(at: AppTest) -> str:
    """Flatten every text-bearing element the panel could have produced."""
    parts: list[str] = []
    # Note: st.write(str) is dispatched by Streamlit itself to st.markdown,
    # so AppTest exposes no separate `.write` collection -- markdown covers
    # both call sites.
    for group in (at.markdown, at.caption, at.info, at.warning, at.error):
        parts.extend(el.value for el in group)
    for exp in at.expander:
        parts.append(exp.proto.label)
    return "\n".join(str(p) for p in parts)


# A "real-shaped" fixture matching production generate_report() output shape,
# covering: RETRIEVED with full citation metadata, NOT_ATTEMPTED, and a
# supporting_evidence record. Built for test coverage only -- not a claim
# about what production data currently contains.
_REAL_SHAPED_REPORT = {
    "report_id": "test-report-001",
    "generated_at": "2026-09-12T00:00:00Z",
    "model_version": "test-model-v1",
    "is_mock": False,
    "evidence_coverage": {"retrieved": 1, "not_found": 1, "total": 2},
    "disclaimers": ["This is a synthetic test fixture, not production output."],
    "sections": [
        {
            "heading": "Fairness",
            "technical_finding": {
                "status": "WARNING",
                "ref": "fairness.demographic_parity",
                "source_module": "app.fairness",
                "provenance": "computed",
                "value": {"demographic_parity_difference": 0.12},
            },
            "llm_interpretation": {
                "text": "MARKER_INTERPRETATION_TEXT",
                "regulatory_basis": "RBI-FAIR-1",
                "grounded_in": ["fairness.demographic_parity"],
                "is_mock": False,
            },
            "supporting_evidence": [
                {
                    "evidence_type": "fairness_group",
                    "group": "personal_status_and_sex",
                    "note": "synthetic supporting evidence for test coverage",
                }
            ],
            "retrieved_evidence": {
                "evidence_status": "RETRIEVED",
                "citations": [
                    {
                        "source": "RBI Master Direction on Fair Practices",
                        "locator": "Section 4.2",
                        "quote": "Lenders must not discriminate on prohibited grounds.",
                        "provenance": "verified",
                        "source_url": "https://rbi.org.in/example-doc",
                        "publication_date": "2014-03-01",
                        "document_type": "Master Direction",
                        "is_excerpt": True,
                        "is_current": False,
                    }
                ],
            },
        },
        {
            "heading": "Drift",
            "technical_finding": {
                "status": "PASS",
                "ref": "drift.psi",
                "source_module": "app.drift",
                "provenance": "computed",
                "value": {"psi": 0.05},
            },
            "llm_interpretation": {
                "text": "MARKER_DRIFT_INTERPRETATION",
                "regulatory_basis": "not stated",
                "grounded_in": [],
                "is_mock": False,
            },
            "supporting_evidence": [],
            "retrieved_evidence": {
                "evidence_status": "NOT_ATTEMPTED",
                "citations": [],
            },
        },
    ],
}

_MISSING_FIELDS_CITATION_REPORT = {
    "report_id": "test-report-002",
    "generated_at": "2026-09-12T00:00:00Z",
    "model_version": "test-model-v1",
    "is_mock": False,
    "evidence_coverage": {"retrieved": 1, "not_found": 0, "total": 1},
    "disclaimers": [],
    "sections": [
        {
            "heading": "Compliance",
            "technical_finding": {"status": "PASS", "ref": "compliance.rule_1"},
            "llm_interpretation": {},
            "supporting_evidence": [],
            "retrieved_evidence": {
                "evidence_status": "RETRIEVED",
                "citations": [
                    {
                        "source": "RBI Circular X",
                        "locator": "para 2",
                        "quote": "quoted text",
                        "provenance": "illustrative",
                        # source_url, publication_date, document_type,
                        # is_excerpt, is_current all intentionally omitted.
                    }
                ],
            },
        }
    ],
}


# ---------------------------------------------------------------------------
# Realistic rendering
# ---------------------------------------------------------------------------


def test_renders_mock_report_result_without_exception():
    at = _run(MOCK_REPORT_RESULT)
    assert not at.exception, f"Panel raised: {at.exception}"


def test_renders_realistic_live_shaped_report_without_exception():
    at = _run(_REAL_SHAPED_REPORT, source="api")
    assert not at.exception, f"Panel raised: {at.exception}"


# ---------------------------------------------------------------------------
# Citation rendering
# ---------------------------------------------------------------------------


def test_citation_full_metadata_is_rendered():
    at = _run(_REAL_SHAPED_REPORT)
    text = _all_text(at)
    citation = _REAL_SHAPED_REPORT["sections"][0]["retrieved_evidence"]["citations"][0]
    assert citation["source"] in text
    assert citation["locator"] in text
    assert citation["quote"] in text
    assert citation["source_url"] in text
    assert citation["document_type"] in text
    assert citation["publication_date"] in text


def test_is_excerpt_true_renders_as_limited_excerpt_not_complete():
    at = _run(_REAL_SHAPED_REPORT)
    text = _all_text(at)
    assert "Limited excerpt" in text
    assert "not the complete source document" in text


def test_is_current_false_renders_as_not_current_and_never_implies_no_regulation():
    at = _run(_REAL_SHAPED_REPORT)
    text = _all_text(at)
    assert "NOT current" in text
    assert "Do not treat as binding current regulation" in text
    # The only acceptable appearance of "no RBI rule exists" is inside the
    # required NOT_FOUND negation ("does not mean that no RBI rule exists").
    # It must never appear as a bare, unnegated claim.
    for line in text.splitlines():
        if "no RBI rule exists" in line:
            assert "does not mean" in line
        assert "RBI has no requirement" not in line
        assert "no regulation exists" not in line


def test_missing_optional_citation_fields_render_not_stated():
    at = _run(_MISSING_FIELDS_CITATION_REPORT)
    assert not at.exception
    text = _all_text(at)
    # is_excerpt / is_current absent -> rendered as "not stated", not guessed.
    assert "Excerpt status:** not stated" in text
    assert "Currency:** not stated" in text
    assert "Document type:** not stated" in text
    assert "Publication date:** not stated" in text
    assert "Source URL:** not stated" in text


# ---------------------------------------------------------------------------
# Evidence status vocabulary
# ---------------------------------------------------------------------------


def test_not_found_renders_required_wording():
    at = _run(MOCK_REPORT_RESULT)
    text = _all_text(at)
    assert "No verified evidence was retrieved from the current" in text
    assert "approved/indexed corpus" in text
    assert "does not mean that no RBI rule" in text


def test_not_attempted_is_handled_and_distinct_from_not_found():
    at = _run(_REAL_SHAPED_REPORT)
    text = _all_text(at)
    assert "Evidence retrieval was not attempted for this section" in text
    assert "distinct from NOT_FOUND" in text


# ---------------------------------------------------------------------------
# Supporting evidence
# ---------------------------------------------------------------------------


def test_supporting_evidence_renders_when_present():
    at = _run(_REAL_SHAPED_REPORT)
    text = _all_text(at)
    assert "fairness_group" in text
    assert "personal_status_and_sex" in text


def test_supporting_evidence_absent_renders_honest_message():
    at = _run(_REAL_SHAPED_REPORT)  # Drift section has supporting_evidence: []
    text = _all_text(at)
    assert "No supporting evidence records are attached to this section" in text
    assert "does not mean none exists elsewhere" in text


# ---------------------------------------------------------------------------
# Layer separation
# ---------------------------------------------------------------------------


def test_report_layers_remain_separate_and_in_order():
    at = _run(_REAL_SHAPED_REPORT)
    text = _all_text(at)
    for marker in (
        "**1. Technical Finding**",
        "**2. Interpretation**",
        "**3. Supporting Evidence**",
        "**4. Retrieved Regulatory Evidence**",
    ):
        assert marker in text, f"missing layer header: {marker}"

    idx_1 = text.index("**1. Technical Finding**")
    idx_2 = text.index("**2. Interpretation**")
    idx_3 = text.index("**3. Supporting Evidence**")
    idx_4 = text.index("**4. Retrieved Regulatory Evidence**")
    assert idx_1 < idx_2 < idx_3 < idx_4


def test_interpretation_text_never_merges_into_technical_finding_block():
    at = _run(_REAL_SHAPED_REPORT)
    markdown_blocks = [m.value for m in at.markdown]
    # The interpretation marker text must never appear inside a markdown
    # block that also carries the "1. Technical Finding" header -- i.e. the
    # two layers are rendered as genuinely separate elements, not just
    # separate lines of the same string.
    for block in markdown_blocks:
        if "**1. Technical Finding**" in block:
            assert "MARKER_INTERPRETATION_TEXT" not in block
            assert "MARKER_DRIFT_INTERPRETATION" not in block


# ---------------------------------------------------------------------------
# Mock / real distinction
# ---------------------------------------------------------------------------


def test_mock_real_distinction_is_visible_top_level():
    at_mock = _run(MOCK_REPORT_RESULT)
    at_real = _run(_REAL_SHAPED_REPORT)
    mock_text = _all_text(at_mock)
    real_text = _all_text(at_real)
    assert "is_mock: **True**" in mock_text
    assert "is_mock: **False**" in real_text


def test_per_section_interpretation_is_mock_is_visible():
    at = _run(_REAL_SHAPED_REPORT)
    text = _all_text(at)
    assert "is_mock: **False**" in text


# ---------------------------------------------------------------------------
# No recalculation / no fabrication (source inspection)
# ---------------------------------------------------------------------------


def test_panel_does_not_import_generation_or_analytical_modules():
    """The panel must present a report, never generate or recompute one."""
    import dashboard.panels.report_panel as panel_module

    tree = ast.parse(inspect.getsource(panel_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    forbidden_prefixes = (
        "app.report.generate",
        "app.rag",
        "app.compliance",
        "app.config",
        "groq",
        "openai",
    )
    offending = [
        name for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    ]
    assert not offending, f"report_panel.py must not import generation/analytical modules: {offending}"


def test_panel_makes_no_http_calls_itself():
    import dashboard.panels.report_panel as panel_module

    source = inspect.getsource(panel_module)
    assert "requests." not in source
    assert "import requests" not in source
