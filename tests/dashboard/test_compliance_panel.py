"""Phase 4 tests: RBI compliance presentation panel (owner: Nidhi).

Covers presentation only. This panel does not evaluate a rule, does not
call the compliance engine, and does not compute a status -- these tests
verify it renders whatever ``ComplianceResult``-shaped dict it is given,
faithfully, including the empty/mock/illustrative states.

Uses the existing ``streamlit.testing.v1.AppTest`` convention already
established in ``tests/dashboard/test_dashboard_entrypoint.py``, applied to
the panel's render function directly (via ``AppTest.from_function``) rather
than the whole dashboard app, since the panel is a plain function that
takes already-fetched data -- no HTTP call, no Streamlit page of its own.
"""
import ast
import inspect

from streamlit.testing.v1 import AppTest

from app.api.mock_data import MOCK_COMPLIANCE_RESULT


def _run(data: dict, source: str = "api") -> AppTest:
    """Render the panel through AppTest and return the finished app."""

    def _wrapper(data, source):
        from dashboard.panels.compliance_panel import render_compliance_panel
        render_compliance_panel(data=data, source=source)

    at = AppTest.from_function(_wrapper, kwargs={"data": data, "source": source})
    at.run(timeout=15)
    return at


# ---------------------------------------------------------------------------
# Realistic rendering
# ---------------------------------------------------------------------------


def test_realistic_compliance_result_renders_without_exception():
    at = _run(MOCK_COMPLIANCE_RESULT)
    assert not at.exception, f"Panel raised: {at.exception}"


def test_every_finding_renders_as_its_own_expander():
    at = _run(MOCK_COMPLIANCE_RESULT)
    assert len(at.expander) == len(MOCK_COMPLIANCE_RESULT["findings"])


def test_finding_description_and_technical_reference_are_shown():
    at = _run(MOCK_COMPLIANCE_RESULT)
    # Note: the panel renders these via st.write(), which Streamlit itself
    # dispatches to st.markdown -- AppTest exposes no separate `.write`
    # collection, so markdown covers both call sites.
    body_text = "\n".join(m.value for m in at.markdown)
    first = MOCK_COMPLIANCE_RESULT["findings"][0]
    assert first["rule_description"] in body_text
    assert first["technical_finding_ref"] in body_text


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------


def test_status_rendering_uses_only_the_approved_vocabulary():
    from dashboard.api_client import render_status

    at = _run(MOCK_COMPLIANCE_RESULT)
    labels = [e.proto.label for e in at.expander]
    assert len(labels) == len(MOCK_COMPLIANCE_RESULT["findings"])
    for label, finding in zip(labels, MOCK_COMPLIANCE_RESULT["findings"]):
        # The panel must not invent its own status text: the rendered label
        # must be exactly what the shared render_status() helper produces.
        assert render_status(finding["status"]) in label
        assert any(tag in label for tag in ("PASS", "WARNING", "FAIL", "PENDING"))


def test_unknown_status_is_not_silently_normalised():
    data = {
        "findings": [
            {
                "rule_id": "RBI-TEST-99",
                "rule_description": "Synthetic test finding.",
                "technical_finding_ref": "test.ref",
                "status": "SOMETHING_UNEXPECTED",
                "evidence_chunks": [],
            }
        ],
        "is_mock": True,
    }
    at = _run(data)
    assert not at.exception
    label = at.expander[0].proto.label
    assert "SOMETHING_UNEXPECTED" in label


# ---------------------------------------------------------------------------
# is_mock visibility
# ---------------------------------------------------------------------------


def test_is_mock_true_is_visible():
    at = _run({**MOCK_COMPLIANCE_RESULT, "is_mock": True})
    captions = [c.value for c in at.caption]
    assert any("Mock data: **True**" in c for c in captions)


def test_is_mock_false_is_visible_and_not_hidden():
    at = _run({**MOCK_COMPLIANCE_RESULT, "is_mock": False})
    captions = [c.value for c in at.caption]
    assert any("Mock data: **False**" in c for c in captions)


# ---------------------------------------------------------------------------
# Empty states, rendered gracefully and honestly
# ---------------------------------------------------------------------------


def test_empty_findings_are_handled_gracefully():
    at = _run({"findings": [], "is_mock": True})
    assert not at.exception
    assert len(at.expander) == 0
    assert any("No compliance findings" in i.value for i in at.info)


def test_empty_evidence_chunks_render_as_not_populated_not_as_evidence():
    at = _run(MOCK_COMPLIANCE_RESULT)  # every fixture finding has evidence_chunks == []
    assert not at.exception
    captions = "\n".join(c.value for c in at.caption)
    assert "not currently populated" in captions
    # Must not claim RAG evidence lives in this field.
    assert "evidence_chunks" not in captions or "is not the same channel" in captions


def test_non_empty_evidence_chunks_are_rendered_verbatim_and_not_fabricated():
    # SYNTHETIC TEST DATA ONLY: the real compliance engine never populates
    # evidence_chunks today. This exercises the non-empty rendering branch
    # without claiming this reflects real production output.
    synthetic_finding = {
        "rule_id": "RBI-SYNTH-TEST",
        "rule_description": "Synthetic finding for panel test coverage only.",
        "technical_finding_ref": "test.synthetic_ref",
        "status": "PASS",
        "evidence_chunks": ["synthetic-test-chunk-only"],
    }
    at = _run({"findings": [synthetic_finding], "is_mock": True})
    assert not at.exception
    rendered = "\n".join(m.value for m in at.markdown)
    assert "synthetic-test-chunk-only" in rendered
    # Rendered verbatim -- no extra chunk invented alongside it.
    assert rendered.count("synthetic-test-chunk-only") == 1


# ---------------------------------------------------------------------------
# No recalculation / no fabrication (source inspection)
# ---------------------------------------------------------------------------


def test_panel_does_not_import_the_compliance_engine_or_analytical_modules():
    """The panel must present a result, never derive one.

    Asserted by import inspection rather than behavioural probing: if the
    module never imports the rule engine, the RBI rule repository, RAG, or
    the threshold config, it cannot recompute a status or fabricate
    evidence, regardless of what its functions do internally.
    """
    import dashboard.panels.compliance_panel as panel_module

    tree = ast.parse(inspect.getsource(panel_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    forbidden_prefixes = (
        "app.compliance",
        "app.rbi",
        "app.rag",
        "app.config",
        "app.report",
    )
    offending = [
        name for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    ]
    assert not offending, f"compliance_panel.py must not import analytical modules: {offending}"


def test_panel_makes_no_http_calls_itself():
    """Data fetching lives outside the panel (dashboard/api_client.py)."""
    import dashboard.panels.compliance_panel as panel_module

    source = inspect.getsource(panel_module)
    assert "requests." not in source
    assert "import requests" not in source
