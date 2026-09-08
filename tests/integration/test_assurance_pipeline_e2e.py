"""End-to-end integration tests for the full assurance pipeline chain."""
from app.api.orchestration import summarize
from app.api.schemas import AssuranceResult


def test_full_chain_assurance_result_valid(real_pipeline):
    """Validate full chain schema and mock flags from real pipeline execution."""
    result = real_pipeline["assurance_result"]
    validated = AssuranceResult(**result)
    assert validated is not None
    assert result["model"]["is_mock"] is False
    assert result["explainability"]["is_mock"] is False
    assert result["fairness_drift"]["fairness"]["is_mock"] is False
    assert result["fairness_drift"]["drift"]["is_mock"] is False
    assert result["compliance"]["is_mock"] is True


def test_full_chain_summary_uses_only_approved_status_vocab(real_pipeline):
    """Validate that summarized full chain statuses belong only to the approved vocabulary."""
    summary = summarize(real_pipeline["assurance_result"])
    expected_keys = {"model", "explainability", "fairness", "drift", "compliance"}
    assert set(summary.keys()) == expected_keys
    approved_statuses = {"PASS", "WARNING", "FAIL", "PENDING"}
    for domain, status in summary.items():
        assert status in approved_statuses, f"Domain {domain} produced unapproved status: {status}"
