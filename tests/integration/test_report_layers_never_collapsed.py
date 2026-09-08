"""Integration tests ensuring 3-layer report separation is never collapsed or breached."""
import json
from app.api.schemas import (
    LLMInterpretation,
    ReportResult,
    ReportSection,
    RetrievedEvidence,
    TechnicalFinding,
)
from app.report.generate import generate_report
from tests.report.test_generate_report import FakeGroqClient, fake_retrieval_mixed


def test_every_section_has_three_distinct_typed_layers(real_pipeline):
    """Ensure every report section preserves three separate, top-level typed layers."""
    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=FakeGroqClient(),
        retrieval_fn=fake_retrieval_mixed,
    )

    validated_report = ReportResult(**out)
    assert validated_report is not None
    assert len(out["sections"]) == 5

    for section in out["sections"]:
        # Layer keys must be present at top level of the section dict, not nested in each other
        assert "technical_finding" in section
        assert "retrieved_evidence" in section
        assert "llm_interpretation" in section

        # Pydantic validation of the whole section and individual layer models
        sec_model = ReportSection(**section)
        assert sec_model is not None
        assert TechnicalFinding(**section["technical_finding"]) is not None
        assert RetrievedEvidence(**section["retrieved_evidence"]) is not None
        assert LLMInterpretation(**section["llm_interpretation"]) is not None


def test_llm_text_isolated_to_layer_3(real_pipeline):
    """Ensure LLM generated prose is strictly confined to Layer 3 and cannot leak into Layers 1 or 2."""
    sentinel_dict = {
        k: f"SENTINEL_{k}_PROSE"
        for k in ["model", "explainability", "fairness", "drift", "compliance"]
    }
    fake_client = FakeGroqClient(canned_response_dict=sentinel_dict)

    out = generate_report(
        model=real_pipeline["model"],
        explainability=real_pipeline["explainability"],
        fairness=real_pipeline["fairness"],
        drift=real_pipeline["drift"],
        compliance=real_pipeline["compliance"],
        llm_client=fake_client,
        retrieval_fn=fake_retrieval_mixed,
    )

    assert len(out["sections"]) == 5
    for section in out["sections"]:
        layer3_json = json.dumps(section["llm_interpretation"])
        layer1_json = json.dumps(section["technical_finding"])
        layer2_json = json.dumps(section["retrieved_evidence"])

        assert "SENTINEL_" in layer3_json, f"Sentinel missing from Layer 3 of section {section['heading']}"
        assert "SENTINEL_" not in layer1_json, f"Sentinel leaked into Layer 1 of section {section['heading']}"
        assert "SENTINEL_" not in layer2_json, f"Sentinel leaked into Layer 2 of section {section['heading']}"
