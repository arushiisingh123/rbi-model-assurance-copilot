"""Deterministic management guidance.

The risk this module carries is not a wrong number -- it computes none. It is
that management-facing prose quietly asserts something the measurement does not
support: that a model is broken, that an employee is at fault, or that a
finding is a regulatory breach.

These tests therefore pin the LANGUAGE as much as the structure.
"""
import pytest

from app.report.guidance import (
    CHANNEL_EXPLAINABILITY,
    CHANNEL_FAIRNESS,
    CHANNEL_FEATURE_DRIFT,
    CHANNEL_PREDICTION_DRIFT,
    INVESTIGATION_CATEGORIES,
    channel_guidance,
    instance_risk_interpretation,
    investigation_guidance,
)

ALL_CHANNELS = (
    CHANNEL_FEATURE_DRIFT,
    CHANNEL_PREDICTION_DRIFT,
    CHANNEL_FAIRNESS,
    CHANNEL_EXPLAINABILITY,
)

# Words that would turn an observation into an accusation or a diagnosis.
FORBIDDEN_PHRASES = (
    "retrain",
    "is defective",
    "is broken",
    "model failure",
    "employee error",
    "employee is",
    "staff error",
    "misconduct",
    "fraudulent",
    "non-compliant",
    "violation",
    "breach of",
    "root cause is",
    "caused by",
)

# The neutral verbs the module is allowed to use.
EXPECTED_VERBS = ("review", "validate", "investigate", "compare", "assess", "confirm")


def _assertive_text() -> str:
    """Every string that ASSERTS something, joined.

    ``does_not_establish`` is deliberately excluded: it is the negation list,
    and it is *supposed* to contain phrases like "is defective" and "breach of"
    in negated form. Scanning it for those phrases would fail the module for
    saying exactly what it must say.
    """
    parts = []
    interpretation = instance_risk_interpretation()
    parts.append(interpretation["meaning"])
    parts.extend(interpretation["investigate"])
    for channel in ALL_CHANNELS:
        entry = channel_guidance(channel)
        parts.append(entry["means"])
        parts.extend(entry["investigate"])
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Language safety -- the reason this module is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES)
def test_no_assertive_guidance_text_accuses_or_diagnoses(phrase):
    assert phrase not in _assertive_text()


def test_the_negation_list_does_state_what_is_not_established():
    """The complement of the test above: the disclaimers must be explicit.

    Excluding this list from the phrase scan would be meaningless if the list
    were vague, so it is checked here for the opposite property.
    """
    disclaimers = " ".join(
        instance_risk_interpretation()["does_not_establish"]
    ).lower()

    for phrase in ("does not establish",):
        assert phrase in disclaimers
    # And every entry is phrased as a negation, never a bare assertion.
    for entry in instance_risk_interpretation()["does_not_establish"]:
        assert entry.lower().startswith("it does not establish")


def test_every_investigation_step_uses_a_neutral_verb():
    """Each step proposes an action to take, not a conclusion to accept."""
    for channel in ALL_CHANNELS:
        for step in channel_guidance(channel)["investigate"]:
            first_word = step.split()[0].lower().rstrip(",")
            assert first_word in EXPECTED_VERBS, (
                f"{channel}: step starts with {first_word!r}, which is not one "
                f"of the neutral verbs {EXPECTED_VERBS}"
            )


def test_the_module_makes_no_model_call_and_imports_no_analytics():
    """Deterministic by construction: no LLM, no analytics, no randomness.

    Checked on the module's IMPORTS and CALLS rather than on its prose -- the
    docstring legitimately discusses the LLM in order to say it is not used,
    and a substring scan would flag that explanation as a violation.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("app/report/guidance.py").read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    # The whole module may import nothing but typing.
    assert imported <= {"typing"}, f"unexpected imports: {imported - {'typing'}}"


# ---------------------------------------------------------------------------
# Instance interpretation
# ---------------------------------------------------------------------------


def test_a_red_instance_is_explained_as_a_prediction_not_a_verdict():
    result = instance_risk_interpretation(prediction=1, probability=0.91)

    assert "model's estimate" in result["meaning"]
    assert "not a decision" in result["meaning"]


def test_the_four_things_a_red_instance_does_not_establish():
    """The mentor's question is exactly this; the answer must be explicit."""
    disclaimers = " ".join(instance_risk_interpretation()["does_not_establish"]).lower()

    assert "customer" in disclaimers
    assert "employee" in disclaimers
    assert "rbi" in disclaimers
    assert "defective" in disclaimers


def test_investigation_categories_are_offered_without_ranking_a_cause():
    result = instance_risk_interpretation()

    assert result["investigation_categories"] == list(INVESTIGATION_CATEGORIES)
    assert len(result["investigate"]) >= 3


def test_supplied_identity_values_are_echoed_not_recomputed():
    result = instance_risk_interpretation(
        prediction=1, probability=0.77, favorable_label=0
    )

    assert result["prediction"] == 1
    assert result["probability"] == 0.77
    assert result["favorable_label"] == 0


def test_interpretation_works_with_nothing_supplied():
    result = instance_risk_interpretation()

    assert result["prediction"] is None
    assert result["meaning"]


# ---------------------------------------------------------------------------
# Channel guidance selection
# ---------------------------------------------------------------------------


def test_only_actionable_channels_produce_guidance():
    """PASS needs no action; PENDING was never measured."""
    guidance = investigation_guidance(
        {
            CHANNEL_FEATURE_DRIFT: "FAIL",
            CHANNEL_PREDICTION_DRIFT: "PASS",
            CHANNEL_FAIRNESS: "PENDING",
        }
    )

    assert [entry["channel"] for entry in guidance] == [CHANNEL_FEATURE_DRIFT]
    assert guidance[0]["status"] == "FAIL"


def test_warning_is_actionable_too():
    guidance = investigation_guidance({CHANNEL_FAIRNESS: "WARNING"})

    assert len(guidance) == 1
    assert guidance[0]["status"] == "WARNING"


def test_an_all_passing_run_produces_no_guidance():
    assert investigation_guidance({c: "PASS" for c in ALL_CHANNELS}) == []


def test_feature_and_prediction_drift_get_different_guidance():
    """Generic advice would be useless -- they lead an investigator elsewhere."""
    feature = channel_guidance(CHANNEL_FEATURE_DRIFT)
    prediction = channel_guidance(CHANNEL_PREDICTION_DRIFT)

    assert feature["means"] != prediction["means"]
    assert feature["investigate"] != prediction["investigate"]
    assert "input" in feature["means"].lower()
    assert "output" in prediction["means"].lower()


def test_an_unknown_channel_gets_no_invented_guidance():
    assert channel_guidance("something_we_never_wrote_guidance_for") is None
    assert investigation_guidance({"something_unknown": "FAIL"}) == []


def test_fairness_guidance_does_not_assert_discrimination():
    means = channel_guidance(CHANNEL_FAIRNESS)["means"].lower()

    assert "not a determination" in means


def test_guidance_does_not_mutate_its_static_source():
    """A caller editing a returned list must not corrupt the next call."""
    first = channel_guidance(CHANNEL_FAIRNESS)
    first["investigate"].append("mutated")

    assert "mutated" not in channel_guidance(CHANNEL_FAIRNESS)["investigate"]
