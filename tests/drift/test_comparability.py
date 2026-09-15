"""Phase 5 tests: cross-model drift comparability enforcement (owner: Arushi).

Covers the refusal logic only. This module computes no metric, so these tests
assert which pairs may be compared and why a refused pair was refused -- never
a PSI or KS value.

The headline test is ``test_max_aggregation_trap_is_refused``: it pins the
worked example that motivated the whole module, so a future change that
weakens the feature-set rule fails loudly with the arithmetic reason attached.
"""
import ast
import inspect
from pathlib import Path

import pytest

from app.drift import comparability as comparability_module
from app.drift.comparability import (
    COMPARABLE,
    NOT_COMPARABLE,
    REASON_PREFIXES,
    compare_drift_envelopes,
)

# ---------------------------------------------------------------------------
# Envelope builders (plain dicts, mirroring DriftAssuranceEnvelope's shape)
# ---------------------------------------------------------------------------

DEFAULT_FEATURES = ["duration_months", "credit_amount", "age"]


def _result(
    *,
    features=None,
    psi=0.0725,
    ks_statistic=0.0737,
    status="PASS",
    per_feature=None,
):
    features = DEFAULT_FEATURES if features is None else list(features)
    if per_feature is None:
        per_feature = [
            {"feature": name, "psi": 0.01, "ks_statistic": 0.02}
            for name in features
        ]
    return {
        "features_evaluated": features,
        "psi": psi,
        "ks_statistic": ks_statistic,
        "status": status,
        "is_mock": False,
        "per_feature": per_feature,
    }


def _envelope(
    *,
    model_id="german-credit-logistic-regression",
    model_version="0.1.0",
    assurance_run_id="run-a",
    adapter_id=None,
    dataset_id="data/german_credit/german_credit.csv",
    dataset_version=None,
    feature_space="bedaf2bc78799733",
    result=None,
):
    return {
        "context": {
            "model_id": model_id,
            "model_version": model_version,
            "assurance_run_id": assurance_run_id,
            "adapter_id": adapter_id,
        },
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "feature_space": feature_space,
        "result": _result() if result is None else result,
    }


def _prefix_of(outcome) -> str:
    """The stable machine-readable prefix of a refusal reason."""
    reason = outcome["reason"]
    matches = [p for p in REASON_PREFIXES if reason.startswith(p)]
    assert len(matches) == 1, f"reason has no single stable prefix: {reason!r}"
    return matches[0]


# ---------------------------------------------------------------------------
# Comparable outcomes
# ---------------------------------------------------------------------------


def test_identical_envelopes_are_comparable():
    outcome = compare_drift_envelopes(_envelope(), _envelope())

    assert outcome["comparability"] == COMPARABLE
    assert "same model identity" in outcome["reason"]


def test_different_model_id_is_comparable():
    """Comparing two models is the intended Phase 5 use case, not a failure."""
    a = _envelope(model_id="german-credit-logistic-regression")
    b = _envelope(model_id="german-credit-random-forest")

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == COMPARABLE
    # Not reported as a self-comparison, since the identities differ.
    assert "same model identity" not in outcome["reason"]
    assert "satisfy the required compatibility conditions" in outcome["reason"]


def test_different_model_version_is_comparable():
    """Comparing versions of one model is a legitimate assurance question."""
    a = _envelope(model_version="0.1.0")
    b = _envelope(model_version="0.2.0")

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == COMPARABLE


def test_different_assurance_run_id_does_not_block_comparison():
    """Run ids are minted per envelope, so they always differ in practice."""
    a = _envelope(assurance_run_id="f78379e9-46a0-4137-b52c-4c1c95ece8e7")
    b = _envelope(assurance_run_id="9ec0fa41-cb1c-4ff1-b232-d71034b74db5")

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == COMPARABLE


def test_different_adapter_id_does_not_block_comparison():
    a = _envelope(adapter_id=None)
    b = _envelope(adapter_id="sklearn-adapter")

    assert compare_drift_envelopes(a, b)["comparability"] == COMPARABLE


# ---------------------------------------------------------------------------
# The MAX aggregation trap -- the reason this module exists
# ---------------------------------------------------------------------------


def test_max_aggregation_trap_is_refused():
    """Identical shared-feature drift, incomparable aggregates.

    Reproduced from a real ``drift_report()`` run during the Phase 5
    inspection: envelope A evaluated one stable feature, envelope B evaluated
    that same feature -- with an IDENTICAL per-feature PSI -- plus one extra
    volatile feature. The aggregates are 0.0061 (PASS) and 2.9351 (FAIL).

    Nothing the two envelopes share drifted differently. The entire divergence,
    including the PASS/FAIL flip, comes from the aggregate being a MAX over a
    different set of features. Refusing this pair is the whole point of the
    module.

    DO NOT weaken the feature-set rule to make this pair comparable. If a
    future change needs shared-feature comparison, that is a separate
    per-feature intersection feature -- not a relaxation of this check.
    """
    shared_entry = {"feature": "shared", "psi": 0.0061, "ks_statistic": 0.0460}

    a = _envelope(
        result=_result(
            features=["shared"],
            psi=0.0061,
            ks_statistic=0.0460,
            status="PASS",
            per_feature=[shared_entry],
        )
    )
    b = _envelope(
        feature_space="8872dbd91eb6570a",
        result=_result(
            features=["shared", "extra"],
            psi=2.9351,
            ks_statistic=0.2420,
            status="FAIL",
            per_feature=[
                shared_entry,
                {"feature": "extra", "psi": 2.9351, "ks_statistic": 0.2420},
            ],
        ),
    )

    # The shared feature really is identical in both envelopes.
    shared_a = [e for e in a["result"]["per_feature"] if e["feature"] == "shared"][0]
    shared_b = [e for e in b["result"]["per_feature"] if e["feature"] == "shared"][0]
    assert shared_a["psi"] == shared_b["psi"] == 0.0061

    # And the aggregates still diverge, status included.
    assert a["result"]["psi"] == 0.0061 and a["result"]["status"] == "PASS"
    assert b["result"]["psi"] == 2.9351 and b["result"]["status"] == "FAIL"

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "feature_set_mismatch:"
    assert "extra" in outcome["reason"]
    assert "MAX" in outcome["reason"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_different_feature_sets_are_refused():
    a = _envelope(result=_result(features=["age", "credit_amount"]))
    b = _envelope(result=_result(features=["age", "duration_months"]))

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "feature_set_mismatch:"
    # The reason names which features were on which side.
    assert "credit_amount" in outcome["reason"]
    assert "duration_months" in outcome["reason"]


def test_same_members_different_order_is_refused_as_an_ordering_mismatch():
    """Distinguished from a set mismatch, because the fix is different."""
    a = _envelope(result=_result(features=["age", "credit_amount"]))
    b = _envelope(result=_result(features=["credit_amount", "age"]))

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "feature_order_mismatch:"
    assert "different order" in outcome["reason"]


def test_different_feature_space_is_refused():
    """Backstop: identical ordered features but disagreeing fingerprints."""
    a = _envelope(feature_space="aaaaaaaaaaaaaaaa")
    b = _envelope(feature_space="bbbbbbbbbbbbbbbb")

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "feature_space_mismatch:"


def test_different_dataset_id_is_refused():
    a = _envelope(dataset_id="data/german_credit/german_credit.csv")
    b = _envelope(dataset_id="data/other/other.csv")

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "dataset_id_mismatch:"


def test_dataset_version_none_on_both_sides_stays_comparable():
    """Unknown-vs-unknown is accepted deliberately, not by accident.

    Every envelope orchestration builds today carries
    ``dataset_version=None``, so refusing a None/None pair would make every
    real comparison NOT_COMPARABLE. ``dataset_id`` equality already constrains
    the pair. See the note at the dataset_version check in
    ``app/drift/comparability.py``.
    """
    outcome = compare_drift_envelopes(
        _envelope(dataset_version=None), _envelope(dataset_version=None)
    )

    assert outcome["comparability"] == COMPARABLE


def test_missing_dataset_version_key_matches_an_explicit_none():
    a = _envelope(dataset_version=None)
    b = _envelope()
    del b["dataset_version"]

    assert compare_drift_envelopes(a, b)["comparability"] == COMPARABLE


def test_different_dataset_version_is_refused():
    a = _envelope(dataset_version=None)
    b = _envelope(dataset_version="2026-09-01")

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "dataset_version_mismatch:"


# ---------------------------------------------------------------------------
# PENDING and defensive handling
# ---------------------------------------------------------------------------


def test_pending_status_envelope_is_refused():
    a = _envelope()
    b = _envelope(result=_result(features=[], psi=0.0, ks_statistic=0.0,
                                 status="PENDING", per_feature=[]))

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "pending_result:"
    assert "drift_b" in outcome["reason"]


def test_pending_on_the_first_envelope_is_reported_against_that_envelope():
    a = _envelope(result=_result(features=[], status="PENDING", per_feature=[]))
    b = _envelope()

    outcome = compare_drift_envelopes(a, b)

    assert _prefix_of(outcome) == "pending_result:"
    assert "drift_a" in outcome["reason"]


def test_empty_features_without_pending_status_is_still_refused():
    """An empty evaluated-feature list holds no measurement, whatever the status."""
    a = _envelope()
    b = _envelope(result=_result(features=[], status="PASS", per_feature=[]))

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "pending_result:"


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"result": None},
        {"result": {}},
        {"context": {}, "dataset_id": "x", "feature_space": "y"},
    ],
)
def test_hand_built_incomplete_envelope_is_refused_not_crashed(broken):
    """A hand-built envelope missing its result must refuse, never raise."""
    outcome = compare_drift_envelopes(_envelope(), broken)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert _prefix_of(outcome) == "pending_result:"


def test_pending_is_checked_before_dataset_and_feature_conditions():
    """Ordering matters: a PENDING pair reports PENDING, not a later mismatch."""
    a = _envelope(dataset_id="one", result=_result(features=["age"]))
    b = _envelope(
        dataset_id="two",
        feature_space="zzzzzzzzzzzzzzzz",
        result=_result(features=[], status="PENDING", per_feature=[]),
    )

    outcome = compare_drift_envelopes(a, b)

    assert _prefix_of(outcome) == "pending_result:"


def test_dataset_id_is_checked_before_feature_conditions():
    a = _envelope(dataset_id="one", result=_result(features=["age"]))
    b = _envelope(
        dataset_id="two",
        feature_space="zzzzzzzzzzzzzzzz",
        result=_result(features=["credit_amount"]),
    )

    outcome = compare_drift_envelopes(a, b)

    assert _prefix_of(outcome) == "dataset_id_mismatch:"


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_compare_drift_envelopes_is_exported_from_the_drift_package():
    """The one change made to app/drift/__init__.py."""
    from app.drift import compare_drift_envelopes as exported

    assert exported is compare_drift_envelopes


def test_outcome_key_set_matches_the_comparison_schema_shape():
    outcome = compare_drift_envelopes(_envelope(), _envelope())

    assert set(outcome.keys()) == {"comparability", "reason", "drift_a", "drift_b"}


@pytest.mark.parametrize(
    "envelope_b_kwargs",
    [
        {},
        {"dataset_id": "other"},
        {"feature_space": "ffffffffffffffff"},
        {"result": _result(features=["age"])},
    ],
)
def test_both_envelopes_are_preserved_unchanged(envelope_b_kwargs):
    """Every outcome returns both inputs by reference, unmodified."""
    a = _envelope()
    b = _envelope(**envelope_b_kwargs)
    a_snapshot = repr(a)
    b_snapshot = repr(b)

    outcome = compare_drift_envelopes(a, b)

    assert outcome["drift_a"] is a
    assert outcome["drift_b"] is b
    assert repr(a) == a_snapshot
    assert repr(b) == b_snapshot


@pytest.mark.parametrize(
    "envelope_b_kwargs",
    [
        {"dataset_id": "other"},
        {"dataset_version": "2026-09-01"},
        {"feature_space": "ffffffffffffffff"},
        {"result": _result(features=["age"])},
        {"result": _result(features=[], status="PENDING", per_feature=[])},
    ],
)
def test_not_comparable_always_carries_a_non_empty_reason(envelope_b_kwargs):
    outcome = compare_drift_envelopes(_envelope(), _envelope(**envelope_b_kwargs))

    assert outcome["comparability"] == NOT_COMPARABLE
    assert isinstance(outcome["reason"], str)
    assert outcome["reason"].strip()


# ---------------------------------------------------------------------------
# Architecture constraints (source inspection)
# ---------------------------------------------------------------------------


def _module_tree() -> ast.AST:
    return ast.parse(inspect.getsource(comparability_module))


def test_module_imports_nothing_from_the_api_layer():
    """app/drift must not depend on app/api -- the layering runs the other way."""
    imported: set[str] = set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    offending = [
        name
        for name in imported
        if name == "app.api" or name.startswith("app.api.")
    ]
    assert not offending, f"comparability.py must not import app.api: {offending}"

    # Nothing from the wider app package either, beyond the drift module's own
    # needs -- this module reads dictionaries and compares them.
    assert not [name for name in imported if name.startswith("app.")], (
        f"comparability.py should need no app.* imports, found: {sorted(imported)}"
    )


def _is_set_call(node: ast.AST) -> bool:
    """True for a literal ``set(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
    )


def test_module_performs_no_metric_arithmetic():
    """No PSI, KS, rounding, or aggregation may be introduced here.

    Division, multiplication, floor division, modulo and exponentiation have no
    legitimate use in this module and are rejected outright. Subtraction is
    rejected too, EXCEPT between two ``set(...)`` calls -- ``set(a) - set(b)``
    is a set difference used to report which features were on which side, not
    arithmetic on a measurement. Distinguishing the two is the point: a blunt
    "no Sub" rule would fail on correct code and teach the next person to
    delete the check.
    """
    tree = _module_tree()

    forbidden_ops = (ast.Div, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, forbidden_ops)
    ], "comparability.py must not perform metric arithmetic"

    numeric_subtraction = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and not (_is_set_call(node.left) and _is_set_call(node.right))
    ]
    assert not numeric_subtraction, (
        "the only subtraction allowed here is set difference between two "
        "set(...) calls"
    )


def test_module_defines_no_thresholds_and_classifies_nothing():
    """Thresholds stay in app/config/thresholds.py; this module classifies nothing."""
    assert not [
        name for name in dir(comparability_module) if "THRESHOLD" in name.upper()
    ]

    tree = _module_tree()
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and "THRESHOLD" in target.id.upper()
    ], "comparability.py must not define a threshold constant"

    # No comparison against a PSI or disparate-impact band edge.
    band_edges = {0.10, 0.25, 0.70, 0.80}
    offending = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        for operand in [node.left, *node.comparators]
        if isinstance(operand, ast.Constant)
        and isinstance(operand.value, float)
        and operand.value in band_edges
    ]
    assert not offending, "comparability.py must not classify against a threshold"

    # Inspect calls, not text: the module docstring legitimately quotes
    # ``status = classify_psi(psi)`` when explaining why the MAX aggregate
    # cannot be compared across feature sets. Prose may name a classifier;
    # the code must not invoke one.
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "classify_psi" not in called
    assert "classify_disparate_impact" not in called


def test_per_feature_intersection_comparison_is_not_implemented():
    """Explicitly out of scope -- a future extension, not a silent addition."""
    source = Path(comparability_module.__file__).read_text(encoding="utf-8")

    # The module may mention the idea in prose, but must not compute one.
    tree = _module_tree()
    intersections = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)
    ]
    assert not intersections, "no set-intersection comparison belongs here yet"
    assert "intersection(" not in source


# ---------------------------------------------------------------------------
# Interoperability with the real orchestration path (D1)
# ---------------------------------------------------------------------------


def test_real_orchestration_envelopes_interoperate(real_model_output):
    """Two genuinely orchestration-built envelopes compare as COMPARABLE.

    Every other test in this file uses hand-built fixtures, so none of them
    would notice if the real envelope shape drifted away from what this module
    reads. This one runs the actual path -- compute_real_drift() ->
    build_drift_assurance_envelope() -> DriftAssuranceEnvelope -> model_dump()
    -- and feeds the result in, so a shape change breaks a test rather than
    production.

    Deliberately does not recompute PSI/KS or re-derive feature_space; it only
    checks that what orchestration produces is accepted.
    """
    from app.api.orchestration import (
        build_drift_assurance_envelope,
        compute_real_drift,
    )
    from app.api.schemas import DriftAssuranceEnvelope

    drift = compute_real_drift(real_model_output)

    # Two envelopes from the same drift result, differing only in a field that
    # must not affect comparability (model_version).
    envelope_a = DriftAssuranceEnvelope(
        **build_drift_assurance_envelope(drift, model_version="0.1.0")
    )
    envelope_b = DriftAssuranceEnvelope(
        **build_drift_assurance_envelope(drift, model_version="0.2.0")
    )

    outcome = compare_drift_envelopes(
        envelope_a.model_dump(), envelope_b.model_dump()
    )

    assert outcome["comparability"] == COMPARABLE

    # The run ids really did differ (orchestration mints one per envelope),
    # confirming that difference does not block a real comparison.
    assert (
        envelope_a.context.assurance_run_id != envelope_b.context.assurance_run_id
    )


def test_real_feature_space_is_the_value_the_comparison_consumes(real_model_output):
    """The fingerprint orchestration derived is what drives the decision.

    Proven by perturbing it: this test never computes or re-derives a
    feature_space, it only shows that replacing the real one flips the outcome.
    """
    from app.api.orchestration import (
        build_drift_assurance_envelope,
        compute_real_drift,
    )
    from app.api.schemas import DriftAssuranceEnvelope

    drift = compute_real_drift(real_model_output)
    real = DriftAssuranceEnvelope(
        **build_drift_assurance_envelope(drift, model_version="0.1.0")
    ).model_dump()

    # A real, non-empty fingerprint.
    assert isinstance(real["feature_space"], str)
    assert real["feature_space"]

    perturbed = DriftAssuranceEnvelope(
        **build_drift_assurance_envelope(drift, model_version="0.1.0")
    ).model_dump()

    # Built independently from the same drift result, the fingerprint is
    # identical -- it is derived from the evaluated features, not from the
    # per-envelope run id. (This file's fixture default is deliberately a real
    # production fingerprint, so it is not a useful thing to contrast against.)
    assert perturbed["feature_space"] == real["feature_space"]
    assert perturbed["context"]["assurance_run_id"] != real["context"]["assurance_run_id"]

    perturbed["feature_space"] = "0000000000000000"

    refused = compare_drift_envelopes(real, perturbed)

    assert refused["comparability"] == NOT_COMPARABLE
    assert _prefix_of(refused) == "feature_space_mismatch:"
