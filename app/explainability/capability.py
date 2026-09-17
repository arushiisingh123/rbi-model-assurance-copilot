"""Explainability capability detection and explainer routing (owner: Manas, Phase 5).

WHAT THIS MODULE DOES
---------------------
Answers one question for a given ``ModelAdapter``: *which explanation methods
can honestly be produced for this model, by which explainer, on which scale,
and with what fidelity?*

WHAT THIS MODULE DOES NOT DO
----------------------------
It never executes SHAP or LIME, never loads data, and never computes a
contribution. It is a pure decision layer. Executing the decision belongs to
``app/explainability/explain.py`` (Step 4). Keeping the decision separate from
the execution is what makes the decision testable without running either
library.

WHY CAPABILITIES AND NOT MODEL TYPES
------------------------------------
The system must support models this repository has never seen. Routing on a
concrete estimator class, or on an adapter's ``model_type`` string, only works
for models already on an allowlist -- every new model silently falls off the
end. So every decision here is derived from *observable structure*:

- does the adapter actually hand back a fitted artifact?
- does that artifact expose fitted linear coefficients?
- does it expose an ensemble of sub-estimators / a booster?
- does the adapter contract promise probabilities?
- is there reference data to explain against?
- can the model be scored in batches?

No estimator class is imported, named, or type-checked anywhere in this file.
A future model family that exposes recognisable structure is routed correctly
without this module changing; one that exposes none is routed to an
approximate black-box explainer rather than being rejected for being unknown.

CAPABILITY HONESTY
------------------
An adapter declaring ``capabilities["explainability"] = True`` is NOT evidence
that SHAP or LIME can be produced -- that flag is a claim, not an observation,
and at least one adapter in the current registry declares it while being
unexplainable. This module ignores that flag entirely and derives method
availability from the requirements each method actually has.

Where a method cannot be produced, the result is an explicit, structured
``ExplainerDecision`` with ``available=False`` and a human-readable reason --
never ``None``, never a bare exception for the caller to interpret, and never
a silent degradation to a different model's numbers.

SCALE IS A PROPERTY OF THE EXPLAINER, NOT OF THE METHOD
-------------------------------------------------------
This is easy to get wrong and important. "SHAP" does not imply one scale:

- SHAP over a linear structure is additive to the decision function, so its
  contributions are on the LOG-ODDS scale.
- SHAP over a tree structure is additive to ``predict_proba``, so its
  contributions are on the PROBABILITY scale.

Two models' SHAP values can therefore be on different scales, and must never
be compared, averaged, or plotted on a shared axis. Every decision carries its
own ``scale`` so a downstream consumer cannot assume one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SUPPORTED_METHODS",
    "METHOD_SHAP",
    "METHOD_LIME",
    "SCALE_LOG_ODDS",
    "SCALE_PROBABILITY",
    "FIDELITY_EXACT",
    "FIDELITY_APPROXIMATE",
    "FIDELITY_SURROGATE",
    "EXPLAINER_LINEAR",
    "EXPLAINER_TREE",
    "EXPLAINER_KERNEL",
    "EXPLAINER_LIME",
    "ExplainerCapability",
    "ExplainerDecision",
    "detect",
    "select_explainer",
    # Structural probes, public so the execution layer (explain.py) can locate
    # a model's parts the same way this module classified them, instead of
    # reimplementing the probing and risking the two drifting apart.
    "is_pipeline_like",
    "final_estimator",
    "fitted_column_transformer",
]

METHOD_SHAP = "shap"
METHOD_LIME = "lime"
SUPPORTED_METHODS: Tuple[str, ...] = (METHOD_SHAP, METHOD_LIME)

# Units contributions are expressed in. Carried on every decision so a
# consumer never has to infer one from the method name.
SCALE_LOG_ODDS = "log_odds"
SCALE_PROBABILITY = "probability"

# How faithful the explanation is to the model's actual behaviour.
FIDELITY_EXACT = "exact"              # additive decomposition of the model itself
FIDELITY_APPROXIMATE = "approximate"  # sampled estimate of the same quantity
FIDELITY_SURROGATE = "surrogate"      # local model fitted to mimic the model

EXPLAINER_LINEAR = "LinearExplainer"
EXPLAINER_TREE = "TreeExplainer"
EXPLAINER_KERNEL = "KernelExplainer"
EXPLAINER_LIME = "LimeTabularExplainer"


# ---------------------------------------------------------------------------
# Structural probes
#
# Every probe below asks "does this object expose the attribute that makes the
# technique applicable?" rather than "is this object an instance of a class I
# recognise?". That is the whole point of the module, so each probe is kept
# tiny and named for the capability it detects, not for the family it happens
# to match today.
# ---------------------------------------------------------------------------


def _is_pipeline_like(obj: Any) -> bool:
    """Whether ``obj`` behaves like a fitted multi-step estimator chain.

    Duck-typed on ``steps`` / ``named_steps`` rather than an import, so any
    chain object exposing the same surface is handled. Deliberately does NOT
    require particular step *names*: this module must work with a chain whose
    steps are called anything at all.
    """
    steps = getattr(obj, "steps", None)
    if not isinstance(steps, Sequence) or len(steps) == 0:
        return False
    if not hasattr(obj, "named_steps"):
        return False
    return all(isinstance(step, tuple) and len(step) == 2 for step in steps)


def _final_estimator(obj: Any) -> Any:
    """The object that actually produces predictions.

    For a chain that is the last step; for a bare estimator it is the object
    itself. Structural position, not step name -- a chain is free to call its
    final step whatever it likes.
    """
    if _is_pipeline_like(obj):
        return obj.steps[-1][1]
    return obj


def _fitted_column_transformer(obj: Any) -> Optional[Any]:
    """The fitted column-wise transformer in a chain, if there is one.

    Identified by the fitted attribute a column-wise transformer exposes
    (``transformers_``), not by step name or class. Returns ``None`` when the
    chain has no such step, which is a legitimate shape -- an estimator may
    consume raw features directly.
    """
    if not _is_pipeline_like(obj):
        return None
    for _name, step in obj.steps:
        if hasattr(step, "transformers_"):
            return step
    return None


def _has_linear_structure(estimator: Any) -> bool:
    """Whether the estimator exposes fitted linear coefficients.

    ``coef_`` exists only after fitting, so this doubles as a fitted-ness
    check. An exact additive attribution is available for such an estimator.
    """
    return hasattr(estimator, "coef_")


def _has_tree_structure(estimator: Any) -> bool:
    """Whether the estimator exposes a fitted tree or ensemble-of-trees.

    Covers the three structural shapes trees present in practice: a fitted
    ensemble of sub-estimators, a gradient-boosted booster handle, or a
    single fitted tree. Any one of them makes an exact tree attribution
    available.
    """
    return (
        hasattr(estimator, "estimators_")
        or hasattr(estimator, "get_booster")
        or hasattr(estimator, "tree_")
    )


# ---------------------------------------------------------------------------
# Public aliases for the structural probes
#
# The execution layer must locate a model's preprocessing step and final
# estimator using EXACTLY the probes that classified it here. Exposing these
# rather than letting explain.py reimplement them is what keeps the decision
# and the execution from disagreeing about what a model's structure is.
# ---------------------------------------------------------------------------


def is_pipeline_like(obj: Any) -> bool:
    """Whether ``obj`` behaves like a fitted multi-step estimator chain."""
    return _is_pipeline_like(obj)


def final_estimator(obj: Any) -> Any:
    """The object that actually produces predictions (last step, or itself)."""
    return _final_estimator(obj)


def fitted_column_transformer(obj: Any) -> Optional[Any]:
    """The fitted column-wise transformer in a chain, or None."""
    return _fitted_column_transformer(obj)


# ---------------------------------------------------------------------------
# Capability record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplainerCapability:
    """What is observably true about one adapter, for explainability purposes.

    ``model_id`` / ``model_type`` / ``integration_type`` are carried for
    traceability and provenance ONLY. Nothing in ``select_explainer()`` reads
    them, and nothing ever should: routing on identity strings is exactly the
    allowlist behaviour this module exists to avoid.
    """

    model_id: str
    model_type: str
    integration_type: str

    has_local_artifact: bool
    is_pipeline: bool
    has_linear_structure: bool
    has_tree_structure: bool
    supports_probability: bool
    has_background_data: bool
    has_raw_transformed_mapping: bool
    supports_batch_scoring: bool

    # Observations worth surfacing that are not themselves capabilities --
    # e.g. why an artifact could not be loaded.
    notes: Tuple[str, ...] = field(default=())

    @property
    def available_methods(self) -> Tuple[str, ...]:
        """The methods that can actually be produced for this adapter.

        Derived by asking ``select_explainer()`` rather than stored, so it can
        never drift out of step with the routing decision itself.
        """
        return tuple(
            method
            for method in SUPPORTED_METHODS
            if select_explainer(self, method).available
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view, for provenance and for tests."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "integration_type": self.integration_type,
            "has_local_artifact": self.has_local_artifact,
            "is_pipeline": self.is_pipeline,
            "has_linear_structure": self.has_linear_structure,
            "has_tree_structure": self.has_tree_structure,
            "supports_probability": self.supports_probability,
            "has_background_data": self.has_background_data,
            "has_raw_transformed_mapping": self.has_raw_transformed_mapping,
            "supports_batch_scoring": self.supports_batch_scoring,
            "notes": list(self.notes),
            "available_methods": list(self.available_methods),
        }


@dataclass(frozen=True)
class ExplainerDecision:
    """How (or whether) one method should be executed for one adapter.

    An unavailable decision is a first-class result, not an error: it carries
    ``available=False`` plus the reason, so a caller can report the limitation
    instead of guessing, crashing, or quietly substituting something else.
    """

    method: str
    available: bool
    explainer: Optional[str] = None
    scale: Optional[str] = None
    fidelity: Optional[str] = None
    limitations: Tuple[str, ...] = field(default=())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "available": self.available,
            "explainer": self.explainer,
            "scale": self.scale,
            "fidelity": self.fidelity,
            "limitations": list(self.limitations),
        }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect(adapter: Any) -> ExplainerCapability:
    """Observe what one adapter can support, without executing any explainer.

    Every field is established by probing the adapter and (where available)
    its artifact. Nothing is taken on the adapter's word except the two things
    the adapter contract makes authoritative: ``supports_probability`` and the
    ``capabilities`` flags it declares for batch scoring.

    Notably, ``capabilities["explainability"]`` is NOT consulted. It is a
    self-declared claim, and at least one adapter in the current registry
    declares it while being unexplainable in practice.
    """
    notes: List[str] = []

    artifact, artifact_note = _load_artifact(adapter)
    if artifact_note is not None:
        notes.append(artifact_note)

    has_local_artifact = artifact is not None
    is_pipeline = has_local_artifact and _is_pipeline_like(artifact)

    if has_local_artifact:
        estimator = _final_estimator(artifact)
        has_linear_structure = _has_linear_structure(estimator)
        has_tree_structure = _has_tree_structure(estimator)
        has_raw_transformed_mapping = _detect_mapping(artifact, is_pipeline, notes)
    else:
        has_linear_structure = False
        has_tree_structure = False
        has_raw_transformed_mapping = False

    return ExplainerCapability(
        model_id=str(getattr(adapter, "model_id", "")),
        model_type=str(getattr(adapter, "model_type", "")),
        integration_type=str(getattr(adapter, "integration_type", "")),
        has_local_artifact=has_local_artifact,
        is_pipeline=is_pipeline,
        has_linear_structure=has_linear_structure,
        has_tree_structure=has_tree_structure,
        supports_probability=_detect_probability(adapter, notes),
        has_background_data=_detect_background(adapter, notes),
        has_raw_transformed_mapping=has_raw_transformed_mapping,
        supports_batch_scoring=_detect_batch_scoring(adapter, notes),
        notes=tuple(notes),
    )


def _load_artifact(adapter: Any) -> Tuple[Optional[Any], Optional[str]]:
    """Try to obtain the fitted artifact; report why if we cannot.

    Artifact availability is established by ASKING, not by inspecting
    ``integration_type``. A remote-integration adapter that can still hand
    back a local artifact is explainable exactly like any other, and an
    in-process adapter whose artifact is missing must not be treated as
    though it had one.
    """
    loader = getattr(adapter, "load_fitted_model", None)
    if loader is None:
        return None, "Adapter exposes no load_fitted_model(); treated as having no artifact."
    try:
        artifact = loader()
    except NotImplementedError:
        return None, "Adapter has no local model artifact (load_fitted_model() is not implemented)."
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return None, (
            f"Local model artifact could not be loaded "
            f"({type(exc).__name__}: {exc}); treated as unavailable."
        )
    if artifact is None:
        return None, "Adapter's load_fitted_model() returned None; treated as having no artifact."
    return artifact, None


def _detect_mapping(artifact: Any, is_pipeline: bool, notes: List[str]) -> bool:
    """Whether raw-feature attribution looks reachable for this artifact.

    STRUCTURAL PRECONDITION ONLY -- deliberately not the full check.

    Explaining in a transformed space and reporting in the raw space requires
    mapping every transformed column back to the raw feature that produced it,
    and that mapping is only *correct* if it tiles the transformed space
    exactly once. Verifying the tiling requires the fitted transformer, the
    adapter's raw schema, and transformed reference data -- i.e. execution,
    which this module does not do.

    So this reports whether a mapping is structurally possible, and the
    execution layer (Step 4) revalidates it and fails loudly if the tiling
    does not hold. Being wrong in the optimistic direction here cannot produce
    a wrong explanation; it can only produce a loud failure later.
    """
    if not is_pipeline:
        # A bare estimator consumes its features directly, so raw space and
        # model space are the same space -- the mapping is the identity.
        return True
    if _fitted_column_transformer(artifact) is not None:
        return True
    notes.append(
        "Chained estimator exposes no fitted column-wise transformer, so "
        "transformed-to-raw feature attribution could not be established."
    )
    return False


def _detect_probability(adapter: Any, notes: List[str]) -> bool:
    """Read the adapter contract's probability capability.

    This one IS taken from the adapter, because the contract makes it
    authoritative: ``supports_probability`` is defined as the adapter's own
    statement about whether ``predict_proba()`` will work.
    """
    try:
        return bool(adapter.supports_probability)
    except Exception as exc:  # noqa: BLE001
        notes.append(
            f"Adapter's supports_probability could not be read "
            f"({type(exc).__name__}: {exc}); treated as unsupported."
        )
        return False


def _detect_background(adapter: Any, notes: List[str]) -> bool:
    """Whether reference/background data is actually present.

    Must be non-null AND non-empty. An empty frame is not a reference
    distribution, and SHAP handed one would produce numbers with nothing
    meaningful behind them.
    """
    getter = getattr(adapter, "background_data", None)
    if getter is None:
        notes.append("Adapter exposes no background_data().")
        return False
    try:
        background = getter()
    except Exception as exc:  # noqa: BLE001
        notes.append(
            f"Adapter's background_data() raised "
            f"({type(exc).__name__}: {exc}); treated as unavailable."
        )
        return False
    if background is None:
        return False
    try:
        return len(background) > 0
    except TypeError:
        notes.append("Adapter's background_data() returned a sizeless object.")
        return False


def _detect_batch_scoring(adapter: Any, notes: List[str]) -> bool:
    """Whether the adapter declares it can score a batch in one call.

    Absent an explicit declaration this is False. That default matters: a
    remotely-served model scored one row per network call makes perturbation
    methods (which draw thousands of samples per explained row) operationally
    impossible, so assuming batch support would turn an honest "unavailable"
    into an unbounded request storm.
    """
    capabilities = getattr(adapter, "capabilities", None)
    if not isinstance(capabilities, dict):
        return False
    return bool(capabilities.get("batch_scoring", False))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_LIMIT_NO_PROBABILITY = (
    "The model does not expose probability predictions. Both SHAP and LIME "
    "explain P(class == 1) here, so neither can be produced from hard labels "
    "alone."
)
_LIMIT_NO_BACKGROUND = (
    "No reference/background dataset is available for this model. SHAP needs "
    "a reference distribution to attribute against and LIME needs a sampling "
    "distribution to perturb within; without one, any number produced would "
    "describe nothing."
)
_LIMIT_NO_ARTIFACT_NO_BATCH = (
    "The model has no local artifact and does not declare batch scoring, so "
    "it can only be reached one prediction at a time. Perturbation-based "
    "explanation draws thousands of predictions per explained row, which is "
    "not feasible against a single-row interface. Enable batch scoring on the "
    "adapter to make approximate explanation available."
)
_LIMIT_APPROXIMATE = (
    "Contributions are a sampled approximation, not an exact decomposition of "
    "the model. They must not be described as model-internal SHAP."
)
_LIMIT_BLACK_BOX = (
    "The explanation is computed through the model's prediction interface "
    "only; the model's internals were never inspected. Quality depends "
    "entirely on the supplied reference dataset."
)
_LIMIT_TREE_SCALE = (
    "Contributions are on the probability scale (additive to predict_proba), "
    "NOT the log-odds scale produced for linear structures. Values from the "
    "two scales must never be compared, combined, or shown on a shared axis."
)
_LIMIT_SURROGATE = (
    "LIME contributions are weights of a local surrogate fitted around each "
    "row, not an attribution of the model itself. They are not comparable in "
    "magnitude to SHAP contributions and their feature ranking may diverge."
)
_LIMIT_UNKNOWN_METHOD = (
    "Unsupported explainability method. Supported methods are 'shap' and "
    "'lime'."
)


def select_explainer(capability: ExplainerCapability, method: str) -> ExplainerDecision:
    """Decide how ``method`` should be executed for a model with ``capability``.

    Pure function of the capability record -- no adapter access, no I/O, no
    explainer execution. Always returns a decision; ``available=False`` is a
    normal outcome carrying its reason, never an exception or ``None``.

    The gates are applied in order of how fundamental they are, so the
    limitation a caller is shown is the most basic unmet requirement rather
    than an incidental one.
    """
    method_lower = str(method).lower()
    if method_lower not in SUPPORTED_METHODS:
        return ExplainerDecision(
            method=method_lower,
            available=False,
            limitations=(_LIMIT_UNKNOWN_METHOD,),
        )

    # --- Gates common to every method -------------------------------------
    if not capability.supports_probability:
        return _unavailable(method_lower, _LIMIT_NO_PROBABILITY)
    if not capability.has_background_data:
        return _unavailable(method_lower, _LIMIT_NO_BACKGROUND)
    if not capability.has_local_artifact and not capability.supports_batch_scoring:
        return _unavailable(method_lower, _LIMIT_NO_ARTIFACT_NO_BATCH)

    if method_lower == METHOD_LIME:
        return _select_lime(capability)
    return _select_shap(capability)


def _unavailable(method: str, *limitations: str) -> ExplainerDecision:
    return ExplainerDecision(
        method=method, available=False, limitations=tuple(limitations)
    )


def _select_shap(capability: ExplainerCapability) -> ExplainerDecision:
    """Pick the most faithful SHAP explainer the model's structure allows.

    Preference order is fidelity, not familiarity: an exact decomposition is
    chosen whenever the structure supports one, and the sampled black-box
    estimator is the fallback for everything else -- including model families
    this module has never encountered. An unrecognised estimator is explained
    approximately, not rejected.
    """
    if capability.has_local_artifact and capability.has_raw_transformed_mapping:
        if capability.has_linear_structure:
            return ExplainerDecision(
                method=METHOD_SHAP,
                available=True,
                explainer=EXPLAINER_LINEAR,
                scale=SCALE_LOG_ODDS,
                fidelity=FIDELITY_EXACT,
            )
        if capability.has_tree_structure:
            return ExplainerDecision(
                method=METHOD_SHAP,
                available=True,
                explainer=EXPLAINER_TREE,
                scale=SCALE_PROBABILITY,
                fidelity=FIDELITY_EXACT,
                limitations=(_LIMIT_TREE_SCALE,),
            )

    limitations = [_LIMIT_APPROXIMATE]
    if not capability.has_local_artifact:
        limitations.append(_LIMIT_BLACK_BOX)
    return ExplainerDecision(
        method=METHOD_SHAP,
        available=True,
        explainer=EXPLAINER_KERNEL,
        scale=SCALE_PROBABILITY,
        fidelity=FIDELITY_APPROXIMATE,
        limitations=tuple(limitations),
    )


def _select_lime(capability: ExplainerCapability) -> ExplainerDecision:
    """LIME needs only a probability callable and a sampling distribution.

    That is why it is the more model-agnostic of the two methods: it never
    inspects the model, so the artifact's structure is irrelevant to it. Both
    of its requirements are already enforced by the common gates above.
    """
    limitations = [_LIMIT_SURROGATE]
    if not capability.has_local_artifact:
        limitations.append(_LIMIT_BLACK_BOX)
    return ExplainerDecision(
        method=METHOD_LIME,
        available=True,
        explainer=EXPLAINER_LIME,
        scale=SCALE_PROBABILITY,
        fidelity=FIDELITY_SURROGATE,
        limitations=tuple(limitations),
    )
