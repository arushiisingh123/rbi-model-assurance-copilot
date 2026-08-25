"""Phase 0 stub for explainability (owner: Manas).

Real SHAP/LIME logic is Phase 1 work. This only returns a correctly
shaped fake result so other modules can be built against the agreed
interface (see docs/module-interfaces.md).
"""


def explain(model_output: dict = None) -> dict:
    """Stub: return fake SHAP-shaped explanations.

    Phase 1 will call real SHAP/LIME against a model (a dummy model is
    fine until Namitha's real model is ready).
    """
    return {
        "method": "stub",
        "per_instance": [
            {"row_index": 0, "contributions": {"income": 0.31, "age": -0.05}}
        ],
        "global_importance": {"income": 0.42, "age": 0.10},
        "is_mock": True,
    }
