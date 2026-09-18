/** Explainability: GET /explainability?method=&model_id= */
import { cleanParams, get } from "./client";

export const EXPLAIN_METHODS = ["shap", "lime"];

/**
 * The response carries its OWN explainer/scale/fidelity/limitations. The UI
 * must read them rather than deriving anything from the method name:
 * linear SHAP is log-odds while tree and kernel SHAP are probability, so
 * inferring a scale from `method === "shap"` mislabels two of the three.
 */
export function getExplainability({ modelId, method = "shap", signal } = {}) {
  return get("/explainability", {
    params: cleanParams({ model_id: modelId, method }),
    signal,
  });
}
