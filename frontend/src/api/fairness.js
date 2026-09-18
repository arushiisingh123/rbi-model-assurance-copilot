/** Fairness + feature drift: GET /fairness-drift?model_id= */
import { cleanParams, get } from "./client";

/**
 * Returns { fairness, drift }.
 *
 * A model that declares no protected attribute comes back as
 * protected_attribute "none_declared" with status PENDING. That is a real,
 * honest result and is displayed verbatim -- never upgraded to PASS and
 * never given an invented attribute.
 */
export function getFairnessDrift({ modelId, signal } = {}) {
  return get("/fairness-drift", {
    params: cleanParams({ model_id: modelId }),
    signal,
  });
}

/** Cross-model drift comparability: GET /drift-comparison (default models). */
export function getDriftComparison({ signal } = {}) {
  return get("/drift-comparison", { signal });
}
