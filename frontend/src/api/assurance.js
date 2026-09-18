/** Full assurance run: GET /assurance-result?model_id=... */
import { cleanParams, get } from "./client";

/**
 * One backend call runs every domain for one model under ONE
 * assurance_run_id. Assurance logic is never re-implemented in JavaScript --
 * this is purely a fetch.
 *
 * Returns: model, explainability, fairness_drift, compliance, note,
 * model_id, assurance_run_id, monitoring, monitoring_unavailable_reason.
 */
export function runAssurance({ modelId, signal } = {}) {
  return get("/assurance-result", {
    params: cleanParams({ model_id: modelId }),
    signal,
  });
}
