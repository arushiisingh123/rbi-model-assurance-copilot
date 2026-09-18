/** RBI compliance: GET /compliance?model_id= */
import { cleanParams, get } from "./client";

/**
 * Rule findings with their evidence chunk ids, plus run identity.
 *
 * Findings are produced by the deterministic rule engine. The UI renders them
 * and never evaluates a rule, invents a requirement, or fabricates a citation.
 */
export function getCompliance({ modelId, signal } = {}) {
  return get("/compliance", {
    params: cleanParams({ model_id: modelId }),
    signal,
  });
}
