/** RBI compliance: GET /compliance?model_id= */
import { cleanParams, get } from "./client";

/**
 * Two layers in one payload: technical assurance findings, and the verified
 * RBI requirement register with its applicability.
 *
 * Findings are produced by the deterministic rule engine. The UI renders them
 * and never evaluates a rule, invents a requirement, or fabricates a citation.
 *
 * `profile` carries the caller's DECLARED assessment context (entity type,
 * digital lending, external vendor, ...). It is passed through verbatim and
 * only when supplied: the backend treats an absent dimension as undeclared
 * and reports APPLICABILITY_UNCLEAR rather than assuming a value, so sending
 * a fabricated default here would silently manufacture the applicability
 * decision the whole layer exists to make honestly. See
 * ../config/assessmentContext.js for where a declared context comes from.
 */
export function getCompliance({ modelId, profile, signal } = {}) {
  return get("/compliance", {
    params: cleanParams({ model_id: modelId, ...(profile || {}) }),
    signal,
  });
}
