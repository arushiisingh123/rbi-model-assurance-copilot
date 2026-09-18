/** Assurance report: GET /report?model_id= */
import { cleanParams, get } from "./client";

/**
 * The LLM-narrated, evidence-grounded report.
 *
 * Report generation lives entirely in the backend; this only fetches it. The
 * backend never returns 500 here -- without an LLM key it returns a clearly
 * disclaimered fallback, and the UI surfaces those disclaimers rather than
 * hiding them.
 */
export function getReport({ modelId, signal } = {}) {
  return get("/report", { params: cleanParams({ model_id: modelId }), signal });
}
