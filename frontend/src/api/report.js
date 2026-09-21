/** Assurance report: GET /report?model_id= and GET /report/pdf?model_id= */
import { client, cleanParams, get } from "./client";

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

/**
 * The same assurance data as GET /assurance-result, rendered as a PDF --
 * a second, additive presentation of the backend's own computed results, not
 * this page's LLM-narrated JSON report. Fetched as a blob (not JSON), so this
 * bypasses the shared `get()` helper, which assumes a JSON body.
 *
 * Returns the raw blob plus the filename the backend named it (carrying the
 * PDF's own assurance_run_id, read from Content-Disposition), so the caller
 * never has to re-derive a filename from unrelated page state.
 */
export async function getReportPdf({ modelId, signal } = {}) {
  const response = await client.get("/report/pdf", {
    params: cleanParams({ model_id: modelId }),
    responseType: "blob",
    signal,
  });
  const disposition = response.headers?.["content-disposition"] || "";
  const match = disposition.match(/filename="?([^";]+)"?/i);
  const filename = match ? match[1] : "assurance-report.pdf";
  return { blob: response.data, filename };
}
