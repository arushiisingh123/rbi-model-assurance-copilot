/**
 * The assurance report: three layers per section, exactly as the backend
 * produced them.
 *
 *   Layer 1  technical_finding    deterministic Python result
 *   Layer 2  retrieved_evidence   RBI citations, or NOT_FOUND
 *   Layer 3  llm_interpretation   narrative, grounded in the two above
 *
 * The layers are rendered SEPARATELY and labelled, because that separation is
 * the report's whole safety property: a reader must be able to see which
 * sentence is a computed fact, which is a quoted regulation, and which is
 * model-generated prose. Flattening them into one paragraph would make
 * narrative indistinguishable from evidence.
 *
 * No report content is generated here. The download button serialises exactly
 * what the backend returned.
 */
import { useState } from "react";
import { getReport, getReportPdf } from "../api/report";
import { IdentityBar } from "../components/IdentityBar";
import { SourceCitation } from "../components/SourceCitation";
import {
  AsyncSection,
  Card,
  ErrorState,
  Explainer,
  Field,
  StatusBadge,
  Unavailable,
} from "../components/states";
import { useModels } from "../hooks/ModelContext";
import { useApiResource } from "../hooks/useApiResource";
import { timestamp } from "../utils/format";


function ReportSectionCard({ section }) {
  const finding = section.technical_finding;
  const retrieved = section.retrieved_evidence;
  const interpretation = section.llm_interpretation;
  const notFound = retrieved?.evidence_status === "NOT_FOUND";

  return (
    <Card
      title={section.heading}
      right={<StatusBadge status={finding?.status} />}
    >
      <div className="space-y-5">
        {/* Layer 1 */}
        <div>
          <h4 className="text-xs uppercase tracking-wide text-base-content/50 mb-2">
            Layer 1 · technical finding (computed)
          </h4>
          {finding ? (
            <div className="space-y-2">
              <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Field label="Reference" value={finding.ref} mono />
                <Field label="Source module" value={finding.source_module} mono />
                <Field label="Provenance" value={finding.provenance} />
                <Field label="Model" value={finding.model_id} mono />
              </dl>
              <pre className="text-xs bg-base-200/60 border border-base-300 rounded p-3 overflow-x-auto">
                {JSON.stringify(finding.value, null, 2)}
              </pre>
            </div>
          ) : (
            <Unavailable
              title="No technical finding"
              reason="This section carried no Layer 1 finding."
            />
          )}
        </div>

        {/* Layer 2 */}
        <div className="border-t border-base-300 pt-4">
          <h4 className="text-xs uppercase tracking-wide text-base-content/50 mb-2">
            Layer 2 · retrieved RBI evidence
            <span className="ml-2">
              <StatusBadge status={retrieved?.evidence_status} size="sm" />
            </span>
          </h4>
          {notFound ? (
            <Unavailable
              title="No regulatory evidence retrieved for this section"
              reason="Retrieval returned NOT_FOUND. This means no verified evidence was found in the indexed RBI corpus — it does NOT mean no RBI rule exists. The narrative below is therefore not grounded in regulatory text and must not be read as a regulatory claim."
            />
          ) : retrieved?.citations?.length ? (
            <div className="space-y-2">
              {retrieved.citations.map((citation, index) => (
                <SourceCitation key={index} citation={citation} />
              ))}
            </div>
          ) : (
            <Unavailable
              title="No citations attached"
              reason="The section reports no citations."
            />
          )}
        </div>

        {/* Layer 3 */}
        <div className="border-t border-base-300 pt-4">
          <h4 className="text-xs uppercase tracking-wide text-base-content/50 mb-2">
            Layer 3 · generated narrative
            {interpretation?.regulatory_basis && (
              <span className="ml-2 badge badge-ghost border-base-300 badge-sm">
                regulatory basis: {interpretation.regulatory_basis}
              </span>
            )}
          </h4>
          {interpretation ? (
            <>
              <p className="text-sm leading-relaxed text-base-content/80">
                {interpretation.text}
              </p>
              <div className="mt-2 flex flex-wrap gap-4 text-xs text-base-content/50">
                {interpretation.grounded_in?.length ? (
                  <span>grounded in: {interpretation.grounded_in.join(", ")}</span>
                ) : (
                  <span>grounded in: nothing declared</span>
                )}
                {interpretation.is_mock && (
                  <span className="text-warning">generated from mock fallback</span>
                )}
              </div>
            </>
          ) : (
            <Unavailable
              title="No narrative generated"
              reason="This section carried no LLM interpretation."
            />
          )}
          <p className="mt-3 text-xs italic text-base-content/50">
            This text is generated from the two layers above. It does not
            compute any metric and does not establish any regulatory
            requirement.
          </p>
        </div>

        {section.supporting_evidence?.length ? (
          <SupportingEvidence records={section.supporting_evidence} />
        ) : null}
      </div>
    </Card>
  );
}

/**
 * Traceability records carried alongside a section.
 *
 * These are the platform's own evidence records (fairness groups, importance
 * values, verified RBI requirements), not retrieved regulatory text. They were
 * previously dumped as raw JSON, which is unreadable for the people this page
 * is for. Rendered as a table instead, with the full record still reachable
 * for anyone who needs it.
 *
 * A "rbi_verified_requirement" record is a clause from the register with an
 * applicability and an evidence status -- never a pass.
 */
function SupportingEvidence({ records }) {
  const byType = records.reduce((acc, record) => {
    const type = record.evidence_type || "unknown";
    acc[type] = (acc[type] || 0) + 1;
    return acc;
  }, {});

  return (
    <details className="border-t border-base-300 pt-4">
      <summary className="cursor-pointer select-none text-xs uppercase tracking-wide text-base-content/50">
        Traceability records ({records.length})
      </summary>

      <p className="mt-2 text-xs text-base-content/60">
        Produced by this platform for traceability. They are not retrieved
        regulatory text, and none of them is a statement of compliance.
      </p>

      <div className="mt-2 flex flex-wrap gap-2">
        {Object.entries(byType).map(([type, count]) => (
          <span
            key={type}
            className="badge badge-ghost border-base-300 badge-sm font-mono"
          >
            {type} · {count}
          </span>
        ))}
      </div>

      <div className="mt-3 overflow-x-auto max-h-80">
        <table className="table table-xs">
          <thead>
            <tr>
              <th>Type</th>
              <th>Reference</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {records.slice(0, 50).map((record, index) => (
              <tr key={index}>
                <td className="font-mono text-xs">{record.evidence_type}</td>
                <td className="font-mono text-xs">
                  {record.requirement_id || record.clause || record.feature ||
                    record.group || "—"}
                </td>
                <td className="text-xs">
                  {record.requirement || record.status || record.reason || "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {records.length > 50 && (
          <p className="mt-2 text-xs text-base-content/50">
            Showing the first 50 of {records.length}. The full set is in the
            downloaded report.
          </p>
        )}
      </div>
    </details>
  );
}

export function ReportPage() {
  const { selectedModelId, selectedModel } = useModels();

  const { data, error, loading, refetch } = useApiResource(
    ({ signal }) => getReport({ modelId: selectedModelId, signal }),
    [selectedModelId],
    { enabled: Boolean(selectedModelId) },
  );

  function download() {
    if (!data) return;
    // Serialises exactly what the backend returned -- no client-side
    // re-authoring of report content.
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `assurance-report-${data.report_id || "export"}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState(null);

  async function downloadPdf() {
    setPdfError(null);
    setPdfLoading(true);
    try {
      // A DIFFERENT render of the backend's computed results (technical
      // checks, explainability/fairness/drift, verified RBI requirements)
      // via GET /report/pdf -- not this page's LLM-narrated JSON. Its own
      // filename (from Content-Disposition) carries that PDF's own
      // assurance_run_id, which is why this ignores `data.report_id`.
      const { blob, filename } = await getReportPdf({ modelId: selectedModelId });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setPdfError(err);
    } finally {
      setPdfLoading(false);
    }
  }

  const coverage = data?.evidence_coverage;

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-6 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold">
            Assurance Report — the written summary
          </h1>
          <Explainer className="mt-2 max-w-3xl">
            A written summary of the assessment, built in three separate
            layers: what was measured, which verified regulatory sources were
            found, and a written explanation of the two. The written
            explanation can never change a result or decide whether something
            passed.
          </Explainer>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            className="btn btn-sm btn-outline"
            onClick={refetch}
            disabled={loading}
          >
            Regenerate
          </button>
          <button
            type="button"
            className="btn btn-sm btn-primary"
            onClick={download}
            disabled={!data}
          >
            Download report (JSON)
          </button>
          <button
            type="button"
            className="btn btn-sm btn-primary"
            onClick={downloadPdf}
            disabled={!selectedModelId || pdfLoading}
          >
            {pdfLoading ? "Generating PDF…" : "Download PDF"}
          </button>
        </div>
      </header>

      {pdfError && (
        <ErrorState error={pdfError} context="PDF download" onRetry={downloadPdf} />
      )}

      <IdentityBar
        modelId={data?.model_id ?? selectedModelId}
        modelType={selectedModel?.model_type}
        modelVersion={data?.model_version ?? selectedModel?.model_version}
        integrationType={selectedModel?.integration_type}
        runId={data?.assurance_run_id}
      />

      <AsyncSection
        loading={loading}
        error={error}
        data={data}
        onRetry={refetch}
        context="Report"
        loadingLabel="Generating report (retrieval + narrative)…"
      >
        {data && (
          <>
            <Card title="Report metadata">
              <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                <Field label="Report ID" value={data.report_id} mono />
                <Field label="Generated" value={timestamp(data.generated_at)} />
                <Field label="Model version" value={data.model_version} />
                <Field label="Sections" value={data.sections?.length ?? 0} />
                <Field
                  label="Evidence coverage"
                  value={
                    coverage
                      ? `${coverage.retrieved} retrieved / ${coverage.not_found} not found`
                      : null
                  }
                />
              </dl>

              {data.is_mock && (
                <div className="mt-4 rounded border border-warning/40 bg-warning/10 p-4">
                  <p className="text-sm font-semibold text-warning-content/90">
                    This is a fallback mock report
                  </p>
                  <p className="mt-1 text-sm text-base-content/75">
                    The backend marks it{" "}
                    <span className="font-mono text-xs">is_mock: true</span>,
                    meaning live generation was unavailable (typically no LLM
                    API key configured). The content is a static fixture and
                    must not be read as an assessment of this model.
                  </p>
                </div>
              )}

              {data.disclaimers?.length ? (
                <div className="mt-4">
                  <h3 className="text-xs uppercase tracking-wide text-base-content/50 mb-2">
                    Disclaimers and limitations
                  </h3>
                  <ul className="space-y-1.5">
                    {data.disclaimers.map((disclaimer, index) => (
                      <li key={index} className="text-sm text-base-content/70">
                        • {disclaimer}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </Card>

            {data.sections?.length ? (
              data.sections.map((section, index) => (
                <ReportSectionCard key={section.heading || index} section={section} />
              ))
            ) : (
              <Card title="Report sections">
                <Unavailable
                  title="No sections in this report"
                  reason="The backend returned a report with no sections."
                />
              </Card>
            )}
          </>
        )}
      </AsyncSection>
    </div>
  );
}
