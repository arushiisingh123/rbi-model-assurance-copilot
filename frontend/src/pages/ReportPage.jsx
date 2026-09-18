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
import { getReport } from "../api/report";
import { IdentityBar } from "../components/IdentityBar";
import {
  AsyncSection,
  Card,
  Field,
  StatusBadge,
  Unavailable,
} from "../components/states";
import { useModels } from "../hooks/ModelContext";
import { useApiResource } from "../hooks/useApiResource";
import { timestamp } from "../utils/format";

function CitationBlock({ citation }) {
  return (
    <div className="rounded border border-base-300 bg-base-200/50 p-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="font-medium">{citation.source}</span>
        {citation.locator && (
          <span className="font-mono text-base-content/60">{citation.locator}</span>
        )}
        {citation.is_excerpt && (
          <span className="badge badge-ghost border-base-300 badge-sm">excerpt</span>
        )}
        {citation.is_current === false && (
          <span
            className="badge badge-warning badge-sm"
            title="The backend flags this source as not current."
          >
            not current
          </span>
        )}
        {citation.document_type && (
          <span className="text-base-content/50">{citation.document_type}</span>
        )}
      </div>
      {citation.quote && (
        <blockquote className="mt-2 border-l-2 border-base-300 pl-3 text-sm italic text-base-content/75">
          {citation.quote}
        </blockquote>
      )}
      <div className="mt-2 flex flex-wrap gap-4 text-xs text-base-content/50">
        {citation.publication_date && <span>published {citation.publication_date}</span>}
        {citation.provenance && <span>provenance: {citation.provenance}</span>}
        {citation.source_url && (
          <a
            className="link link-primary"
            href={citation.source_url}
            target="_blank"
            rel="noreferrer noopener"
          >
            source
          </a>
        )}
      </div>
    </div>
  );
}

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
                <CitationBlock key={index} citation={citation} />
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
          <details className="border-t border-base-300 pt-4">
            <summary className="cursor-pointer select-none text-xs uppercase tracking-wide text-base-content/50">
              Supporting evidence records ({section.supporting_evidence.length})
            </summary>
            <pre className="mt-2 text-xs bg-base-200/60 border border-base-300 rounded p-3 overflow-x-auto max-h-80">
              {JSON.stringify(section.supporting_evidence, null, 2)}
            </pre>
          </details>
        ) : null}
      </div>
    </Card>
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

  const coverage = data?.evidence_coverage;

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-6 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold">Assurance Report</h1>
          <p className="mt-1 text-sm text-base-content/60 max-w-2xl">
            The backend&apos;s three-layer report: computed findings, retrieved
            RBI evidence, and the narrative generated from them.
          </p>
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
        </div>
      </header>

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
