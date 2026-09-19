/**
 * RBI compliance: rule findings from the deterministic rule engine.
 *
 * Findings are rendered, never evaluated here. No rule is interpreted in
 * JavaScript, no requirement text is authored here, and no citation is
 * invented. Where the backend attached no evidence chunks to a finding, that
 * absence is shown as an absence.
 */
import { getCompliance } from "../api/compliance";
import { IdentityBar } from "../components/IdentityBar";
import {
  AsyncSection,
  Card,
  Explainer,
  Field,
  StatusBadge,
  Unavailable,
} from "../components/states";
import { useAssurance } from "../hooks/AssuranceContext";
import { useModels } from "../hooks/ModelContext";
import { useDomainData } from "../hooks/useDomainData";
import { SourceNotice } from "../components/SourceNotice";
import {
  DataHandlingNotice,
  RegulatoryGroundingNotice,
} from "../components/ScopeNotice";

/**
 * The verified RBI requirement register.
 *
 * A DIFFERENT layer from the rule findings above: these are clauses read from
 * official RBI instruments, each carrying its own clause reference and source
 * link. They are reported with their applicability and evidence status and
 * NEVER as compliance — every one of them requires evidence only the regulated
 * entity holds, so none can be satisfied by model analytics.
 *
 * The count deliberately reads "requirements assessed", never "requirements
 * met": a reviewer must not be able to read eight rows as eight passes.
 */
function VerifiedRequirements({ requirements }) {
  const rows = requirements || [];
  if (!rows.length) return null;

  const applies = rows.filter((r) => r.applicability === "APPLIES").length;
  const unclear = rows.filter(
    (r) => r.applicability === "APPLICABILITY_UNCLEAR",
  ).length;
  const instruments = new Set(rows.map((r) => r.document_id)).size;

  return (
    <Card
      title="Verified RBI requirements"
      subtitle="Clauses read from official RBI instruments, with their applicability and evidence status."
    >
      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Field label="Requirements assessed" value={rows.length} />
        <Field label="RBI instruments" value={instruments} />
        <Field label="Applicable" value={applies} />
        <Field label="Applicability unclear" value={unclear} />
      </dl>

      <p className="mt-4 text-xs leading-relaxed text-base-content/70 bg-base-200/60 border border-base-300 rounded p-3">
        These are <strong>not</strong> compliance results. Every requirement
        here requires evidence held by the regulated entity — board committees,
        contracts, due-diligence records — so none can be satisfied by model
        analytics and none returns a pass. Applicability is decided from an
        entity profile the caller declares; where a dimension was not declared
        it stays <span className="font-mono">APPLICABILITY_UNCLEAR</span> rather
        than being assumed.
      </p>

      <div className="mt-4 space-y-2">
        {rows.map((requirement) => (
          <details
            key={requirement.requirement_id}
            className="rounded-md border border-base-300 bg-base-100"
          >
            <summary className="cursor-pointer select-none px-4 py-3 flex flex-wrap items-center gap-3">
              <span className="font-mono text-xs text-base-content/70">
                {requirement.requirement_id}
              </span>
              <span className="text-sm flex-1 min-w-[16rem]">
                {requirement.requirement}
              </span>
              <span className="badge badge-outline badge-sm">
                {requirement.applicability}
              </span>
              <span className="text-xs text-base-content/50">
                {requirement.status}
              </span>
            </summary>
            <div className="px-4 pb-4 pt-1 space-y-3 border-t border-base-300">
              <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3">
                <Field label="Instrument" value={requirement.document_title} />
                <Field label="Clause" value={requirement.clause} mono />
                <Field
                  label="Instrument type"
                  value={`${requirement.instrument_type} · ${requirement.regulatory_status}`}
                />
                <Field label="Assessment mode" value={requirement.assessment_mode} />
                <Field label="Verified on" value={requirement.verified_on} />
                <Field label="Model" value={requirement.model_id} mono />
              </dl>
              {requirement.quote && (
                <blockquote className="text-xs italic leading-relaxed text-base-content/70 border-l-2 border-base-300 pl-3">
                  “{requirement.quote}”
                </blockquote>
              )}
              {requirement.reason && (
                <p className="text-xs text-base-content/70">{requirement.reason}</p>
              )}
              {requirement.limitation && (
                <p className="text-xs text-base-content/60">
                  <span className="font-semibold">Limitation: </span>
                  {requirement.limitation}
                </p>
              )}
              <a
                className="link link-primary text-xs font-mono break-all"
                href={requirement.source_url}
                target="_blank"
                rel="noreferrer"
              >
                {requirement.source_url}
              </a>
            </div>
          </details>
        ))}
      </div>
    </Card>
  );
}

function FindingRow({ finding }) {
  const chunks = finding.evidence_chunks || [];
  return (
    <details className="rounded-md border border-base-300 bg-base-100">
      <summary className="cursor-pointer select-none px-4 py-3 flex flex-wrap items-center gap-3">
        <StatusBadge status={finding.status} size="sm" />
        <span className="font-mono text-xs text-base-content/70">{finding.rule_id}</span>
        <span className="text-sm flex-1 min-w-[16rem]">{finding.rule_description}</span>
        <span className="text-xs text-base-content/50">
          {chunks.length} evidence chunk{chunks.length === 1 ? "" : "s"}
        </span>
      </summary>
      <div className="px-4 pb-4 pt-1 space-y-4 border-t border-base-300">
        <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3">
          <Field label="Rule" value={finding.rule_id} mono />
          <Field
            label="Technical finding reference"
            value={finding.technical_finding_ref}
            mono
          />
          <Field label="Status" value={<StatusBadge status={finding.status} size="sm" />} />
          <Field label="Model" value={finding.model_id} mono />
          <Field label="Assurance run" value={finding.assurance_run_id} mono />
        </dl>

        <div>
          <h4 className="text-xs uppercase tracking-wide text-base-content/50 mb-2">
            Retrieved RBI evidence
          </h4>
          {chunks.length ? (
            <ul className="space-y-1.5">
              {chunks.map((chunk, index) => (
                <li
                  key={index}
                  className="font-mono text-xs bg-base-200/60 border border-base-300 rounded px-3 py-2 break-all"
                >
                  {chunk}
                </li>
              ))}
            </ul>
          ) : (
            <Unavailable
              title="No RBI evidence attached to this finding"
              reason="The retrieval layer returned no supporting chunk for this rule. The finding's status is still the rule engine's own deterministic evaluation — but it is not grounded in retrieved regulatory text, so it must not be cited as such."
            />
          )}
        </div>
      </div>
    </details>
  );
}

export function CompliancePage() {
  const { selectedModelId, selectedModel } = useModels();
  const { result: assurance, start } = useAssurance();

  // The assurance run already contains this domain, stamped with the run's
  // own id. Reading it avoids a second /compliance request, which would mint
  // a fresh assurance_run_id and show a different run than the dashboard.
  const { data, error, loading, refetch, source } = useDomainData({
    fromRun: assurance?.compliance,
    fetcher: ({ signal }) => getCompliance({ modelId: selectedModelId, signal }),
    deps: [selectedModelId],
    enabled: Boolean(selectedModelId),
  });

  const findings = data?.findings || [];
  const counts = findings.reduce((acc, finding) => {
    acc[finding.status] = (acc[finding.status] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">
          RBI Compliance — rule checks and requirements
        </h1>
        <Explainer className="mt-2 max-w-3xl">
          This page shows two separate things. First, a set of rule checks run
          against what the assessment measured. Second, a small number of
          verified RBI requirements, with whether each one applies to your
          organisation and whether the evidence to satisfy it is available.
        </Explainer>
        <Explainer className="mt-2 max-w-3xl">
          Neither is a statement of legal compliance. Where a requirement needs
          records only your organisation holds, the platform says so rather
          than guessing.
        </Explainer>
      </header>

      <IdentityBar
        modelId={data?.model_id ?? selectedModelId}
        modelType={selectedModel?.model_type}
        modelVersion={selectedModel?.model_version}
        integrationType={selectedModel?.integration_type}
        runId={data?.assurance_run_id}
      />

      <SourceNotice
        source={source}
        runId={data?.assurance_run_id}
        domain="compliance"
        onRunAssurance={start}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RegulatoryGroundingNotice />
        <DataHandlingNotice />
      </div>

      <AsyncSection
        loading={loading}
        error={error}
        data={data}
        onRetry={refetch}
        context="Compliance"
        loadingLabel="Evaluating RBI rules against technical findings…"
      >
        {data && (
          <>
            <Card title="Evaluation summary">
              <div className="flex flex-wrap gap-3">
                {["PASS", "WARNING", "FAIL", "PENDING"].map((status) => (
                  <div
                    key={status}
                    className="flex items-center gap-2 rounded border border-base-300 bg-base-100 px-4 py-2"
                  >
                    <StatusBadge status={status} size="sm" />
                    <span className="text-lg font-semibold">{counts[status] || 0}</span>
                  </div>
                ))}
              </div>
              <dl className="mt-5 grid grid-cols-2 sm:grid-cols-3 gap-4">
                <Field label="Rules evaluated" value={findings.length} />
                <Field
                  label="Rule set provenance"
                  value={data.is_mock ? "illustrative sample rules" : "verified rule set"}
                />
                <Field label="Model" value={data.model_id} mono />
              </dl>
              {data.is_mock && (
                <p className="mt-4 text-sm leading-relaxed text-base-content/70 bg-base-200/60 border border-base-300 rounded p-3">
                  The backend marks this rule set as illustrative
                  (<span className="font-mono text-xs">is_mock: true</span>).
                  The technical findings behind each status are real, but the
                  rule text is sample material and is not verified RBI
                  regulatory text.
                </p>
              )}
            </Card>

            <Card
              title="Rule findings"
              subtitle="Expand a finding to see its technical reference and retrieved evidence."
            >
              {findings.length ? (
                <div className="space-y-2">
                  {findings.map((finding) => (
                    <FindingRow key={finding.rule_id} finding={finding} />
                  ))}
                </div>
              ) : (
                <Unavailable
                  title="No rule findings returned"
                  reason="The rule engine produced no findings for this model."
                />
              )}
            </Card>

            <VerifiedRequirements requirements={data.verified_requirements} />
          </>
        )}
      </AsyncSection>
    </div>
  );
}
