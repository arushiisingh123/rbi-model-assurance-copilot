/**
 * Two separate layers, deliberately never merged on screen.
 *
 *   RBI REGULATORY REQUIREMENTS  verified clauses from official RBI
 *                                instruments, each with a source URL and an
 *                                exact clause reference.
 *   TECHNICAL ASSURANCE          the rule engine's own fairness, drift,
 *                                explainability and model-metadata checks,
 *                                run against project thresholds.
 *
 * The regulatory layer is rendered FIRST and the technical layer second,
 * under headings that say which is which. The technical checks are evidence
 * ABOUT a model; they carry no RBI clause and are never described as RBI
 * requirements. Presenting a threshold check as a regulatory obligation would
 * manufacture a legal claim the platform has no basis for -- which is exactly
 * the failure this separation exists to prevent.
 *
 * Nothing is evaluated here. No rule is interpreted in JavaScript, no
 * requirement text is authored here, and no citation is invented. Where the
 * backend attached no evidence chunks to a finding, that absence is shown as
 * an absence.
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
import {
  getAssessmentContext,
  toEntityProfileParams,
} from "../config/assessmentContext";
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

  // An empty register must still render. If this section disappeared, the
  // page would show technical checks alone under a compliance heading, and a
  // reader could reasonably take those checks for the regulatory assessment
  // -- the exact conflation this page is built to prevent.
  if (!rows.length) {
    return (
      <Card
        title="RBI regulatory requirements"
        subtitle="A curated set of verified RBI requirements, each read from an official RBI instrument."
      >
        <Unavailable
          title="No RBI requirements assessed for this request"
          reason="The regulatory layer is assessed against an entity profile — entity type, NBFC layer, whether you undertake digital lending, whether an external vendor serves the model. This request declared none, and applicability is never guessed from a model or its data. Nothing below has been assessed against RBI regulation; the technical assurance checks that follow are not a substitute for it."
        />
      </Card>
    );
  }

  const applies = rows.filter((r) => r.applicability === "APPLIES").length;
  const unclear = rows.filter(
    (r) => r.applicability === "APPLICABILITY_UNCLEAR",
  ).length;
  const instruments = new Set(rows.map((r) => r.document_id)).size;

  return (
    <Card
      title="RBI regulatory requirements"
      subtitle="A curated set of verified RBI requirements, each read from an official RBI instrument and carrying its exact clause reference and source link."
    >
      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Field label="Requirements assessed" value={rows.length} />
        <Field label="RBI instruments" value={instruments} />
        <Field label="Applicable" value={applies} />
        <Field label="Applicability unclear" value={unclear} />
      </dl>

      <p className="mt-4 text-xs leading-relaxed text-base-content/70 bg-base-200/60 border border-base-300 rounded p-3">
        <strong>Coverage is partial.</strong> This is a curated set of
        requirements that have been individually verified against the official
        RBI source — not the whole of RBI regulation, and not everything that
        applies to your organisation. A requirement missing from this list has
        not been assessed; that is not evidence that it does not exist or that
        you meet it.
      </p>

      <p className="mt-3 text-xs leading-relaxed text-base-content/70 bg-base-200/60 border border-base-300 rounded p-3">
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

/**
 * The declared assessment context, shown as a first-class part of the page.
 *
 * Applicability is only as good as what was declared, so what was declared is
 * put on screen next to the result it produced. A reviewer who disagrees with
 * a row here knows immediately why a requirement resolved the way it did --
 * which is not true of a context hidden inside a request builder.
 */
function DeclaredContext({ context }) {
  if (!context) {
    return (
      <Card title="Assessment context">
        <Unavailable
          title="No assessment context declared for this model"
          reason="Applicability depends on facts about the organisation — entity type, whether it lends digitally, whether a third party serves the model. None was declared for this model, and none is inferred from the model or its data, so every requirement is reported APPLICABILITY_UNCLEAR."
        />
      </Card>
    );
  }

  return (
    <Card
      title="Assessment context — explicitly declared"
      subtitle={context.label}
    >
      <p className="text-xs leading-relaxed text-base-content/70 bg-base-200/60 border border-base-300 rounded p-3">
        {context.note} Nothing here was observed from the model, its features
        or its data. It is a <strong>demo</strong> declaration, not a statement
        about any real institution.
      </p>

      <dl className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {context.declared.map((item) => (
          <div key={item.key}>
            <dt className="text-xs uppercase tracking-wide text-base-content/50">
              {item.key}
            </dt>
            <dd className="mt-1 flex items-center gap-2">
              <span className="font-mono text-sm">{item.value}</span>
              {!item.consumed && (
                <span
                  className="badge badge-ghost border-base-300 badge-sm"
                  title="Recorded for the reader. The applicability engine does not currently read this dimension, so it affects no result below."
                >
                  not assessed
                </span>
              )}
            </dd>
          </div>
        ))}
      </dl>

      <p className="mt-4 text-xs leading-relaxed text-base-content/60">
        Dimensions marked <span className="font-mono">not assessed</span> are
        recorded for the reader only — the applicability engine does not read
        them, so they change nothing below. Dimensions this context does{" "}
        <em>not</em> state (such as NBFC layer) are sent as undeclared, not as
        a guess, which is why some requirements can still resolve to{" "}
        <span className="font-mono">APPLICABILITY_UNCLEAR</span>.
      </p>

      <p className="mt-3 text-xs leading-relaxed text-base-content/60">
        <span className="font-semibold">Scope limit worth knowing: </span>
        the entity vocabulary models NBFC layers in detail and banks only
        coarsely. Where an instrument's scope for banks has not been modelled,
        the requirement is reported with its stated reason and limitation —
        read those before concluding an instrument does not bind you.
      </p>
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

  // The declared assessment context for this model, if one exists. It is
  // read here and passed to the backend; it is never merged into the engine.
  const context = getAssessmentContext(selectedModelId);
  const profile = toEntityProfileParams(context);

  // The assurance run already contains this domain, stamped with the run's
  // own id, so reading it avoids a second /compliance request that would mint
  // a fresh assurance_run_id and show a different run than the dashboard.
  //
  // But GET /assurance-result takes no assessment context, so its compliance
  // slice carries no applicability assessment at all. When a context HAS been
  // declared, the run therefore cannot answer this page's question, and
  // useDomainData's documented fallback applies: fetch the domain endpoint
  // directly, with the context attached, and let SourceNotice say plainly
  // that the result is an independent computation. Showing the run's
  // context-free slice under a declared context would be worse -- it would
  // display UNCLEAR for requirements the caller had in fact scoped.
  const { data, error, loading, refetch, source } = useDomainData({
    fromRun: context ? null : assurance?.compliance,
    fetcher: ({ signal }) =>
      getCompliance({ modelId: selectedModelId, profile, signal }),
    deps: [selectedModelId, JSON.stringify(profile)],
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
          Compliance — RBI requirements and technical assurance
        </h1>
        <Explainer className="mt-2 max-w-3xl">
          This page shows two separate things, and the difference matters.
          First, <strong>RBI regulatory requirements</strong>: a curated set of
          requirements verified against official RBI instruments, each shown
          with whether it applies to your organisation and whether the evidence
          to satisfy it is available. Second,{" "}
          <strong>technical assurance</strong>: fairness, drift, explainability
          and model-metadata checks run against project thresholds.
        </Explainer>
        <Explainer className="mt-2 max-w-3xl">
          The technical assurance checks are <strong>not</strong> RBI
          requirements. They cite no RBI clause and no regulator set their
          thresholds. They tell you about the model; they do not tell you what
          the law asks of you.
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

      <DeclaredContext context={context} />

      <AsyncSection
        loading={loading}
        error={error}
        data={data}
        onRetry={refetch}
        context="Compliance"
        loadingLabel="Assessing RBI requirements and running technical checks…"
      >
        {data && (
          <>
            <VerifiedRequirements requirements={data.verified_requirements} />

            <Card
              title="Technical assurance summary"
              subtitle="Fairness, drift, explainability and model-metadata checks. Not RBI requirements."
            >
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
                <Field label="Checks evaluated" value={findings.length} />
                <Field label="Check type" value="technical assurance" />
                <Field label="Model" value={data.model_id} mono />
              </dl>
              <p className="mt-4 text-sm leading-relaxed text-base-content/70 bg-base-200/60 border border-base-300 rounded p-3">
                The measurements behind each status are real — computed by the
                fairness, drift and explainability modules from this model.
                The <em>thresholds</em> that turn a measurement into a status
                are project and industry conventions, not RBI requirements. No
                check below cites an RBI clause, and a{" "}
                <span className="font-mono text-xs">PASS</span> here says
                nothing about regulatory compliance. The verified regulatory
                requirements are in the section above.
              </p>
            </Card>

            <Card
              title="Technical assurance checks"
              subtitle="Expand a check to see its technical reference and any retrieved evidence. These are not RBI requirements."
            >
              {findings.length ? (
                <div className="space-y-2">
                  {findings.map((finding) => (
                    <FindingRow key={finding.rule_id} finding={finding} />
                  ))}
                </div>
              ) : (
                <Unavailable
                  title="No technical assurance checks returned"
                  reason="The rule engine produced no findings for this model."
                />
              )}
            </Card>

          </>
        )}
      </AsyncSection>
    </div>
  );
}
