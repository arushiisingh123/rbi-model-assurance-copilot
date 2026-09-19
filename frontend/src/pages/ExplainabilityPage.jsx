/**
 * Explainability: global importance, per-instance attribution, SHAP and LIME.
 *
 * EVERY LABEL COMES FROM THE RESPONSE. The explainer name, the scale, the
 * fidelity and the limitations are read from the payload, never derived from
 * the method or the model. That matters concretely: linear SHAP is log-odds
 * while tree and kernel SHAP are probability, so a UI that inferred the unit
 * from `method === "shap"` would mislabel two of the three models this
 * platform supports — with numbers that look completely normal.
 */
import { useState } from "react";

import { EXPLAIN_METHODS, getExplainability } from "../api/explainability";
import { ContributionChart } from "../components/charts";
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
import { importanceRows, num, scaleLabel } from "../utils/format";

function MethodTabs({ method, onChange, disabled }) {
  return (
    <div role="tablist" className="tabs tabs-boxed tabs-sm bg-base-200">
      {EXPLAIN_METHODS.map((candidate) => (
        <button
          key={candidate}
          type="button"
          role="tab"
          className={`tab ${method === candidate ? "tab-active" : ""}`}
          onClick={() => onChange(candidate)}
          disabled={disabled}
        >
          {candidate.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

/** The reading guide. Purely backend-reported values. */
function ExplanationMeta({ explanation }) {
  return (
    <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
      <Field label="Explainer" value={explanation.explainer} />
      <Field label="Scale" value={explanation.scale ? scaleLabel(explanation.scale) : null} />
      <Field label="Fidelity" value={explanation.fidelity} />
      <Field label="Method" value={explanation.method?.toUpperCase()} />
      <Field
        label="Features explained"
        value={explanation.feature_space?.length ?? Object.keys(explanation.global_importance || {}).length}
      />
    </dl>
  );
}

function Limitations({ limitations }) {
  if (!limitations?.length) return null;
  return (
    <details className="mt-4 rounded-md border border-warning/30 bg-warning/5">
      <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium text-warning-content/90">
        {limitations.length} limitation{limitations.length === 1 ? "" : "s"} reported
        by the backend — read before citing these numbers
      </summary>
      <ul className="px-4 pb-4 pt-1 space-y-2">
        {limitations.map((limitation, index) => (
          <li key={index} className="text-sm leading-relaxed text-base-content/75">
            • {limitation}
          </li>
        ))}
      </ul>
    </details>
  );
}

export function ExplainabilityPage() {
  const { selectedModelId, selectedModel } = useModels();
  const { result: assurance, start } = useAssurance();
  const [method, setMethod] = useState("shap");
  const [rowIndex, setRowIndex] = useState(0);

  // The assurance run computes SHAP only, so it can answer this page for
  // "shap" but not for "lime". Asking for LIME therefore falls back to the
  // standalone endpoint -- which is an independent computation, and is
  // labelled as one rather than borrowing the run's id.
  const servedByRun = method === "shap" ? assurance?.explainability : null;

  const { data, error, loading, refetch, source } = useDomainData({
    fromRun: servedByRun,
    fetcher: ({ signal }) => getExplainability({ modelId: selectedModelId, method, signal }),
    deps: [selectedModelId, method],
    enabled: Boolean(selectedModelId),
  });

  // ExplainabilityResult carries no assurance_run_id of its own. When this
  // page IS the run's explainability, the run's id is shown because these are
  // that run's findings; standalone, no id exists and none is invented.
  const runId = source === "assurance-run" ? assurance?.assurance_run_id : null;

  // Reset the selected row whenever the model or method changes, so an index
  // valid for one explanation is never applied to another.
  const perInstance = data?.per_instance || [];
  const safeRowIndex = Math.min(rowIndex, Math.max(perInstance.length - 1, 0));
  const currentRow = perInstance[safeRowIndex];

  const globalRows = importanceRows(data?.global_importance, 15);
  const instanceRows = importanceRows(currentRow?.contributions, 15);
  const unit = data?.scale ? scaleLabel(data.scale) : "unlabelled";

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-6 flex-wrap">
        <div className="max-w-2xl">
          <h1 className="text-2xl font-semibold">
            Explainability — why the model decided
          </h1>
          <Explainer className="mt-2">
            This page shows which pieces of information the model leaned on,
            and how strongly each one pushed its answer one way or the other.
            It describes what the model <em>did</em> — it does not judge
            whether the model was right.
          </Explainer>
        </div>
        <MethodTabs method={method} onChange={setMethod} disabled={loading} />
      </header>

      <IdentityBar
        modelId={data?.model_id ?? selectedModelId}
        modelType={data?.model_type ?? selectedModel?.model_type}
        modelVersion={selectedModel?.model_version}
        integrationType={data?.integration_type ?? selectedModel?.integration_type}
        runId={runId}
      />

      <SourceNotice
        source={source}
        runId={runId}
        domain={`${method.toUpperCase()} explainability`}
        onRunAssurance={start}
      />

      {method === "lime" && assurance && (
        <p className="text-xs text-base-content/50">
          The assurance run computes SHAP only, so this LIME result is a
          separate computation and carries no assurance run identifier.
        </p>
      )}

      <AsyncSection
        loading={loading}
        error={error}
        data={data}
        onRetry={refetch}
        context="Explainability"
        loadingLabel={`Computing ${method.toUpperCase()} attributions…`}
      >
        {data && data.available === false ? (
          <Card title="Explanation" right={<StatusBadge status="PENDING" />}>
            <Unavailable
              title={`${method.toUpperCase()} is not available for this model`}
              reason={
                data.limitations?.length
                  ? data.limitations.join(" ")
                  : "The backend reported this method as unavailable and gave no further detail."
              }
            />
          </Card>
        ) : (
          data && (
            <>
              <Card
                title="How to read these charts"
                right={<StatusBadge status={data.available ? "PASS" : "PENDING"} />}
              >
                <div className="rounded-md border border-base-300 bg-base-200/40 p-4">
                  <p className="text-sm font-medium">Reading a bar</p>
                  <ul className="mt-2 space-y-1.5 text-sm text-base-content/75 leading-snug">
                    <li className="flex gap-2">
                      <span
                        aria-hidden="true"
                        className="mt-1 inline-block w-3 h-3 rounded-sm shrink-0"
                        style={{ background: "#b91c1c" }}
                      />
                      <span>
                        A bar in this colour means that piece of information
                        pushed the model <strong>towards</strong> the
                        higher-risk answer for that case.
                      </span>
                    </li>
                    <li className="flex gap-2">
                      <span
                        aria-hidden="true"
                        className="mt-1 inline-block w-3 h-3 rounded-sm shrink-0"
                        style={{ background: "#0369a1" }}
                      />
                      <span>
                        A bar in this colour means it pushed the model{" "}
                        <strong>away</strong> from the higher-risk answer.
                      </span>
                    </li>
                    <li className="flex gap-2">
                      <span aria-hidden="true" className="text-base-content/40">
                        ↔
                      </span>
                      <span>
                        The longer the bar, the more weight that information
                        carried for that case.
                      </span>
                    </li>
                  </ul>
                  <p className="mt-3 text-sm leading-relaxed text-base-content/70">
                    A case showing strong higher-risk contributions is the
                    model&apos;s estimate, not a decision and not a finding. On
                    its own it does <strong>not</strong> establish that the
                    customer did anything wrong, that a member of staff acted
                    improperly, that a regulation was breached, or that the
                    model is faulty. To interpret one, check whether the
                    underlying values are correct, whether the case is unusual
                    compared with others, and whether the score sits close to
                    the cut-off used for decisions.
                  </p>
                </div>

                <div className="mt-5">
                  <p className="text-xs font-semibold uppercase tracking-wide text-base-content/50">
                    Technical detail
                  </p>
                  <div className="mt-2">
                    <ExplanationMeta explanation={data} />
                  </div>
                  <p className="mt-3 text-sm leading-relaxed text-base-content/70">
                    Contributions are on the <strong>{unit}</strong> scale with{" "}
                    <strong>{data.fidelity}</strong> fidelity, produced by{" "}
                    <strong>{data.explainer}</strong>. Values on different
                    scales are not comparable across models — an exact log-odds
                    attribution and an approximate probability attribution
                    answer different questions.
                  </p>
                </div>
                <Limitations limitations={data.limitations} />
              </Card>

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <Card
                  title="Global feature importance"
                  subtitle={`Top ${globalRows.length} features by mean absolute contribution.`}
                >
                  {globalRows.length ? (
                    <ContributionChart rows={globalRows} unit={unit} signed={false} />
                  ) : (
                    <Unavailable
                      title="No global importance reported"
                      reason="The backend returned an empty global_importance map for this explanation."
                    />
                  )}
                </Card>

                <Card
                  title="Per-instance attribution"
                  subtitle={
                    perInstance.length
                      ? `Record ${safeRowIndex + 1} of ${perInstance.length} explained.`
                      : "No per-instance rows in this explanation."
                  }
                  right={
                    perInstance.length > 1 && (
                      <select
                        className="select select-bordered select-xs"
                        value={safeRowIndex}
                        onChange={(event) => setRowIndex(Number(event.target.value))}
                      >
                        {perInstance.map((row, index) => (
                          <option key={row.row_index ?? index} value={index}>
                            row {row.row_index ?? index}
                          </option>
                        ))}
                      </select>
                    )
                  }
                >
                  {instanceRows.length ? (
                    <>
                      <ContributionChart rows={instanceRows} unit={unit} signed />
                      <p className="mt-2 text-xs text-base-content/50">
                        Rows are identified by position within the explained
                        frame (<span className="font-mono">row_index</span>).
                        The backend does not attach a stable instance id to
                        explanation rows, so this is a position, not an
                        applicant identity.
                      </p>
                    </>
                  ) : (
                    <Unavailable
                      title="No per-instance contributions"
                      reason="This explanation contains no per_instance rows."
                    />
                  )}
                </Card>
              </div>

              <Card title="Global importance values">
                <div className="overflow-x-auto max-h-96">
                  <table className="table table-xs table-pin-rows">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th className="text-right">
                          Importance ({unit})
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {importanceRows(data.global_importance, 200).map((row) => (
                        <tr key={row.feature}>
                          <td className="font-mono text-xs">{row.feature}</td>
                          <td className="text-right font-mono text-xs">
                            {num(row.value, 6)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          )
        )}
      </AsyncSection>
    </div>
  );
}
