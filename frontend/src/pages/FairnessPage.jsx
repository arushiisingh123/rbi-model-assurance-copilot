/**
 * Fairness (plus the feature-drift result returned alongside it).
 *
 * THE RULE THIS PAGE ENFORCES: a model that declares no protected attribute
 * comes back as protected_attribute "none_declared" with status PENDING, and
 * that is displayed exactly as it is. It is never upgraded to PASS, and no
 * attribute is ever invented to produce a number — a fabricated fairness
 * result is a fabricated regulatory finding.
 */
import { getFairnessDrift } from "../api/fairness";
import { DriftChart, GroupRateChart } from "../components/charts";
import { IdentityBar } from "../components/IdentityBar";
import {
  AsyncSection,
  Card,
  Field,
  StatusBadge,
  Unavailable,
} from "../components/states";
import { useAssurance } from "../hooks/AssuranceContext";
import { useModels } from "../hooks/ModelContext";
import { useDomainData } from "../hooks/useDomainData";
import { SourceNotice } from "../components/SourceNotice";
import { num, pct } from "../utils/format";

const NONE_DECLARED = "none_declared";

function FairnessBody({ fairness }) {
  const undeclared = fairness.protected_attribute === NONE_DECLARED;

  if (undeclared) {
    return (
      <Unavailable
        title="No protected attribute is declared for this model"
        reason={
          "The model's adapter declares no protected attribute, so the fairness module has nothing to group applicants by. " +
          `It therefore reports status ${fairness.status} rather than computing a metric. Choosing an attribute here would fabricate a regulatory finding.`
        }
        details={
          <dl className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-4">
            <Field label="Protected attribute" value={fairness.protected_attribute} mono />
            <Field label="Reported status" value={<StatusBadge status={fairness.status} size="sm" />} />
            <Field label="Groups evaluated" value={fairness.groups?.length ?? 0} />
          </dl>
        }
      />
    );
  }

  return (
    <div className="space-y-6">
      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Field label="Protected attribute" value={fairness.protected_attribute} mono />
        <Field
          label="Demographic parity difference"
          value={num(fairness.demographic_parity_diff)}
        />
        <Field
          label="Disparate impact ratio"
          value={num(fairness.disparate_impact_ratio)}
        />
        <Field label="Groups" value={fairness.groups?.length ?? 0} />
      </dl>

      {fairness.groups?.length ? (
        <>
          <GroupRateChart groups={fairness.groups} />
          <div className="overflow-x-auto">
            <table className="table table-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wide">
                  <th>Group</th>
                  <th className="text-right">Records</th>
                  <th className="text-right">Favourable outcomes</th>
                  <th className="text-right">Selection rate</th>
                </tr>
              </thead>
              <tbody>
                {fairness.groups.map((group) => (
                  <tr key={group.group}>
                    <td className="font-mono text-xs">{group.group}</td>
                    <td className="text-right">{group.count}</td>
                    <td className="text-right">{group.favorable_count}</td>
                    <td className="text-right font-mono text-xs">
                      {pct(group.selection_rate)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <Unavailable
          title="No per-group breakdown reported"
          reason="The fairness module returned no group rows for this model."
        />
      )}

      <p className="text-xs italic text-base-content/50">
        The thresholds behind this status are project/industry conventions
        maintained by the analytics modules. They are not themselves RBI
        requirements and must not be presented as such.
      </p>
    </div>
  );
}

export function FairnessPage() {
  const { selectedModelId, selectedModel } = useModels();
  const { result: assurance, start } = useAssurance();

  // fairness_drift is part of the assurance result, so a run answers this
  // page without a second request. NOTE: FairnessDriftResult has no
  // assurance_run_id field of its own -- when this page is served from a run,
  // the run's id is shown because these ARE that run's findings; standalone,
  // the endpoint carries none and the field stays "not provided".
  const { data, error, loading, refetch, source } = useDomainData({
    fromRun: assurance?.fairness_drift,
    fetcher: ({ signal }) => getFairnessDrift({ modelId: selectedModelId, signal }),
    deps: [selectedModelId],
    enabled: Boolean(selectedModelId),
  });
  const runId = source === "assurance-run" ? assurance?.assurance_run_id : null;

  const fairness = data?.fairness;
  const drift = data?.drift;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Fairness</h1>
        <p className="mt-1 text-sm text-base-content/60 max-w-2xl">
          Group fairness metrics for the selected model, with the
          feature-drift result the backend returns alongside them.
        </p>
      </header>

      <IdentityBar
        modelId={selectedModelId}
        modelType={selectedModel?.model_type}
        modelVersion={selectedModel?.model_version}
        integrationType={selectedModel?.integration_type}
        runId={runId}
      />

      <SourceNotice
        source={source}
        runId={runId}
        domain="fairness"
        onRunAssurance={start}
      />

      <AsyncSection
        loading={loading}
        error={error}
        data={data}
        onRetry={refetch}
        context="Fairness"
        loadingLabel="Computing fairness and drift…"
      >
        {data && (
          <>
            <Card
              title="Fairness assessment"
              right={<StatusBadge status={fairness?.status} size="lg" />}
            >
              {fairness ? (
                <FairnessBody fairness={fairness} />
              ) : (
                <Unavailable
                  title="No fairness result returned"
                  reason="The backend response contained no fairness block."
                />
              )}
            </Card>

            <Card
              title="Feature drift (reference vs current)"
              subtitle="Input-distribution drift. This is not prediction drift — see the Monitoring page for that."
              right={<StatusBadge status={drift?.status} size="lg" />}
            >
              {drift ? (
                <div className="space-y-5">
                  <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <Field label="Max PSI" value={num(drift.psi)} />
                    <Field label="Max KS statistic" value={num(drift.ks_statistic)} />
                    <Field
                      label="Features evaluated"
                      value={drift.features_evaluated?.length ?? 0}
                    />
                    <Field
                      label="Real computation"
                      value={drift.is_mock ? "mock" : "yes (is_mock false)"}
                    />
                  </dl>

                  {drift.note && (
                    <p className="text-sm text-base-content/70 bg-base-200/60 border border-base-300 rounded p-3">
                      {drift.note}
                    </p>
                  )}

                  {drift.per_feature?.length ? (
                    <DriftChart perFeature={drift.per_feature} />
                  ) : (
                    <Unavailable
                      title="No per-feature drift detail"
                      reason={
                        drift.status === "PENDING"
                          ? "Nothing could be evaluated, so no per-feature metrics exist. Absent data is not evidence of stability."
                          : "The backend returned no per-feature breakdown for this result."
                      }
                    />
                  )}
                </div>
              ) : (
                <Unavailable
                  title="No drift result returned"
                  reason="The backend response contained no drift block."
                />
              )}
            </Card>
          </>
        )}
      </AsyncSection>
    </div>
  );
}
