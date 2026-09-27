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
  Explainer,
  Field,
  MetricField,
  StatusBadge,
  Unavailable,
} from "../components/states";
import { useAssurance } from "../hooks/AssuranceContext";
import { useModels } from "../hooks/ModelContext";
import { useDomainData } from "../hooks/useDomainData";
import { SourceNotice } from "../components/SourceNotice";
import { num, pct } from "../utils/format";

const NONE_DECLARED = "none_declared";

function FairnessBody({ fairness, stability }) {
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
        <MetricField
          label="Protected attribute"
          term="protected attribute"
          value={fairness.protected_attribute}
          mono
        />
        <MetricField
          label="Demographic parity difference"
          term="demographic parity difference"
          value={num(fairness.demographic_parity_diff)}
        />
        <MetricField
          label="Disparate impact ratio"
          term="disparate impact ratio"
          value={num(fairness.disparate_impact_ratio)}
        />
        <MetricField
          label="Groups compared"
          help="How many distinct groups of the protected attribute were found in this data."
          value={fairness.groups?.length ?? 0}
        />
      </dl>

      <Explainer>
        The disparate impact ratio compares the group least likely to receive a
        favourable decision against the group most likely to.{" "}
        <strong>1.00 means every group is treated equally</strong>; the further
        below 1.00, the wider the gap. This model scored{" "}
        <strong>{num(fairness.disparate_impact_ratio, 2)}</strong>.
      </Explainer>

      {stability?.note && (
        <div
          className={`rounded border p-3 text-sm ${
            stability.driving_groups_small
              ? "border-warning/40 bg-warning/10"
              : "border-base-300 bg-base-200/50"
          }`}
        >
          <p className="font-semibold">
            {stability.driving_groups_small
              ? "Small sample — this ratio may be sensitive to individual records"
              : "Some groups have few records"}
          </p>
          <p className="mt-1 text-base-content/75">{stability.note}</p>

          {stability.most_favoured_group && stability.least_favoured_group && (
            <p className="mt-2 text-xs text-base-content/60">
              The ratio compares{" "}
              <span className="font-mono">{stability.least_favoured_group}</span>{" "}
              (least likely to receive a favourable decision) against{" "}
              <span className="font-mono">{stability.most_favoured_group}</span>{" "}
              (most likely). Only these two groups determine it.
            </p>
          )}

          {/* The threshold behind this notice is an unvalidated project
              convention, so the screen says so rather than implying a
              statistical standard. */}
          <p className="mt-2 text-xs text-base-content/50">
            Reporting threshold: fewer than{" "}
            <span className="font-mono">{stability.min_group_size}</span> records
            {stability.threshold_is_project_default
              ? " (project default — not configured by the team, and not validated by any project document or statistical method)"
              : " (configured for this project)"}
            . It is a reporting convention only: it changes which groups are
            flagged here, never a metric and never the PASS/FAIL status above.
          </p>
        </div>
      )}

      {fairness.groups?.length ? (
        <>
          <GroupRateChart groups={fairness.groups} />
          <div className="overflow-x-auto">
            <table className="table table-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wide">
                  <th>Group</th>
                  <th className="text-right">Cases</th>
                  <th className="text-right">Favourable decisions</th>
                  <th className="text-right" title="The share of this group that received a favourable decision.">
                    Share receiving a favourable decision
                  </th>
                  <th title="Groups below the configured reporting threshold are flagged here. They remain included in every calculation.">
                    Sample size note
                  </th>
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
                    <td>
                      {(stability?.small_groups || []).includes(
                        group.group,
                      ) ? (
                        <span
                          className="badge badge-warning badge-sm"
                          title={`Fewer than ${
                            stability?.min_group_size ?? "the configured"
                          } records. This rate may be sensitive to individual records. The group is counted in full and this flag does not affect the status.`}
                        >
                          small sample
                        </span>
                      ) : (
                        <span className="text-xs text-base-content/40">—</span>
                      )}
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
        <h1 className="text-2xl font-semibold">
          Fairness — are groups treated evenly?
        </h1>
        <Explainer className="mt-2 max-w-3xl">
          This page checks whether the model&apos;s favourable decisions are
          spread evenly across different groups of people. The group
          characteristic is the one your organisation declared as protected —
          the platform never picks one itself.
        </Explainer>
        <Explainer className="mt-2 max-w-3xl">
          A gap between groups is a prompt to investigate, not proof of
          discrimination. Small groups in particular can show large gaps from
          very few cases.
        </Explainer>
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
              title="Fair treatment across groups"
              right={<StatusBadge status={fairness?.status} size="lg" />}
            >
              {fairness ? (
                <FairnessBody fairness={fairness} stability={data?.fairness_rate_stability} />
              ) : (
                <Unavailable
                  title="No fairness result returned"
                  reason="The backend response contained no fairness block."
                />
              )}
            </Card>

            <Card
              title="Have the incoming cases changed?"
              subtitle="Compares the kind of cases in this data against an earlier baseline. This is about the incoming data, not the model's answers — for those, see Monitoring."
              right={<StatusBadge status={drift?.status} size="lg" />}
            >
              {drift ? (
                <div className="space-y-5">
                  <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <MetricField
                      label="Largest shift (PSI)"
                      term="population stability index"
                      value={num(drift.psi)}
                    />
                    <MetricField
                      label="Largest shift (KS)"
                      term="ks statistic"
                      value={num(drift.ks_statistic)}
                    />
                    <MetricField
                      label="Fields compared"
                      help="How many input fields were compared between the two periods."
                      value={drift.features_evaluated?.length ?? 0}
                    />
                    <MetricField
                      label="Data source"
                      help="Whether these figures were calculated from real model output or are sample data."
                      value={drift.is_mock ? "sample data" : "real calculation"}
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
