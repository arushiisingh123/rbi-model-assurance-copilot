/**
 * Dashboard: run an assessment and see every domain's status at a glance.
 *
 * The status tiles are read straight from the backend's own status fields. No
 * status is computed here -- deriving an "overall risk" in JavaScript would
 * be the frontend inventing a finding, and the backend already publishes each
 * domain's verdict.
 *
 * The "Needs your attention" panel is a FILTER over those same backend
 * statuses, in a fixed order. It ranks nothing and decides nothing: it just
 * lifts the items already marked WARNING or FAIL to the top of the page so a
 * reader does not have to find them by scanning four tiles.
 *
 * PENDING is counted and shown separately from FAIL throughout, so "we could
 * not measure this" is never folded into "this failed".
 */
import { Link } from "react-router-dom";

import { IdentityBar } from "../components/IdentityBar";
import {
  Card,
  ErrorState,
  Explainer,
  Field,
  HelpTip,
  Loading,
  MetricField,
  StatusBadge,
  StatusLegend,
  Unavailable,
} from "../components/states";
import { useAssurance } from "../hooks/AssuranceContext";
import { useModels } from "../hooks/ModelContext";
import { explanationCaption, num } from "../utils/format";
import { describeStatus, needsAttention } from "../utils/glossary";

/** One domain tile: plain-language name, what it checks, and its status. */
function StatusTile({ label, question, status, caption, to }) {
  const body = (
    <div className="bg-base-100 border border-base-300 rounded-lg p-4 h-full hover:border-primary/40 focus-within:border-primary/40 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-semibold leading-tight">{label}</p>
        <StatusBadge status={status} size="sm" />
      </div>
      <p className="mt-1 text-xs text-base-content/50 leading-snug">{question}</p>
      <p className="mt-2 text-sm text-base-content/70 leading-snug min-h-[2.5rem]">
        {caption}
      </p>
      {to && (
        <p className="mt-2 text-xs font-medium text-primary">View details →</p>
      )}
    </div>
  );
  return to ? (
    <Link to={to} className="block h-full rounded-lg" aria-label={`${label}: view details`}>
      {body}
    </Link>
  ) : (
    body
  );
}

/**
 * The items a reader should look at first.
 *
 * Shown only when something is actually flagged. When nothing is, the panel
 * states that plainly AND restates what was not measured -- an all-clear that
 * hid two unmeasured checks would be a misleading all-clear.
 */
function AttentionPanel({ items, pendingItems }) {
  const flagged = items.filter((item) => needsAttention(item.status));

  if (!flagged.length) {
    return (
      <Card title="Needs your attention">
        <div className="flex items-start gap-3">
          <span aria-hidden="true" className="text-lg leading-none mt-0.5">
            ✓
          </span>
          <div>
            <p className="text-sm font-medium">
              Nothing is flagged for review in this assessment.
            </p>
            <Explainer className="mt-1">
              Every check that could be carried out came back within its agreed
              limits.
              {pendingItems.length > 0 && (
                <>
                  {" "}
                  However, {pendingItems.length}{" "}
                  {pendingItems.length === 1 ? "check was" : "checks were"} not
                  measured at all ({pendingItems.map((i) => i.label).join(", ")}).
                  That is not the same as passing — nothing was verified there.
                </>
              )}
            </Explainer>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card
      title="Needs your attention"
      subtitle="Start here. These are the checks that came back outside their agreed limits."
    >
      <ul className="space-y-3">
        {flagged.map((item) => {
          const { meaning } = describeStatus(item.status);
          return (
            <li
              key={item.label}
              className="flex flex-wrap items-start gap-3 rounded-md border border-base-300 bg-base-200/40 px-4 py-3"
            >
              <StatusBadge status={item.status} size="sm" />
              <div className="flex-1 min-w-[14rem]">
                <p className="text-sm font-medium">{item.label}</p>
                <p className="mt-0.5 text-sm text-base-content/70 leading-snug">
                  {item.why || meaning}
                </p>
              </div>
              {item.to && (
                <Link to={item.to} className="btn btn-sm btn-outline">
                  Look into this
                </Link>
              )}
            </li>
          );
        })}
      </ul>
      <Explainer className="mt-4">
        A flagged result is a prompt to investigate, not a conclusion. It does
        not by itself mean the model is faulty, that anyone acted improperly, or
        that a regulation has been breached.
      </Explainer>
    </Card>
  );
}

export function DashboardPage() {
  const { selectedModel, selectedModelId, modelsLoading } = useModels();
  const { result, error, running, start, hasRun } = useAssurance();

  const explain = result?.explainability;
  const fairness = result?.fairness_drift?.fairness;
  const drift = result?.fairness_drift?.drift;
  const compliance = result?.compliance;
  const monitoring = result?.monitoring;

  // Backend-issued statuses only. Nothing below decides a verdict.
  const complianceStatus = compliance?.findings?.some((f) => f.status === "FAIL")
    ? "FAIL"
    : compliance?.findings?.some((f) => f.status === "WARNING")
      ? "WARNING"
      : compliance?.findings?.length
        ? "PASS"
        : "PENDING";

  const explainStatus =
    explain?.available === false ? "PENDING" : explain?.available ? "PASS" : null;
  const monitoringStatus = monitoring ? monitoring.result?.monitoring_status : "PENDING";

  const domains = [
    {
      label: "How the model decides",
      question: "Which information drove its answers?",
      status: explainStatus,
      to: "/explainability",
      caption:
        explain?.available === false
          ? "Could not be produced for this model — the page explains why."
          : explanationCaption(explain) || "An explanation was produced.",
      why:
        explain?.available === false
          ? "No explanation could be produced for this model, so there is nothing to review here yet."
          : null,
    },
    {
      label: "Fair treatment across groups",
      question: "Are favourable decisions spread evenly?",
      status: fairness?.status,
      to: "/fairness",
      caption:
        fairness?.protected_attribute === "none_declared"
          ? "No protected attribute was declared for this model."
          : `Measured across: ${fairness?.protected_attribute ?? "—"}`,
      why:
        fairness?.protected_attribute && fairness?.protected_attribute !== "none_declared"
          ? `Favourable decisions are not spread evenly across groups of "${fairness.protected_attribute}".`
          : null,
    },
    {
      label: "Changes since the last period",
      question: "Has the data or the model's output shifted?",
      status: monitoringStatus,
      to: "/monitoring",
      caption: monitoring
        ? "Compares a recent period against an earlier baseline."
        : "Not performed in this assessment.",
      why: monitoring
        ? "Something has moved between the two periods compared — the monitoring page shows which part."
        : null,
    },
    {
      // NOT "RBI rule checks". These findings are the six-rule engine in
      // app/rbi/rules, every one of which carries rbi_source =
      // "ILLUSTRATIVE ..." and thresholds that are project conventions. The
      // API schema for ComplianceFinding states they "must never be labelled
      // or displayed as RBI requirements", and the Compliance page already
      // calls them technical assurance. Calling them RBI rule checks here
      // made a threshold FAIL read as a regulatory breach on the one screen
      // most people start from. The verified RBI register is a separate
      // layer, shown on the Compliance page.
      label: "Technical assurance checks",
      question: "What do the built-in model checks say?",
      status: complianceStatus,
      to: "/compliance",
      caption: `${compliance?.findings?.length ?? 0} illustrative sample rule${
        compliance?.findings?.length === 1 ? "" : "s"
      } checked against this model's results — not RBI requirements.`,
      why: compliance?.findings?.length
        ? "At least one technical check came back outside its agreed limit. These limits are project conventions, not RBI rules."
        : null,
    },
  ];

  const failing = domains.filter((d) => d.status === "FAIL").length;
  const warning = domains.filter((d) => d.status === "WARNING").length;
  const pendingItems = domains.filter((d) => d.status === "PENDING");

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-6 flex-wrap">
        <div className="max-w-2xl">
          <h1 className="text-2xl font-semibold">Model Assurance Dashboard</h1>
          <Explainer className="mt-2">
            This dashboard checks one of your organisation's decision-making
            models and reports what it found. It looks at how the model reaches
            its decisions, whether those decisions are spread evenly across
            groups of people, and whether anything has changed since an
            earlier period. It also runs a set of built-in technical checks.
            Those checks use project thresholds and are <strong>not</strong>{" "}
            RBI regulatory requirements — the verified RBI requirements are a
            separate layer on the Compliance page.
          </Explainer>
          <Explainer className="mt-2">
            Choose a model on the left, then run an assessment. Results appear
            only once the assessment has actually run — this page never shows
            placeholder figures.
          </Explainer>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={start}
          disabled={running || !selectedModelId || modelsLoading}
        >
          {running && <span className="loading loading-spinner loading-sm" />}
          {running
            ? "Running assessment…"
            : hasRun
              ? "Run assessment again"
              : "Run assessment"}
        </button>
      </header>

      <IdentityBar
        modelId={selectedModelId}
        modelType={selectedModel?.model_type}
        modelVersion={selectedModel?.model_version}
        integrationType={selectedModel?.integration_type}
        runId={result?.assurance_run_id}
      />

      {error && <ErrorState error={error} onRetry={start} context="The assessment" />}

      {running && (
        <Card title="Assessment in progress">
          <Loading
            label="Checking how the model decides, fair treatment, changes since the last period, and the technical assurance checks…"
            rows={4}
          />
        </Card>
      )}

      {!running && !hasRun && !error && (
        <Card title="No assessment has been run yet">
          <Explainer>
            Nothing has been assessed for{" "}
            <span className="font-mono text-base-content">
              {selectedModelId || "the selected model"}
            </span>{" "}
            in this session. Run an assessment to see results. Until then this
            page stays empty on purpose — showing sample figures on a risk
            dashboard would be worse than showing nothing.
          </Explainer>
          <button
            type="button"
            className="btn btn-primary btn-sm mt-4"
            onClick={start}
            disabled={!selectedModelId}
          >
            Run assessment
          </button>
        </Card>
      )}

      {hasRun && (
        <>
          <AttentionPanel items={domains} pendingItems={pendingItems} />

          <section>
            <div className="flex items-center justify-between gap-4 mb-3 flex-wrap">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-base-content/70">
                All four checks
              </h2>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {domains.map((domain) => (
                <StatusTile key={domain.label} {...domain} />
              ))}
            </div>
            <div className="mt-4">
              <StatusLegend />
            </div>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card title="About this assessment" className="lg:col-span-1">
              <dl className="space-y-3">
                <Field label="Model checked" value={result.model_id} mono />
                <MetricField
                  label="Assurance run"
                  term="assurance run"
                  value={result.assurance_run_id}
                  mono
                />
                <Field
                  label="Model version"
                  value={result.model?.model_metadata?.version}
                />
                <Field
                  label="Records checked"
                  value={result.model?.predictions?.length}
                />
                <div className="flex flex-col gap-0.5">
                  <dt className="text-xs uppercase tracking-wide text-base-content/50">
                    Result summary
                  </dt>
                  <dd className="text-sm flex flex-wrap gap-1.5 mt-0.5">
                    <span className="badge badge-error badge-sm gap-1">
                      <span aria-hidden="true">✕</span> {failing} need action
                    </span>
                    <span className="badge badge-warning badge-sm gap-1">
                      <span aria-hidden="true">!</span> {warning} need review
                    </span>
                    <span className="badge badge-ghost border-base-300 badge-sm gap-1">
                      <span aria-hidden="true">◔</span> {pendingItems.length} not
                      measured
                    </span>
                  </dd>
                </div>
              </dl>
            </Card>

            <Card
              title="How well the model performs"
              subtitle="Measured on a held-back sample the model was not trained on."
              className="lg:col-span-2"
            >
              {result.model?.model_metrics ? (
                <>
                  <dl className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                    <MetricField
                      label="Accuracy"
                      value={num(result.model.model_metrics.accuracy)}
                    />
                    <MetricField
                      label="Precision"
                      value={num(result.model.model_metrics.precision)}
                    />
                    <MetricField
                      label="Recall"
                      value={num(result.model.model_metrics.recall)}
                    />
                    <MetricField label="F1" value={num(result.model.model_metrics.f1)} />
                    <MetricField
                      label="ROC AUC"
                      term="roc auc"
                      value={
                        result.model.model_metrics.roc_auc_status === "computed"
                          ? num(result.model.model_metrics.roc_auc)
                          : "not available for this model"
                      }
                    />
                    <MetricField
                      label="Cases tested"
                      help="How many held-back cases these figures were measured on."
                      value={result.model.model_metrics.n_test_samples}
                    />
                  </dl>
                  <Explainer className="mt-4">
                    These figures describe how well the model performed on a
                    sample it had never seen. They say nothing about fairness,
                    about whether conditions have changed since, or about
                    regulatory compliance — those are the separate checks above.
                  </Explainer>
                </>
              ) : (
                <Unavailable
                  title="Performance figures do not apply to this model"
                  reason="This model works on a different set of input fields from the dataset these figures are defined on, so there is no accuracy figure that would honestly describe it."
                />
              )}
            </Card>
          </div>

          <Card
            title="What this assessment covered"
            subtitle="Read this before quoting any figure above."
            right={<HelpTip term="provenance" />}
          >
            <Explainer>{result.note}</Explainer>
          </Card>
        </>
      )}
    </div>
  );
}
