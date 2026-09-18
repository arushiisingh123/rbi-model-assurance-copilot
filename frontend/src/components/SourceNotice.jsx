/**
 * Says where the numbers on a page came from.
 *
 * Without this the two cases are indistinguishable on screen: findings that
 * belong to the dashboard's assurance run, and findings from a separate
 * request that happens to be showing the same model. They can legitimately
 * differ (a standalone endpoint recomputes, and three of them mint a new
 * assurance_run_id doing so), so the provenance is stated rather than
 * implied.
 */
import { SOURCE_ASSURANCE_RUN } from "../hooks/useDomainData";

export function SourceNotice({ source, runId, onRunAssurance, domain }) {
  if (source === SOURCE_ASSURANCE_RUN) {
    return (
      <div className="flex flex-wrap items-center gap-2 text-xs text-base-content/60">
        <span className="badge badge-outline badge-sm">assurance run</span>
        <span>
          Showing the {domain} findings from the current assurance run
          {runId && (
            <>
              {" "}
              (<span className="font-mono text-base-content/80">{runId}</span>)
            </>
          )}
          .
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-base-content/60">
      <span className="badge badge-ghost border-base-300 badge-sm">
        independent computation
      </span>
      <span>
        No assurance run has been performed for this model in this session, so
        this {domain} result was computed by its own request and is not part of
        an assurance run.
      </span>
      {onRunAssurance && (
        <button type="button" className="btn btn-xs btn-outline" onClick={onRunAssurance}>
          Run assurance
        </button>
      )}
    </div>
  );
}
