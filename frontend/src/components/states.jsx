/**
 * Status, loading, empty, error and UNAVAILABLE presentation primitives.
 *
 * The distinction these components exist to protect: a capability that could
 * not be measured is NOT a pass, and it is NOT an error either. The backend
 * models three separate outcomes and the UI keeps them three:
 *
 *   PASS / WARNING / FAIL   -> something was measured, here is the verdict
 *   PENDING / unavailable   -> nothing was measured, here is the reason
 *   error                   -> the request itself failed, here is the status
 *
 * Collapsing the middle case into either neighbour is the failure mode that
 * matters most in a compliance tool, so it gets its own component and its own
 * neutral (never green) colour.
 */

/** Map a backend status to a DaisyUI badge class. PENDING is never green. */
const STATUS_STYLES = {
  PASS: "badge-success",
  WARNING: "badge-warning",
  FAIL: "badge-error",
  PENDING: "badge-ghost border-base-300",
  COMPARABLE: "badge-success",
  NOT_COMPARABLE: "badge-ghost border-base-300",
  RETRIEVED: "badge-success",
  NOT_FOUND: "badge-ghost border-base-300",
};

export function StatusBadge({ status, size = "md", title }) {
  if (status === null || status === undefined || status === "") {
    return <span className="badge badge-ghost border-base-300 text-xs">unknown</span>;
  }
  const key = String(status).toUpperCase();
  const style = STATUS_STYLES[key] || "badge-ghost border-base-300";
  const sizeClass = size === "lg" ? "badge-lg" : size === "sm" ? "badge-sm" : "";
  return (
    <span
      className={`badge ${style} ${sizeClass} font-medium whitespace-nowrap`}
      title={title || `Status: ${status}`}
    >
      {String(status)}
    </span>
  );
}

/** A section container. `right` holds per-card actions or a status badge. */
export function Card({ title, subtitle, right, children, className = "" }) {
  return (
    <section
      className={`bg-base-100 border border-base-300 rounded-lg shadow-sm ${className}`}
    >
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 px-5 py-4 border-b border-base-300">
          <div className="min-w-0">
            {title && (
              <h2 className="text-sm font-semibold uppercase tracking-wide text-base-content/70">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-1 text-sm text-base-content/60">{subtitle}</p>
            )}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Loading({ label = "Loading…", rows = 3 }) {
  return (
    <div className="space-y-3" role="status" aria-live="polite">
      <div className="flex items-center gap-2 text-sm text-base-content/60">
        <span className="loading loading-spinner loading-sm" />
        <span>{label}</span>
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-4 bg-base-200 rounded animate-pulse" />
      ))}
    </div>
  );
}

export function Empty({ title = "Nothing to show", hint }) {
  return (
    <div className="text-center py-8">
      <p className="text-sm font-medium text-base-content/70">{title}</p>
      {hint && <p className="mt-1 text-sm text-base-content/50">{hint}</p>}
    </div>
  );
}

/**
 * A capability the backend could not provide, with the backend's own reason.
 *
 * Deliberately NOT styled as an error (nothing is broken) and NOT as a pass
 * (nothing was verified). `reason` is always the backend's text, never a
 * phrase invented here.
 */
export function Unavailable({ title = "Unavailable", reason, details }) {
  return (
    <div className="rounded-md border border-base-300 bg-base-200/60 p-4">
      <div className="flex items-center gap-2">
        <span className="badge badge-ghost border-base-300 badge-sm">unavailable</span>
        <p className="text-sm font-semibold text-base-content/80">{title}</p>
      </div>
      {reason && (
        <p className="mt-2 text-sm leading-relaxed text-base-content/70">{reason}</p>
      )}
      {details}
      <p className="mt-3 text-xs italic text-base-content/50">
        This means the check was not performed. It is not a pass, and it is not
        evidence that the model is compliant.
      </p>
    </div>
  );
}

/**
 * A failed request, showing the mapped message plus the backend's `detail`.
 *
 * The backend raises HTTPException with human-readable detail, so there is no
 * traceback to leak. No fallback data is substituted.
 */
export function ErrorState({ error, onRetry, context }) {
  if (!error) return null;
  return (
    <div className="rounded-md border border-error/30 bg-error/5 p-4">
      <div className="flex items-center gap-2">
        <span className="badge badge-error badge-sm">
          {error.status ? `HTTP ${error.status}` : "network"}
        </span>
        <p className="text-sm font-semibold text-error">
          {context ? `${context} could not be loaded` : "Request failed"}
        </p>
      </div>
      <p className="mt-2 text-sm text-base-content/80">{error.message}</p>
      {error.detail && (
        <p className="mt-1 text-sm text-base-content/60 break-words">{error.detail}</p>
      )}
      {onRetry && (
        <button type="button" className="btn btn-sm btn-outline mt-3" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

/**
 * Standard wrapper: loading -> error -> empty -> content.
 *
 * Keeps every page's state handling identical, so no page can accidentally
 * render `data` while it is still null.
 */
export function AsyncSection({
  loading,
  error,
  data,
  onRetry,
  context,
  loadingLabel,
  emptyTitle,
  emptyHint,
  children,
}) {
  if (loading) return <Loading label={loadingLabel || "Loading…"} />;
  if (error) return <ErrorState error={error} onRetry={onRetry} context={context} />;
  if (data === null || data === undefined) {
    return <Empty title={emptyTitle || "No data yet"} hint={emptyHint} />;
  }
  return children;
}

/** Label/value row used throughout the detail cards. */
export function Field({ label, value, mono = false, className = "" }) {
  const empty = value === null || value === undefined || value === "";
  return (
    <div className={`flex flex-col gap-0.5 ${className}`}>
      <dt className="text-xs uppercase tracking-wide text-base-content/50">{label}</dt>
      <dd
        className={`text-sm ${mono ? "font-mono text-xs break-all" : ""} ${
          empty ? "text-base-content/40 italic" : "text-base-content"
        }`}
      >
        {empty ? "not provided" : value}
      </dd>
    </div>
  );
}
