/**
 * Application shell: sidebar navigation + model selector + backend health.
 *
 * The sidebar keeps the model selector permanently visible, because every
 * page below it is scoped to that model and the user should never have to
 * remember which one is active.
 */
import { NavLink, Outlet } from "react-router-dom";

import { getApiHealth } from "../api/models";
import { ModelSelector } from "../components/ModelSelector";
import { useApiResource } from "../hooks/useApiResource";

/**
 * Each item carries a one-line description of what the page answers.
 *
 * The technical name stays as the label -- it is the term the rest of the
 * organisation uses -- with the plain-language purpose underneath, so a
 * first-time reader does not have to already know what "Explainability" or
 * "Drift" means in order to navigate.
 */
const NAV = [
  { to: "/", label: "Dashboard", hint: "Overall status and what needs attention", end: true },
  { to: "/models", label: "Models", hint: "Which models can be assessed" },
  {
    to: "/explainability",
    label: "Explainability",
    hint: "Why the model decided what it did",
  },
  { to: "/fairness", label: "Fairness", hint: "Are groups treated evenly?" },
  { to: "/monitoring", label: "Monitoring", hint: "What has changed since before?" },
  { to: "/compliance", label: "RBI Compliance", hint: "Rule checks and requirements" },
  { to: "/report", label: "Assurance Report", hint: "The written summary" },
];

function BackendHealth() {
  const { data, error, loading } = useApiResource(
    ({ signal }) => getApiHealth({ signal }),
    [],
  );

  // Plain wording, and a text glyph so the state does not depend on the dot's
  // colour alone.
  let tone = "bg-base-content/30";
  let glyph = "…";
  let text = "Checking connection…";
  let help = "Checking whether the assurance service is reachable.";
  if (!loading && error) {
    tone = "bg-error";
    glyph = "✕";
    text = "Service unavailable";
    help =
      "The assurance service is not responding, so no results can be loaded. Nothing shown on screen is out of date — there is simply nothing to show.";
  } else if (!loading && data) {
    tone = "bg-success";
    glyph = "✓";
    text = "Connected";
    help = "The assurance service is reachable and results can be loaded.";
  }

  return (
    <div
      className="flex items-center gap-2 px-4 py-3 border-t border-base-300 text-xs text-base-content/60"
      title={help}
      aria-label={`${text}. ${help}`}
    >
      <span className={`inline-block w-2 h-2 rounded-full ${tone}`} aria-hidden="true" />
      <span aria-hidden="true">{glyph}</span>
      <span>{text}</span>
    </div>
  );
}

export function AppLayout() {
  return (
    <div className="min-h-screen bg-base-200 flex">
      <aside className="w-64 shrink-0 bg-base-100 border-r border-base-300 flex flex-col">
        <div className="px-4 py-5 border-b border-base-300">
          <p className="text-[11px] uppercase tracking-widest text-base-content/50">
            RBI
          </p>
          <h1 className="text-base font-semibold leading-tight mt-0.5">
            Model Risk &amp; Assurance Copilot
          </h1>
        </div>

        <div className="border-b border-base-300">
          <ModelSelector />
        </div>

        <nav className="flex-1 py-3" aria-label="Sections">
          <ul className="menu menu-sm px-2 gap-0.5">
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  aria-label={`${item.label} — ${item.hint}`}
                  className={({ isActive }) =>
                    `flex flex-col items-start gap-0 leading-tight py-2 ${
                      isActive
                        ? "active font-medium"
                        : "text-base-content/70 hover:text-base-content"
                    }`
                  }
                >
                  <span>{item.label}</span>
                  <span className="text-[11px] font-normal text-base-content/45 leading-snug">
                    {item.hint}
                  </span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <BackendHealth />
      </aside>

      <main className="flex-1 min-w-0">
        <div className="max-w-[1400px] mx-auto px-8 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
