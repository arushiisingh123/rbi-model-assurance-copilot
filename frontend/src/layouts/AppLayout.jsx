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

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/models", label: "Models" },
  { to: "/explainability", label: "Explainability" },
  { to: "/fairness", label: "Fairness" },
  { to: "/monitoring", label: "Monitoring" },
  { to: "/compliance", label: "RBI Compliance" },
  { to: "/report", label: "Assurance Report" },
];

function BackendHealth() {
  const { data, error, loading } = useApiResource(
    ({ signal }) => getApiHealth({ signal }),
    [],
  );

  let tone = "bg-base-content/30";
  let text = "checking backend…";
  if (!loading && error) {
    tone = "bg-error";
    text = "backend unreachable";
  } else if (!loading && data) {
    tone = "bg-success";
    text = "backend connected";
  }

  return (
    <div className="flex items-center gap-2 px-4 py-3 border-t border-base-300 text-xs text-base-content/60">
      <span className={`inline-block w-2 h-2 rounded-full ${tone}`} />
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

        <nav className="flex-1 py-3">
          <ul className="menu menu-sm px-2 gap-0.5">
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    isActive
                      ? "active font-medium"
                      : "text-base-content/70 hover:text-base-content"
                  }
                >
                  {item.label}
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
