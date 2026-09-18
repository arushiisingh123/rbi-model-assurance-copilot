/**
 * Routes and global providers.
 *
 * ModelProvider wraps AssuranceProvider because the assurance run is scoped
 * to the selected model: when the model changes the provider clears the
 * stored run, which is what stops one model's findings appearing under
 * another's identity.
 */
import { Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "./layouts/AppLayout";
import { AssuranceProvider } from "./hooks/AssuranceContext";
import { ModelProvider } from "./hooks/ModelContext";
import { CompliancePage } from "./pages/CompliancePage";
import { DashboardPage } from "./pages/DashboardPage";
import { ExplainabilityPage } from "./pages/ExplainabilityPage";
import { FairnessPage } from "./pages/FairnessPage";
import { ModelsPage } from "./pages/ModelsPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { ReportPage } from "./pages/ReportPage";

export default function App() {
  return (
    <ModelProvider>
      <AssuranceProvider>
        <Routes>
          <Route element={<AppLayout />} path="/">
            <Route index element={<DashboardPage />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="explainability" element={<ExplainabilityPage />} />
            <Route path="fairness" element={<FairnessPage />} />
            <Route path="monitoring" element={<MonitoringPage />} />
            <Route path="compliance" element={<CompliancePage />} />
            <Route path="report" element={<ReportPage />} />
            <Route path="*" element={<Navigate replace to="/" />} />
          </Route>
        </Routes>
      </AssuranceProvider>
    </ModelProvider>
  );
}
