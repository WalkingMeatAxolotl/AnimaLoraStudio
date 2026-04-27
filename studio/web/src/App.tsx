import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import LogPage from './pages/Log'
import ProjectsPage from './pages/Projects'
import QueuePage from './pages/Queue'
import ProjectLayout from './pages/project/Layout'
import ProjectOverview from './pages/project/Overview'
import StepPlaceholder from './pages/project/steps/Placeholder'
import MonitorPage from './pages/tools/Monitor'
import PresetsPage from './pages/tools/Presets'
import SettingsPage from './pages/tools/Settings'

export default function App() {
  return (
    <BrowserRouter basename="/studio">
      <div className="min-h-screen flex">
        <Sidebar />
        <main className="flex-1 px-8 py-6 overflow-auto h-screen">
          <Routes>
            <Route path="/" element={<ProjectsPage />} />
            <Route path="/queue" element={<QueuePage />} />
            <Route path="/queue/:id/log" element={<LogPage />} />

            {/* PP1: project layout + stepper + version tabs */}
            <Route path="/projects/:pid" element={<ProjectLayout />}>
              <Route index element={<ProjectOverview />} />
              <Route
                path="download"
                element={<StepPlaceholder step="① 下载" doc="PP2" />}
              />
              <Route path="v/:vid">
                <Route
                  path="curate"
                  element={<StepPlaceholder step="② 筛选" doc="PP3" />}
                />
                <Route
                  path="tag"
                  element={<StepPlaceholder step="③ 打标" doc="PP4" />}
                />
                <Route
                  path="reg"
                  element={<StepPlaceholder step="④ 正则集" doc="PP5" />}
                />
                <Route
                  path="train"
                  element={<StepPlaceholder step="⑤ 训练" doc="PP6" />}
                />
              </Route>
            </Route>

            <Route path="/tools/presets" element={<PresetsPage />} />
            <Route path="/tools/monitor" element={<MonitorPage />} />
            <Route path="/tools/settings" element={<SettingsPage />} />

            {/* 旧 → 新 路由兼容（PP0 重构）。下个 minor 版本删除。 */}
            <Route
              path="/configs"
              element={<Navigate to="/tools/presets" replace />}
            />
            <Route
              path="/monitor"
              element={<Navigate to="/tools/monitor" replace />}
            />
            <Route path="/datasets" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
