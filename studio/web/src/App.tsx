import { useState } from 'react'
import {
  createBrowserRouter,
  Navigate,
  Outlet,
  RouterProvider,
} from 'react-router-dom'
import SettingsDrawer from './components/SettingsDrawer'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'
import { ProjectContext, ProjectSetterContext, type ProjectCtxValue } from './context/ProjectContext'
import ProjectsPage from './pages/Projects'
import QueuePage from './pages/Queue'
import QueueDetailPage from './pages/QueueDetail'
import ProjectLayout from './pages/project/Layout'
import ProjectOverview from './pages/project/Overview'
import CurationPage from './pages/project/steps/Curation'
import DownloadPage from './pages/project/steps/Download'
import PreprocessHub from './pages/project/steps/PreprocessHub'
import RegularizationPage from './pages/project/steps/Regularization'
import TagEditPage from './pages/project/steps/TagEdit'
import TaggingPage from './pages/project/steps/Tagging'
import TrainPage from './pages/project/steps/Train'
import GeneratePage from './pages/tools/Generate'
import MonitorPage from './pages/tools/Monitor'
import PresetsPage from './pages/tools/Presets'
import SettingsPage from './pages/tools/Settings'

/**
 * 老路径 `/queue/:id/log` 和 `/queue/:id/monitor` 的兼容跳转：保留 URL 不删，
 * 转到新 detail 页对应 tab（用 hash 表达 tab）。让书签 / 收藏链接不失效。
 */
function QueueDetailRedirect({ tab }: { tab: 'log' | 'monitor' }) {
  const path = window.location.pathname
  const id = path.match(/\/queue\/(\d+)/)?.[1]
  if (!id) return <Navigate to="/queue" replace />
  return (
    <Navigate to={{ pathname: `/queue/${id}`, hash: tab }} replace />
  )
}

/** Sidebar + Topbar 外壳；所有路由 element 渲染进 <Outlet />。
 *  SettingsDrawer 也挂在这层 —— 它是个 fixed 定位的全局抽屉，永远在 RouterProvider 内
 *  这样可以共用 react-router context（虽然抽屉本身不依赖路由，但内部的 SettingsPage 用到
 *  useLocation 等 hook，需要 router 上下文）。 */
function RootLayout() {
  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Topbar />
        <main style={{ flex: 1, overflow: 'auto', background: 'var(--bg-canvas)' }}>
          <Outlet />
        </main>
      </div>
      <SettingsDrawer />
    </div>
  )
}

// DataRouter 单例：从经典 BrowserRouter 迁过来是为了让 react-router v6 的
// useBlocker 可用（BrowserRouter 不支持），TagEdit 等页面的"未保存切页"
// 提示需要它。结构跟原来一致 —— RootLayout 包 Sidebar/Topbar，业务路由
// 作为 children 渲染进 <Outlet />。
const router = createBrowserRouter(
  [
    {
      element: <RootLayout />,
      children: [
        { path: '/', element: <ProjectsPage /> },
        { path: '/queue', element: <QueuePage /> },
        { path: '/queue/:id', element: <QueueDetailPage /> },
        { path: '/queue/:id/log', element: <QueueDetailRedirect tab="log" /> },
        { path: '/queue/:id/monitor', element: <QueueDetailRedirect tab="monitor" /> },
        {
          path: '/projects/:pid',
          element: <ProjectLayout />,
          children: [
            { index: true, element: <ProjectOverview /> },
            { path: 'download', element: <DownloadPage /> },
            { path: 'preprocess', element: <PreprocessHub /> },
            {
              path: 'v/:vid',
              children: [
                { path: 'curate', element: <CurationPage /> },
                { path: 'tag', element: <TaggingPage /> },
                { path: 'edit', element: <TagEditPage /> },
                { path: 'reg', element: <RegularizationPage /> },
                { path: 'train', element: <TrainPage /> },
              ],
            },
          ],
        },
        { path: '/tools/presets', element: <PresetsPage /> },
        { path: '/tools/monitor', element: <MonitorPage /> },
        { path: '/tools/settings', element: <SettingsPage /> },
        { path: '/tools/generate', element: <GeneratePage /> },
        { path: '/configs', element: <Navigate to="/tools/presets" replace /> },
        { path: '/monitor', element: <Navigate to="/tools/monitor" replace /> },
        { path: '/datasets', element: <Navigate to="/" replace /> },
      ],
    },
  ],
  {
    basename: '/studio',
    future: { v7_relativeSplatPath: true },
  },
)

export default function App() {
  const [projectCtx, setProjectCtx] = useState<ProjectCtxValue | null>(null)

  return (
    <ProjectContext.Provider value={projectCtx}>
      <ProjectSetterContext.Provider value={setProjectCtx}>
        <RouterProvider router={router} />
      </ProjectSetterContext.Provider>
    </ProjectContext.Provider>
  )
}
