import { useEffect, useState } from 'react'
import {
  createBrowserRouter,
  Navigate,
  Outlet,
  RouterProvider,
  useLocation,
  useNavigate,
} from 'react-router-dom'
import SettingsDrawer from './components/SettingsDrawer'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'
import {
  ProjectContext,
  ProjectSetterContext,
  SelectedProjectContext,
  SelectedProjectSetterContext,
  type ProjectCtxValue,
  type SelectedProjectValue,
} from './context/ProjectContext'
import { useSettingsDrawer } from './lib/SettingsDrawer'
import ProjectsPage from './pages/Projects'
import QueuePage from './pages/Queue'
import QueueDetailPage from './pages/QueueDetail'
import QueueJobDetailPage from './pages/QueueJobDetail'
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

/**
 * 老路径 `/tools/settings?section=…` 的兼容跳转：跳首页同时把抽屉打开（保留
 * section 参数）。Settings 不再有自己的 URL；旧书签 / Topbar 通知按钮链接进
 * 来时不会 404，但落地是抽屉而非整页。
 */
function SettingsRedirect() {
  const drawer = useSettingsDrawer()
  const navigate = useNavigate()
  const location = useLocation()
  useEffect(() => {
    const section = new URLSearchParams(location.search).get('section')
    drawer.open(section ? { section } : undefined)
    navigate('/', { replace: true })
    // 只在 mount 时执行一次；deps 留空是有意的 —— 后续 location.search 变化是
    // navigate('/') 自己引起的，不要重复触发 open。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return null
}

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
 *  SettingsDrawer 用 fixed inset-0 铺满整个 viewport（含左侧 Sidebar）—— 这样点
 *  backdrop 的任意位置（包括 Sidebar 区域）都会收起抽屉。
 *  列父级 position:relative 保留作 <main> 内 absolute 元素（如任务日志抽屉贴底
 *  footer）的定位锚点。 */
function RootLayout() {
  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, position: 'relative' }}>
        <Topbar />
        <main style={{ flex: 1, overflow: 'auto', background: 'var(--bg-canvas)' }}>
          <Outlet />
        </main>
        <SettingsDrawer />
      </div>
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
        // P-G 数据作业详情。静态段 ranking 高于 /queue/:id，无冲突。
        { path: '/queue/jobs/:jid', element: <QueueJobDetailPage /> },
        { path: '/queue/:id', element: <QueueDetailPage /> },
        { path: '/queue/:id/log', element: <QueueDetailRedirect tab="log" /> },
        { path: '/queue/:id/monitor', element: <QueueDetailRedirect tab="monitor" /> },
        {
          path: '/projects/:pid',
          element: <ProjectLayout />,
          children: [
            { index: true, element: <ProjectOverview /> },
            { path: 'download', element: <DownloadPage /> },
            {
              path: 'v/:vid',
              children: [
                { path: 'curate', element: <CurationPage /> },
                // ADR 0010: preprocess 从 project scope 移到 version scope
                { path: 'preprocess', element: <PreprocessHub /> },
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
        { path: '/tools/settings', element: <SettingsRedirect /> },
        { path: '/tools/generate', element: <GeneratePage /> },
        { path: '/configs', element: <Navigate to="/tools/presets" replace /> },
        { path: '/monitor', element: <Navigate to="/tools/monitor" replace /> },
        { path: '/datasets', element: <Navigate to="/" replace /> },
      ],
    },
  ],
  {
    // ADR 0012：SPA 挂在根路径，不再用 /studio 子路径前缀。
    basename: '/',
    future: { v7_relativeSplatPath: true },
  },
)

export default function App() {
  const [projectCtx, setProjectCtx] = useState<ProjectCtxValue | null>(null)
  // 跨页保留的"已选中项目"快照（见 ProjectContext 注释）
  const [selectedProject, setSelectedProject] = useState<SelectedProjectValue | null>(null)

  return (
    <ProjectContext.Provider value={projectCtx}>
      <ProjectSetterContext.Provider value={setProjectCtx}>
        <SelectedProjectContext.Provider value={selectedProject}>
          <SelectedProjectSetterContext.Provider value={setSelectedProject}>
            <RouterProvider router={router} />
          </SelectedProjectSetterContext.Provider>
        </SelectedProjectContext.Provider>
      </ProjectSetterContext.Provider>
    </ProjectContext.Provider>
  )
}
