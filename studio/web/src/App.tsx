import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import ConfigsPage from './pages/Configs'
import MonitorPage from './pages/Monitor'
import Placeholder from './pages/Placeholder'

export default function App() {
  return (
    <BrowserRouter basename="/studio">
      <div className="min-h-screen flex">
        <Sidebar />
        <main className="flex-1 px-8 py-6 overflow-auto h-screen">
          <Routes>
            <Route path="/" element={<MonitorPage />} />
            <Route path="/configs" element={<ConfigsPage />} />
            <Route
              path="/datasets"
              element={
                <Placeholder
                  title="数据集"
                  phase="P4"
                  description="P4 阶段会扫描 dataset/ 目录，按 Kohya 风格 N_xxx 子目录显示样本数、caption 类型分布、缩略图。"
                />
              }
            />
            <Route
              path="/queue"
              element={
                <Placeholder
                  title="任务队列"
                  phase="P3"
                  description="P3 阶段加入 SQLite + supervisor，把多个训练任务排成队列依次跑，能取消、能查实时日志。"
                />
              }
            />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
