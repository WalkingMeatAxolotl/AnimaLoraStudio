import { createContext, useContext } from 'react'
import type { ProjectDetail, Version } from '../api/client'

export interface ProjectCtxValue {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  onSelectVersion: (vid: number) => void
  /** 打开"新版本"对话框。传 forkFromVid 则预填 forkFrom 下拉框（复制配置开新版本 CTA 用）。 */
  onCreateVersion: (forkFromVid?: number) => void
  onExportTrain: () => void
  onDeleteVersion: (vid: number) => void
  exporting: boolean
}

export const ProjectContext = createContext<ProjectCtxValue | null>(null)
export const useProjectCtx = () => useContext(ProjectContext)

// Setter lives at App level so Layout.tsx can push context up to Sidebar
export const ProjectSetterContext = createContext<((v: ProjectCtxValue | null) => void) | null>(null)
export const useProjectCtxSetter = () => useContext(ProjectSetterContext)

// ── sticky "已选中项目" ──────────────────────────────────────────────────────
// ProjectContext 在离开 /projects/:pid 路由（Layout 卸载）时被清空，导致切到
// 队列 / 测试等全局页后侧边栏丢失当前项目。这份只读快照在 Layout 加载时写入、
// 卸载时**不清**，让侧边栏跨页保留"选中项目"用于导航；版本增删/导出等需要
// live Layout 的交互仍只在项目内（ProjectContext 存在时）可用。切换项目仍走
// 项目列表页手动选择 —— 打开另一个项目会用新项目覆盖这份快照。
export interface SelectedProjectValue {
  project: ProjectDetail
  activeVersion: Version | null
}
export const SelectedProjectContext = createContext<SelectedProjectValue | null>(null)
export const useSelectedProject = () => useContext(SelectedProjectContext)
export const SelectedProjectSetterContext = createContext<
  ((v: SelectedProjectValue | null) => void) | null
>(null)
export const useSelectedProjectSetter = () => useContext(SelectedProjectSetterContext)
