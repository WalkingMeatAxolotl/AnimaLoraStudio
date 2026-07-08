import type { ReactNode } from 'react'
import PageHeader from './PageHeader'
import TaskLogDrawer, { type LogSource } from './TaskLogDrawer'

interface Props {
  idx: number | string
  title: string
  subtitle?: string
  actions?: ReactNode
  topRight?: ReactNode
  children: ReactNode
  /** header 与内容区之间的全宽条（如过滤 / 排序行），不随内容滚动。与项目页
   *  FilterBar 一致：自带 `px-6 py-2 border-b`，内容区随之改用 pt-4（贴着分隔线）。 */
  belowHeader?: ReactNode
  /** 本页任务日志源（issue #251 统一抽屉）；falsy 项自动过滤，全空时不渲染。 */
  logSources?: Array<LogSource | null | undefined | false>
}

export default function StepShell({ title, subtitle, actions, topRight, children, belowHeader, logSources }: Props) {
  return (
    <div className="fade-in flex flex-col h-full relative">
      <PageHeader
        title={title}
        subtitle={subtitle}
        actions={actions}
        topRight={topRight}
        sticky
      />
      {belowHeader}
      {/* flex column container: overflow:hidden stops page scroll; children use flex:1 to fill。
          内容区四周统一 p-6（含 belowHeader 分隔线下方也对称留白，不再顶部特例 pt-4）。 */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden p-6">
        {children}
      </div>
      {/* 页面级 footer 抽屉：全宽贴底，展开时 overlay 在内容上方（issue #251） */}
      {logSources && <TaskLogDrawer sources={logSources} />}
    </div>
  )
}
