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
  /** 本页任务日志源（issue #251 统一抽屉）；falsy 项自动过滤，全空时不渲染。 */
  logSources?: Array<LogSource | null | undefined | false>
}

export default function StepShell({ title, subtitle, actions, topRight, children, logSources }: Props) {
  return (
    <div className="fade-in flex flex-col h-full relative">
      <PageHeader
        title={title}
        subtitle={subtitle}
        actions={actions}
        topRight={topRight}
        sticky
      />
      {/* flex column container: overflow:hidden stops page scroll; children use flex:1 to fill */}
      <div className="flex-1 min-h-0 p-6 flex flex-col overflow-hidden">
        {children}
      </div>
      {/* 页面级 footer 抽屉：全宽贴底，展开时 overlay 在内容上方（issue #251） */}
      {logSources && <TaskLogDrawer sources={logSources} />}
    </div>
  )
}
