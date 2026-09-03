import type { ReactNode } from 'react'
import PageHeader from './PageHeader'
import TaskLogDrawer, { type LogSource } from './TaskLogDrawer'

interface Props {
  title: string
  subtitle?: string
  actions?: ReactNode
  topRight?: ReactNode
  children: ReactNode
  /** header 与内容区之间的全宽 Pattern（如 ListToolbar），不随内容滚动。 */
  belowHeader?: ReactNode
  /** 内容区 inset；专业画布可显式选择 none，并自行声明边界与滚动。 */
  inset?: 'page' | 'none'
  /** 本页任务日志源（issue #251 统一抽屉）；falsy 项自动过滤，全空时不渲染。 */
  logSources?: Array<LogSource | null | undefined | false>
}

export default function StepShell({
  title,
  subtitle,
  actions,
  topRight,
  children,
  belowHeader,
  inset = 'page',
  logSources,
}: Props) {
  return (
    <div className="fade-in flex flex-col h-full min-h-0 relative" data-step-shell>
      <PageHeader
        title={title}
        subtitle={subtitle}
        actions={actions}
        topRight={topRight}
      />
      {belowHeader}
      {/* Workspace shell 不参与页面滚动；子工作区声明自己的局部滚动。
          默认沿用 page inset，专业画布可显式选择 none 并自行负责边界。 */}
      <div
        className={`flex-1 min-h-0 flex flex-col overflow-hidden ${inset === 'page' ? 'p-page' : ''}`}
        data-step-shell-content
        data-inset={inset}
      >
        {children}
      </div>
      {/* 页面级 footer 抽屉：全宽贴底，展开时 overlay 在内容上方（issue #251） */}
      {logSources && <TaskLogDrawer sources={logSources} />}
    </div>
  )
}
