import type { ReactNode } from 'react'
import PageHeader from './PageHeader'

interface Props {
  idx: number | string
  title: string
  subtitle?: string
  actions?: ReactNode
  topRight?: ReactNode
  children: ReactNode
}

export default function StepShell({ title, subtitle, actions, topRight, children }: Props) {
  return (
    <div className="fade-in flex flex-col h-full">
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
    </div>
  )
}
