import type { ReactNode } from 'react'

interface Props {
  title: string
  subtitle?: string
  /** Tab 导航条；如果传了 tabs 则 subtitle 不渲染（tab 取代 description 位置）。 */
  tabs?: ReactNode
  actions?: ReactNode
  sticky?: boolean
}

export default function PageHeader({ title, subtitle, tabs, actions, sticky }: Props) {
  return (
    <div className={`px-6 pt-5 pb-4 bg-canvas border-b border-subtle ${sticky ? 'sticky top-0 z-[5]' : 'relative'}`}>
      <div className="flex items-end gap-4 flex-wrap">
        <div className="flex-1 min-w-0">
          <h1 className="m-0 text-2xl font-semibold tracking-tight leading-[1.15]">{title}</h1>
          {/* tabs 在主标题下方取代 subtitle 位置；两者互斥（tabs 优先）。 */}
          {tabs ? (
            <div className="mt-3">{tabs}</div>
          ) : (
            subtitle && (
              <p className="mt-1.5 text-fg-secondary text-md max-w-[720px] m-0">{subtitle}</p>
            )
          )}
        </div>
        {actions && (
          <div className="flex gap-2 items-center">{actions}</div>
        )}
      </div>
    </div>
  )
}
