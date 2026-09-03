import type { ReactNode } from 'react'

interface Props {
  title: string
  subtitle?: string
  /** Tab 导航条；如果传了 tabs 则 subtitle 不渲染（tab 取代 description 位置）。 */
  tabs?: ReactNode
  actions?: ReactNode
  /** 右上角 slot —— 跟 title 顶部对齐的独立位置（脱离 actions 行）。
   *  专用于 PhaseHeaderNav 等"位置必须固定在右上"的辅助导航。 */
  topRight?: ReactNode
  sticky?: boolean
}

export default function PageHeader({ title, subtitle, tabs, actions, topRight, sticky }: Props) {
  return (
    <div className={`ui-page-header px-page pt-page-start pb-section bg-canvas border-b border-subtle ${sticky ? 'sticky top-0 z-[5]' : ''}`}>
      {topRight && (
        <div className="ui-page-header-top-right">{topRight}</div>
      )}
      <div className="ui-page-header-layout">
        <div className="ui-page-header-copy">
          <h1 className="type-page-title">{title}</h1>
          {/* tabs 在主标题下方取代 subtitle 位置；两者互斥（tabs 优先）。 */}
          {tabs ? (
            <div className="mt-field">{tabs}</div>
          ) : (
            subtitle && (
              <p className="type-page-description mt-related">{subtitle}</p>
            )
          )}
        </div>
        {actions && (
          <div className="ui-page-header-actions">{actions}</div>
        )}
      </div>
    </div>
  )
}
