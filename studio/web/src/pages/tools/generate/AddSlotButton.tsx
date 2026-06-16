import type { ReactNode } from 'react'

/** Sidebar 里「+ 添加 …」幽灵按钮的统一样式 —— LoRA 槽（SidebarLoras）和 XY 的
 *  「+ 添加 Y 轴」（SidebarXYAxes）共用，保证两处外观一致。
 *  font-mono + sunken 底 + subtle 边，hover 提亮文字 / 边框。 */
export default function AddSlotButton({
  onClick, children,
}: {
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className="font-mono inline-flex items-center gap-1.5 self-start"
      style={{
        border: '1px solid var(--border-subtle)',
        background: 'var(--bg-sunken)',
        borderRadius: 'var(--r-md)',
        padding: '6px 10px',
        fontSize: 12,
        color: 'var(--fg-tertiary)',
        cursor: 'pointer',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = 'var(--fg-primary)'
        e.currentTarget.style.borderColor = 'var(--border-default)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = 'var(--fg-tertiary)'
        e.currentTarget.style.borderColor = 'var(--border-subtle)'
      }}
    >
      {children}
    </button>
  )
}
