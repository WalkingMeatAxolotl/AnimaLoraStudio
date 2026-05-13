import { useEffect, useRef, useState, type ReactNode } from 'react'

/**
 * ⓘ 风格的 click-toggle 弹层。给字段名 / section title / 卡片角加帮助说明，
 * 不在主流 UI 占空间，用户需要时才显示。
 *
 * 行为：
 * - 点 trigger 切换；点外部 / Esc 关
 * - aria-expanded 给屏幕阅读器；trigger 自带 aria-label
 * - 默认 trigger 是 unicode ⓘ；可通过 label prop 换
 *
 * 不做的事（YAGNI；将来真有别的 trigger 需求再扩）：
 * - 不支持 hover 触发（手机不友好）
 * - 不动态计算 placement（统一 bottom-left；视口溢出靠 max-width 收）
 * - 不接受 trigger 自定义 element（统一 ⓘ button，保持视觉一致性）
 */
interface InfoButtonProps {
  children: ReactNode
  label?: string
  ariaLabel?: string
}

export function InfoButton({ children, label = 'ⓘ', ariaLabel = '更多信息' }: InfoButtonProps) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <span ref={wrapRef} className="info-btn-anchor">
      <button
        type="button"
        className="info-btn-trigger"
        onClick={(e) => {
          // stopPropagation：避免放在 <summary> / clickable row 里触发外层 toggle
          e.stopPropagation()
          setOpen((v) => !v)
        }}
        aria-expanded={open}
        aria-label={ariaLabel}
      >
        {label}
      </button>
      {open && (
        <div className="info-btn-panel" role="dialog">
          {children}
        </div>
      )}
    </span>
  )
}
