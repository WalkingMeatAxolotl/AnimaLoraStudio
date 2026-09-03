/**
 * PaneResizer —— 横向 flex 布局里两栏之间的竖向拖动分隔条（受控组件）。
 *
 * value = anchor 所指固定栏占 containerRef 宽度的百分比，拖动 / 方向键改 value，
 * 持久化与最终 pane geometry 交给父层（本组件不碰 localStorage）。
 *
 * 用 pointer capture 而不是 window 监听：拖过 canvas / img 等子元素时事件
 * 仍回到 handle 上，不会中途丢失。
 */
import { useEffect, useRef, type RefObject } from 'react'

interface Props {
  /** 百分比基准容器（两栏 + 本 handle 的共同父节点） */
  containerRef: RefObject<HTMLElement | null>
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  /**
   * value 量的是哪一侧的栏宽：
   *   'start'（默认）= handle 左边那栏 → 右拖变大
   *   'end'          = handle 右边那栏 → 右拖变小
   * 受控的永远是定宽的那一栏，另一栏 flex-1 吃剩余空间。
   */
  anchor?: 'start' | 'end'
  ariaLabel?: string
  /** ID of the pane whose size this separator controls. */
  ariaControls?: string
  className?: string
}

export const clampPaneValue = (value: number, min: number, max: number): number => {
  const lo = Math.min(min, max)
  const hi = Math.max(min, max)
  if (!Number.isFinite(value)) return lo
  return Math.min(hi, Math.max(lo, value))
}

export interface PanePairBounds {
  startMin: number
  endMin: number
  flexibleMin: number
}

/**
 * Repairs two persisted fixed-pane percentages while reserving a flexible middle pane.
 * Valid values pass through unchanged. If the pair exceeds the shared budget, only
 * the space above each pane's minimum is reduced, proportionally.
 */
export function normalizePanePair(
  start: number,
  end: number,
  { startMin, endMin, flexibleMin }: PanePairBounds,
): { start: number; end: number } {
  const fixedBudget = 100 - flexibleMin
  if (startMin + endMin > fixedBudget) {
    throw new RangeError('Pane minimums exceed the available width budget')
  }

  const boundedStart = clampPaneValue(start, startMin, fixedBudget - endMin)
  const boundedEnd = clampPaneValue(end, endMin, fixedBudget - startMin)
  if (boundedStart + boundedEnd <= fixedBudget) {
    return { start: boundedStart, end: boundedEnd }
  }

  const extraBudget = fixedBudget - startMin - endMin
  const startExtra = boundedStart - startMin
  const endExtra = boundedEnd - endMin
  const scale = extraBudget / (startExtra + endExtra)
  return {
    start: startMin + startExtra * scale,
    end: endMin + endExtra * scale,
  }
}

/** 方向键单步（%） */
const KEY_STEP = 2

export default function PaneResizer({
  containerRef,
  value,
  onChange,
  min = 15,
  max = 60,
  anchor = 'start',
  ariaLabel,
  ariaControls,
  className = '',
}: Props) {
  const dir = anchor === 'end' ? -1 : 1
  const boundedMin = Math.min(min, max)
  const boundedMax = Math.max(min, max)
  const boundedValue = clampPaneValue(value, boundedMin, boundedMax)
  const activeDragCleanupRef = useRef<(() => void) | null>(null)

  useEffect(() => () => activeDragCleanupRef.current?.(), [])

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const container = containerRef.current
    if (!container || e.button !== 0) return
    const width = container.getBoundingClientRect().width
    if (width <= 0) return
    e.preventDefault()

    const startX = e.clientX
    const startPct = boundedValue
    const el = e.currentTarget
    activeDragCleanupRef.current?.()
    el.setPointerCapture(e.pointerId)

    // 拖动期间全局锁光标 + 禁选中，否则划过文字会拖出选区。
    // cleanup 同时覆盖 pointercancel、capture 丢失和组件卸载，避免 body 永久锁住。
    const prevCursor = document.body.style.cursor
    const prevSelect = document.body.style.userSelect
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    let cleaned = false
    const move = (ev: PointerEvent) => {
      onChange(clampPaneValue(
        startPct + (dir * (ev.clientX - startX) * 100) / width,
        boundedMin,
        boundedMax,
      ))
    }
    const cleanup = () => {
      if (cleaned) return
      cleaned = true
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', cleanup)
      el.removeEventListener('pointercancel', cleanup)
      el.removeEventListener('lostpointercapture', cleanup)
      if (el.hasPointerCapture?.(e.pointerId)) el.releasePointerCapture?.(e.pointerId)
      document.body.style.cursor = prevCursor
      document.body.style.userSelect = prevSelect
      if (activeDragCleanupRef.current === cleanup) activeDragCleanupRef.current = null
    }
    activeDragCleanupRef.current = cleanup
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', cleanup)
    el.addEventListener('pointercancel', cleanup)
    el.addEventListener('lostpointercapture', cleanup)
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault()
      onChange(e.key === 'Home' ? boundedMin : boundedMax)
      return
    }
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    onChange(clampPaneValue(
      boundedValue + dir * (e.key === 'ArrowLeft' ? -KEY_STEP : KEY_STEP),
      boundedMin,
      boundedMax,
    ))
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={ariaLabel}
      aria-controls={ariaControls}
      aria-valuenow={Math.round(boundedValue)}
      aria-valuemin={boundedMin}
      aria-valuemax={boundedMax}
      aria-valuetext={`${Math.round(boundedValue)}%`}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
      // -mx-1 让 handle 的命中区吃掉两侧 gap，视觉上不额外撑宽栏间距
      className={
        'shrink-0 self-stretch w-2 -mx-1 rounded-full cursor-col-resize touch-none ' +
        'bg-transparent hover:bg-accent-soft active:bg-accent focus-visible:bg-accent ' +
        'transition-colors ' + className
      }
    />
  )
}
