/**
 * PaneResizer —— 横向 flex 布局里两栏之间的竖向拖动分隔条（受控组件）。
 *
 * value = 左栏占 containerRef 宽度的百分比，拖动 / 方向键改 value，
 * 持久化交给父层（本组件不碰 localStorage）。
 *
 * 用 pointer capture 而不是 window 监听：拖过 canvas / img 等子元素时事件
 * 仍回到 handle 上，不会中途丢失。
 */
import type { RefObject } from 'react'

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
  className?: string
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

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
  className = '',
}: Props) {
  const dir = anchor === 'end' ? -1 : 1

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const container = containerRef.current
    if (!container || e.button !== 0) return
    const width = container.getBoundingClientRect().width
    if (width <= 0) return
    e.preventDefault()

    const startX = e.clientX
    const startPct = value
    const el = e.currentTarget
    el.setPointerCapture(e.pointerId)

    // 拖动期间全局锁光标 + 禁选中，否则划过文字会拖出选区
    const prevCursor = document.body.style.cursor
    const prevSelect = document.body.style.userSelect
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const move = (ev: PointerEvent) => {
      onChange(clamp(startPct + (dir * (ev.clientX - startX) * 100) / width, min, max))
    }
    const up = () => {
      el.releasePointerCapture?.(e.pointerId)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
      el.removeEventListener('pointercancel', up)
      document.body.style.cursor = prevCursor
      document.body.style.userSelect = prevSelect
    }
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    el.addEventListener('pointercancel', up)
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    onChange(clamp(value + dir * (e.key === 'ArrowLeft' ? -KEY_STEP : KEY_STEP), min, max))
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={ariaLabel}
      aria-valuenow={Math.round(value)}
      aria-valuemin={min}
      aria-valuemax={max}
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
