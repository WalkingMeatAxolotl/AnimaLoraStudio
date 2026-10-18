import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'

/**
 * click-toggle 弹层。给字段名 / section title / 卡片角加帮助说明，不在主流
 * UI 占空间，用户需要时才显示。
 *
 * 行为：
 * - 点 trigger 切换；点外部 / Esc / 页面滚动 / 窗口 resize 关
 * - Esc 只关弹层：capture 阶段拦截并 stopPropagation，不连带关外层 Modal / Drawer
 * - aria-expanded + aria-controls 给屏幕阅读器；trigger 自带 aria-label
 * - trigger 是 SVG i-in-circle 图标（之前用 Unicode ⓘ 字符在不同字体里
 *   垂直位置不一致跟相邻文字基线对不齐；SVG 用固定 viewBox 严格居中）
 *
 * 定位：面板 portal 到 document.body，position: fixed 按 trigger 的视口坐标摆。
 * 之前是 trigger 内 absolute + 固定 380px 宽，放进窄的 overflow:auto 容器
 * （Generate 侧栏、设置页窄列）右半截被裁掉。现在：
 * - 宽 = min(380, 视口宽 - 2×16)
 * - 默认贴 trigger 左缘、在下方；left 夹到 [16, 视口宽 - 宽 - 16]
 * - 下方放不下且上方空间更大时翻到 trigger 上方；高度再超就面板内滚动
 *
 * 不做的事（YAGNI；将来真有别的 trigger 需求再扩）：
 * - 不支持 hover 触发（手机不友好）
 * - 打开期间不跟随滚动重新定位（直接关，避免面板跟锚点脱节）
 */
interface InfoButtonProps {
  children: ReactNode
  ariaLabel?: string
}

/** 面板理想宽度；视口窄时收到 innerWidth - 2×VIEWPORT_GUTTER。 */
export const INFO_PANEL_WIDTH = 380
/** 面板离视口边缘的最小距离（跟页面 16px 侧边距一致）。 */
export const VIEWPORT_GUTTER = 16
/** 面板与 trigger 的间距。 */
const PANEL_OFFSET = 8

interface PanelPosition {
  top: number
  left: number
  width: number
}

/** 纯函数：按 trigger 矩形 + 面板高度 + 视口尺寸算面板 fixed 坐标。 */
export function computeInfoPanelPosition(
  trigger: Pick<DOMRect, 'top' | 'bottom' | 'left'>,
  panelHeight: number,
  viewport: { width: number; height: number },
): PanelPosition {
  const width = Math.max(0, Math.min(INFO_PANEL_WIDTH, viewport.width - 2 * VIEWPORT_GUTTER))
  const maxLeft = viewport.width - width - VIEWPORT_GUTTER
  const left = Math.max(VIEWPORT_GUTTER, Math.min(trigger.left, maxLeft))

  const below = trigger.bottom + PANEL_OFFSET
  const spaceBelow = viewport.height - below - VIEWPORT_GUTTER
  const spaceAbove = trigger.top - PANEL_OFFSET - VIEWPORT_GUTTER
  let top = below
  if (panelHeight > spaceBelow && spaceAbove > spaceBelow) {
    top = Math.max(VIEWPORT_GUTTER, trigger.top - PANEL_OFFSET - panelHeight)
  }
  return { top, left, width }
}

function InfoIcon() {
  return (
    <svg
      width={12}
      height={12}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx={8} cy={8} r={6.5} />
      <line x1={8} y1={7.5} x2={8} y2={11.5} />
      <circle cx={8} cy={4.8} r={0.6} fill="currentColor" stroke="none" />
    </svg>
  )
}

export function InfoButton({ children, ariaLabel }: InfoButtonProps) {
  const { t } = useTranslation()
  const resolvedAriaLabel = ariaLabel ?? t('infoButton.label')
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<PanelPosition | null>(null)
  const wrapRef = useRef<HTMLSpanElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const panelId = useId()

  const close = useCallback(() => {
    setOpen(false)
    setPos(null)
  }, [])

  // 先以 visibility:hidden 渲染一帧量高度，再算坐标；都在 paint 前完成，不闪。
  useLayoutEffect(() => {
    if (!open) return
    const trigger = triggerRef.current
    const panel = panelRef.current
    if (!trigger || !panel) return
    setPos(computeInfoPanelPosition(
      trigger.getBoundingClientRect(),
      panel.getBoundingClientRect().height,
      { width: window.innerWidth, height: window.innerHeight },
    ))
  }, [open])

  useEffect(() => {
    if (!open) return
    const inside = (target: EventTarget | null) => (
      target instanceof Node && (
        wrapRef.current?.contains(target) || panelRef.current?.contains(target)
      )
    )
    const onDown = (e: MouseEvent) => {
      if (!inside(e.target)) close()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // 只关最上层：外层 Modal / Drawer 的 Esc 监听在 document 冒泡阶段
      e.stopPropagation()
      close()
    }
    const onScroll = (e: Event) => {
      // 面板自身内容滚动（超高时）不算
      if (panelRef.current && e.target instanceof Node && panelRef.current.contains(e.target)) return
      close()
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey, true)
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', close)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey, true)
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', close)
    }
  }, [open, close])

  return (
    <span ref={wrapRef} className="info-btn-anchor">
      <button
        ref={triggerRef}
        type="button"
        className="info-btn-trigger"
        onClick={(e) => {
          // stopPropagation：避免放在 <summary> / clickable row 里触发外层 toggle
          e.stopPropagation()
          if (open) close()
          else setOpen(true)
        }}
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        aria-label={resolvedAriaLabel}
      >
        <InfoIcon />
      </button>
      {open && createPortal(
        <div
          ref={panelRef}
          id={panelId}
          className="info-btn-panel"
          role="dialog"
          aria-label={resolvedAriaLabel}
          style={pos
            ? { top: pos.top, left: pos.left, width: pos.width }
            : { top: 0, left: 0, visibility: 'hidden' }}
        >
          {children}
        </div>,
        document.body,
      )}
    </span>
  )
}
