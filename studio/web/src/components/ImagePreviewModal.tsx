import { useId, useLayoutEffect, useRef, type KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'

import ZoomableImage from './ZoomableImage'
import '../styles/image-preview.css'

/** 全屏图片 lightbox —— 全站唯一的放大查看容器（原 FullscreenViewer 已并入）。
 *
 * - 纯黑背景；单图模式 = 可缩放视口（滚轮 / 拖拽 / 双击 fit↔100%，useZoomPan），
 *   fit 封顶 100%（小图不被拉大）
 * - 退出三入口：× 按钮 / ESC / 点画板空白处（未拖拽才算点击，点图本体不关）
 * - ←/→/↑/↓ 导航（hasX/onX 可选，屏上箭头随之显示）；Enter/Space = onAccept、
 *   Delete/Backspace = onDelete（筛选页用）
 * - 底部唯一一条 bar：zoom 控件（ZoomableImage readout）+ caption + 计数 +
 *   功能性 shortcutHint；固定黑底浅字，不随主题
 * - compareSrc 传入时改左右分屏（各自独立缩放），窄屏垂直堆叠
 */
interface Props {
  src: string
  /** 主图 alt；缺省用 caption。 */
  alt?: string
  /** 对比图：传了就改成左右 split 布局（左 src + 右 compareSrc）。窄屏时垂直堆叠。 */
  compareSrc?: string
  /** Split 布局时左侧图顶部的小 label（如 "原图"）。 */
  srcLabel?: string
  /** Split 布局时右侧图顶部的小 label（如 "处理后"）。 */
  compareLabel?: string
  caption?: string
  /** 列表中的 0-based 位置；与 total 同传时底部显示 "index+1 / total" 计数。 */
  index?: number
  total?: number
  hasPrev?: boolean
  hasNext?: boolean
  hasUp?: boolean
  hasDown?: boolean
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
  onUp?: () => void
  onDown?: () => void
  onAccept?: () => void
  onDelete?: () => void
  /** 功能性快捷键提示（如「Enter 加入训练集」）。通用操作提示（ESC / 方向键）
   *  不在此列——× 和箭头按钮本身就是可见出口。 */
  shortcutHint?: string
}

const NAV_BTN =
  'absolute z-10 rounded bg-black/45 text-slate-300 hover:text-white hover:bg-black/65'
const BAR =
  'shrink-0 flex items-center gap-2 text-[11px] font-mono text-slate-400 border-t border-white/10 bg-black px-4 py-1.5'

export default function ImagePreviewModal({
  src,
  alt,
  compareSrc,
  srcLabel,
  compareLabel,
  caption,
  index,
  total,
  hasPrev,
  hasNext,
  hasUp,
  hasDown,
  onClose,
  onPrev,
  onNext,
  onUp,
  onDown,
  onAccept,
  onDelete,
  shortcutHint,
}: Props) {
  const { t } = useTranslation()
  const panelRef = useRef<HTMLDivElement>(null)
  const captionId = useId()

  useLayoutEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const panel = panelRef.current
    panel?.focus({ preventScroll: true })
    return () => {
      // Do not steal focus from a later confirmation, or try to revive an unmounted grid cell.
      if (opener?.isConnected && panel?.contains(document.activeElement)) {
        opener.focus({ preventScroll: true })
      }
    }
  }, [])

  // A navigation button can disappear on the last image. Recover only browser fallback
  // focus, never focus already owned by another control or a later confirmation.
  useLayoutEffect(() => {
    const active = document.activeElement
    if (active === document.body || active === document.documentElement) {
      panelRef.current?.focus({ preventScroll: true })
    }
  })

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    // The preview owns its keys, not the background page. Native control defaults still run.
    event.stopPropagation()
    if (event.defaultPrevented || event.nativeEvent.isComposing) return

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault()
      return
    }
    if (event.key === 'Escape') {
      event.preventDefault()
      onClose()
      return
    }
    if (event.key === 'Tab') {
      const panel = event.currentTarget
      // This specialized surface has only button controls, including both compare readouts.
      const controls = Array.from(panel.querySelectorAll<HTMLButtonElement>('button:not(:disabled)'))
        .filter((button) => button.tabIndex >= 0 && !button.closest('[hidden], [aria-hidden="true"], [inert]'))
      const first = controls[0]
      const last = controls[controls.length - 1]
      const active = document.activeElement
      if (!first || !last || (event.shiftKey && active === first) || (!event.shiftKey && active === last)) {
        event.preventDefault()
        panel.focus({ preventScroll: true })
      } else if (active === panel) {
        event.preventDefault()
        const nextControl = event.shiftKey ? last : first
        nextControl.focus()
      }
      return
    }

    if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return
    const target = event.target instanceof Element ? event.target : null
    if (target?.closest('button, a[href], input, textarea, select, summary, [contenteditable]:not([contenteditable="false"])')) return

    let action: (() => void) | undefined
    if (event.key === 'ArrowLeft' && hasPrev) action = onPrev
    else if (event.key === 'ArrowRight' && hasNext) action = onNext
    else if (event.key === 'ArrowUp' && hasUp) action = onUp
    else if (event.key === 'ArrowDown' && hasDown) action = onDown
    else if (!event.repeat && (event.key === 'Enter' || event.key === ' ')) action = onAccept
    else if (!event.repeat && (event.key === 'Delete' || event.key === 'Backspace')) action = onDelete
    if (action) {
      event.preventDefault()
      action()
    }
  }

  const counter = index != null && total != null ? `${index + 1} / ${total}` : null
  // 底 bar 的信息段：单图模式注入 ZoomableImage readout 条右段，
  // 分屏模式由本组件渲染同款独立 bar。caption 恒渲染占位，让计数 / 提示靠右。
  const barInfo = (
    <>
      <span id={captionId} className="flex-1 min-w-0 truncate text-center text-slate-300" title={caption}>
        {caption}
      </span>
      {counter && <span className="text-slate-300">{counter}</span>}
      {shortcutHint && <span className="whitespace-nowrap">{shortcutHint}</span>}
    </>
  )

  return createPortal(
    <div
      ref={panelRef}
      role="dialog"
      aria-modal="true"
      aria-label={t('common.imagePreview')}
      aria-describedby={caption ? captionId : undefined}
      tabIndex={-1}
      className="ui-image-preview fixed inset-0 z-50 bg-black flex flex-col"
      onKeyDown={handleKeyDown}
    >
      <div className="relative flex-1 min-h-0 flex flex-col">
        <button
          type="button"
          onClick={onClose}
          className={`${NAV_BTN} top-3 right-4 px-3 py-1 text-2xl`}
          aria-label={t('common.close')}
          title={t('common.close')}
        >
          ×
        </button>
        {hasUp && onUp && (
          <button
            type="button"
            onClick={onUp}
            className={`${NAV_BTN} top-3 left-1/2 -translate-x-1/2 px-4 py-2 text-2xl`}
            aria-label={t('common.prevRow')}
            title={t('common.prevRow')}
          >
            ↑
          </button>
        )}
        {hasPrev && onPrev && (
          <button
            type="button"
            onClick={onPrev}
            className={`${NAV_BTN} left-4 top-1/2 -translate-y-1/2 px-4 py-3 text-4xl`}
            aria-label={t('common.prevImage')}
            title={t('common.prevImage')}
          >
            ‹
          </button>
        )}
        {hasNext && onNext && (
          <button
            type="button"
            onClick={onNext}
            className={`${NAV_BTN} right-4 top-1/2 -translate-y-1/2 px-4 py-3 text-4xl`}
            aria-label={t('common.nextImage')}
            title={t('common.nextImage')}
          >
            ›
          </button>
        )}
        {hasDown && onDown && (
          <button
            type="button"
            onClick={onDown}
            className={`${NAV_BTN} bottom-4 left-1/2 -translate-x-1/2 px-4 py-2 text-2xl`}
            aria-label={t('common.nextRow')}
            title={t('common.nextRow')}
          >
            ↓
          </button>
        )}
        {compareSrc ? (
          <div className="flex-1 min-h-0 flex flex-col md:flex-row gap-2 md:gap-4 p-3">
            <SplitPane src={src} label={srcLabel} altFallback={caption} onTapEmpty={onClose} />
            <SplitPane src={compareSrc} label={compareLabel} altFallback={caption} onTapEmpty={onClose} />
          </div>
        ) : (
          <ZoomableImage
            src={src}
            alt={alt ?? caption ?? 'preview'}
            dark
            fitMaxScale={1}
            onTap={(hitImage) => { if (!hitImage) onClose() }}
            barExtra={barInfo}
          />
        )}
      </div>
      {compareSrc && (counter || caption || shortcutHint) && (
        <div className={BAR}>{barInfo}</div>
      )}
    </div>,
    document.body,
  )
}

function SplitPane({
  src,
  label,
  altFallback,
  onTapEmpty,
}: { src: string; label?: string; altFallback?: string; onTapEmpty: () => void }) {
  return (
    <div className="flex-1 min-h-0 min-w-0 flex flex-col gap-1.5">
      {label && (
        <div className="shrink-0 text-center text-[11px] font-mono uppercase tracking-wider text-slate-400">
          {label}
        </div>
      )}
      <div className="flex-1 min-h-0">
        <ZoomableImage
          src={src}
          alt={label ?? altFallback ?? 'preview'}
          dark
          fitMaxScale={1}
          onTap={(hitImage) => { if (!hitImage) onTapEmpty() }}
        />
      </div>
    </div>
  )
}
