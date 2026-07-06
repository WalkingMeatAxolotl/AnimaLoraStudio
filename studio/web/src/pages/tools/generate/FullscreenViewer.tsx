import { useEffect, useMemo } from 'react'
import { useTranslation } from 'react-i18next'

/** 全屏图片 modal：双击 grid cell / cell action 触发。
 *
 * - 背景半透明遮罩，居中显示原图（object-contain）
 * - ESC / 点击遮罩关闭
 * - 不开新窗口（之前是 window.open，频繁评测时切换 tab 麻烦）
 * - 方向键 + 屏上箭头切换邻居（PR #64 + P1-G）：caller 喂 hasX / onX；
 *   shortcutHint 不传时按当前可用方向动态拼接（避免在单行 / 单列 / 角落格上撒谎）
 */
export default function FullscreenViewer({
  src, alt, caption, index, total, onClose,
  hasPrev, hasNext, hasUp, hasDown,
  onPrev, onNext, onUp, onDown,
  shortcutHint,
}: {
  src: string
  alt?: string
  caption?: string
  /** 列表中的 0-based 位置；与 total 同传时底部显示 "index+1 / total" 计数。 */
  index?: number
  total?: number
  onClose: () => void
  hasPrev?: boolean
  hasNext?: boolean
  hasUp?: boolean
  hasDown?: boolean
  onPrev?: () => void
  onNext?: () => void
  onUp?: () => void
  onDown?: () => void
  shortcutHint?: string
}) {
  const { t } = useTranslation()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // P3：方向键在 input / textarea / contenteditable 内不抢焦点，
      // 当前组件不带这些元素，纯防御性（未来加 caption 编辑时也安全）
      const t = e.target as HTMLElement | null
      if (t) {
        const tag = t.tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA' || t.isContentEditable) return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      } else if (e.key === 'ArrowLeft' && hasPrev && onPrev) {
        e.preventDefault()
        onPrev()
      } else if (e.key === 'ArrowRight' && hasNext && onNext) {
        e.preventDefault()
        onNext()
      } else if (e.key === 'ArrowUp' && hasUp && onUp) {
        e.preventDefault()
        onUp()
      } else if (e.key === 'ArrowDown' && hasDown && onDown) {
        e.preventDefault()
        onDown()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [hasDown, hasNext, hasPrev, hasUp, onClose, onDown, onNext, onPrev, onUp])

  const computedHint = useMemo(() => {
    if (shortcutHint) return shortcutHint
    const dirs: string[] = []
    if (hasPrev) dirs.push('←')
    if (hasNext) dirs.push('→')
    if (hasUp) dirs.push('↑')
    if (hasDown) dirs.push('↓')
    const nav = dirs.length > 0 ? t('generate.fullscreenNavPrefix', { dirs: dirs.join(' / ') }) : ''
    return `${nav}${t('generate.fullscreenCloseHint')}`
  }, [shortcutHint, hasPrev, hasNext, hasUp, hasDown, t])

  const counter = index != null && total != null ? `${index + 1} / ${total}` : null

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0,
        zIndex: 100,
        background: 'rgba(0, 0, 0, 0.85)',
        display: 'grid',
        placeItems: 'center',
        padding: 20,
      }}
    >
      <div className="flex flex-col items-center gap-2" onClick={(e) => e.stopPropagation()}>
        {hasUp && onUp && (
          <button
            type="button"
            onClick={onUp}
            className="absolute top-4 left-1/2 -translate-x-1/2 z-10 rounded bg-black/45 px-4 py-2 text-slate-300 hover:text-white hover:bg-black/65 text-2xl"
            aria-label={t('generate.prevRow')}
            title={t('generate.prevRow')}
          >
            ↑
          </button>
        )}
        {hasPrev && onPrev && (
          <button
            type="button"
            onClick={onPrev}
            className="absolute left-4 top-1/2 -translate-y-1/2 z-10 rounded bg-black/45 px-4 py-3 text-slate-300 hover:text-white hover:bg-black/65 text-4xl"
            aria-label={t('generate.prevCell')}
            title={t('generate.prevCell')}
          >
            ‹
          </button>
        )}
        {hasNext && onNext && (
          <button
            type="button"
            onClick={onNext}
            className="absolute right-4 top-1/2 -translate-y-1/2 z-10 rounded bg-black/45 px-4 py-3 text-slate-300 hover:text-white hover:bg-black/65 text-4xl"
            aria-label={t('generate.nextCell')}
            title={t('generate.nextCell')}
          >
            ›
          </button>
        )}
        {hasDown && onDown && (
          <button
            type="button"
            onClick={onDown}
            className="absolute bottom-12 left-1/2 -translate-x-1/2 z-10 rounded bg-black/45 px-4 py-2 text-slate-300 hover:text-white hover:bg-black/65 text-2xl"
            aria-label={t('generate.nextRow')}
            title={t('generate.nextRow')}
          >
            ↓
          </button>
        )}
        <img
          src={src}
          alt={alt}
          style={{
            maxWidth: 'calc(100vw - 80px)',
            maxHeight: 'calc(100vh - 100px)',
            objectFit: 'contain',
            borderRadius: 6,
          }}
        />
        {caption && (
          <div className="text-xs text-fg-secondary font-mono text-center">
            {caption}
          </div>
        )}
        {counter && (
          <div className="text-xs text-fg-secondary font-mono text-center">
            {counter}
          </div>
        )}
        <div className="text-2xs text-fg-tertiary">
          {computedHint}
        </div>
      </div>
    </div>
  )
}
