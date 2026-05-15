import { useEffect } from 'react'

/** 全屏图片 modal：双击 grid cell / cell action 触发。
 *
 * - 背景半透明遮罩，居中显示原图（object-contain）
 * - ESC / 点击遮罩关闭
 * - 不开新窗口（之前是 window.open，频繁评测时切换 tab 麻烦）
 */
export default function FullscreenViewer({
  src, alt, caption, onClose,
  hasPrev, hasNext, hasUp, hasDown,
  onPrev, onNext, onUp, onDown,
  shortcutHint,
}: {
  src: string
  alt?: string
  caption?: string
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
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
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
            aria-label="上一行"
            title="上一行"
          >
            ↑
          </button>
        )}
        {hasPrev && onPrev && (
          <button
            type="button"
            onClick={onPrev}
            className="absolute left-4 top-1/2 -translate-y-1/2 z-10 rounded bg-black/45 px-4 py-3 text-slate-300 hover:text-white hover:bg-black/65 text-4xl"
            aria-label="左一格"
            title="左一格"
          >
            ‹
          </button>
        )}
        {hasNext && onNext && (
          <button
            type="button"
            onClick={onNext}
            className="absolute right-4 top-1/2 -translate-y-1/2 z-10 rounded bg-black/45 px-4 py-3 text-slate-300 hover:text-white hover:bg-black/65 text-4xl"
            aria-label="右一格"
            title="右一格"
          >
            ›
          </button>
        )}
        {hasDown && onDown && (
          <button
            type="button"
            onClick={onDown}
            className="absolute bottom-12 left-1/2 -translate-x-1/2 z-10 rounded bg-black/45 px-4 py-2 text-slate-300 hover:text-white hover:bg-black/65 text-2xl"
            aria-label="下一行"
            title="下一行"
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
        <div className="text-2xs text-fg-tertiary">
          {shortcutHint ?? 'ESC / 点击遮罩关闭 · 在新窗口打开请按右键'}
        </div>
      </div>
    </div>
  )
}
