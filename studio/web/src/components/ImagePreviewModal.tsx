import { useEffect } from 'react'

interface Props {
  src: string
  /** 对比图：传了就改成左右 split 布局（左 src + 右 compareSrc）。窄屏时垂直堆叠。 */
  compareSrc?: string
  /** Split 布局时左侧图顶部的小 label（如 "原图"）。 */
  srcLabel?: string
  /** Split 布局时右侧图顶部的小 label（如 "处理后"）。 */
  compareLabel?: string
  caption?: string
  hasPrev?: boolean
  hasNext?: boolean
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
  onAccept?: () => void
  onDelete?: () => void
  shortcutHint?: string
}

export default function ImagePreviewModal({
  src,
  compareSrc,
  srcLabel,
  compareLabel,
  caption,
  hasPrev,
  hasNext,
  onClose,
  onPrev,
  onNext,
  onAccept,
  onDelete,
  shortcutHint,
}: Props) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      } else if (e.key === 'ArrowLeft' && hasPrev && onPrev) {
        e.preventDefault()
        onPrev()
      } else if (e.key === 'ArrowRight' && hasNext && onNext) {
        e.preventDefault()
        onNext()
      } else if ((e.key === 'Enter' || e.key === ' ') && onAccept) {
        e.preventDefault()
        onAccept()
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && onDelete) {
        e.preventDefault()
        onDelete()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [hasPrev, hasNext, onPrev, onNext, onClose, onAccept, onDelete])

  return (
    <div
      className="fixed inset-0 z-50 bg-black flex flex-col"
      onClick={onClose}
    >
      <div className="relative flex-1 min-h-0 flex items-center justify-center p-4 sm:p-6">
        <button
          onClick={(e) => {
            e.stopPropagation()
            onClose()
          }}
          className="absolute top-3 right-4 z-10 rounded bg-black/50 px-3 py-1 text-slate-300 hover:text-white text-2xl"
          aria-label="关闭"
        >
          ×
        </button>
        {hasPrev && onPrev && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onPrev()
            }}
            className="absolute left-4 top-1/2 z-10 -translate-y-1/2 text-slate-300 hover:text-white text-5xl px-4 py-3 bg-black/30 rounded"
            aria-label="上一张"
          >
            ‹
          </button>
        )}
        {hasNext && onNext && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onNext()
            }}
            className="absolute right-4 top-1/2 z-10 -translate-y-1/2 text-slate-300 hover:text-white text-5xl px-4 py-3 bg-black/30 rounded"
            aria-label="下一张"
          >
            ›
          </button>
        )}
        {compareSrc ? (
          // wrapper 不 stopPropagation — 让点击 pane 间 / pane 内空白透到 outer
          // onClose；只 img 自己 stop（点图不关）。
          <div className="w-full h-full flex flex-col md:flex-row items-stretch justify-center gap-2 md:gap-4">
            <SplitPane src={src} label={srcLabel} altFallback={caption} />
            <SplitPane src={compareSrc} label={compareLabel} altFallback={caption} />
          </div>
        ) : (
          <img
            src={src}
            alt={caption ?? 'preview'}
            onClick={(e) => e.stopPropagation()}
            className="max-w-full max-h-full object-contain"
          />
        )}
      </div>
      {(caption || shortcutHint) && (
        <div className="shrink-0 border-t border-white/10 bg-black px-4 py-2 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-xs text-slate-400">
          {caption && <div className="font-mono text-slate-300">{caption}</div>}
          {shortcutHint && <div>{shortcutHint}</div>}
        </div>
      )}
    </div>
  )
}

function SplitPane({
  src,
  label,
  altFallback,
}: { src: string; label?: string; altFallback?: string }) {
  return (
    <div className="flex-1 min-h-0 min-w-0 flex flex-col items-center justify-center gap-1.5 overflow-hidden">
      {label && (
        <div className="shrink-0 text-[11px] font-mono uppercase tracking-wider text-slate-400">
          {label}
        </div>
      )}
      <img
        src={src}
        alt={label ?? altFallback ?? 'preview'}
        onClick={(e) => e.stopPropagation()}
        className="max-w-full max-h-full object-contain"
      />
    </div>
  )
}
