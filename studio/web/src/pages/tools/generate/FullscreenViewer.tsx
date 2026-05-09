import { useEffect } from 'react'

/** 全屏图片 modal：双击 grid cell / cell action 触发。
 *
 * - 背景半透明遮罩，居中显示原图（object-contain）
 * - ESC / 点击遮罩关闭
 * - 不开新窗口（之前是 window.open，频繁评测时切换 tab 麻烦）
 */
export default function FullscreenViewer({
  src, alt, caption, onClose,
}: {
  src: string
  alt?: string
  caption?: string
  onClose: () => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

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
          ESC / 点击遮罩关闭 · 在新窗口打开请按右键
        </div>
      </div>
    </div>
  )
}
