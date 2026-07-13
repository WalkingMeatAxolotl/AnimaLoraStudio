import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useZoomPan } from '../lib/useZoomPan'

/** 可缩放单图查看器（useZoomPan 的查看器包装）。
 *
 *  填满父容器（父决定视口大小，须给出确定的宽高），内部：滚轮指针中心
 *  缩放 / 左键拖拽平移 / 双击 fit↔100% 切换 / 右下角 zoom% 角标。
 *  src 切换自动重新 fit。适用纯查看场景（FullscreenViewer /
 *  ImagePreviewModal / TagEdit 单图）；画笔类（InpaintCanvas）直接用
 *  useZoomPan 组合自己的 pointer 逻辑。
 */
export default function ZoomableImage({
  src,
  alt,
  className,
  style,
  onError,
}: {
  src: string
  alt?: string
  className?: string
  style?: React.CSSProperties
  /** img 加载失败回调（调用方切换占位 UI 用）。 */
  onError?: () => void
}) {
  const { t } = useTranslation()
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null)
  useEffect(() => { setNat(null) }, [src])
  const zp = useZoomPan({
    contentW: nat?.w ?? 0,
    contentH: nat?.h ?? 0,
    primaryButtonPans: true,
  })

  return (
    <div
      ref={zp.wrapRef}
      {...zp.handlers}
      onDoubleClick={() => (zp.zoomPct === 100 ? zp.fit() : zp.reset100())}
      title={t('common.zoomHint')}
      className={className}
      style={{
        position: 'relative',
        overflow: 'hidden',
        touchAction: 'none',
        cursor: 'grab',
        width: '100%',
        height: '100%',
        ...style,
      }}
    >
      <img
        ref={(el) => { zp.contentRef.current = el }}
        src={src}
        alt={alt}
        draggable={false}
        onLoad={(e) => setNat({
          w: e.currentTarget.naturalWidth,
          h: e.currentTarget.naturalHeight,
        })}
        onError={onError}
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          transformOrigin: '0 0',
          maxWidth: 'none',
          maxHeight: 'none',
          visibility: nat ? 'visible' : 'hidden',
        }}
      />
      <span
        className="absolute bottom-1.5 right-2 rounded bg-black/50 px-1.5 py-0.5 text-[10px] font-mono text-slate-300 pointer-events-none select-none"
      >
        {zp.zoomPct}%
      </span>
    </div>
  )
}
