import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useZoomPan } from '../lib/useZoomPan'

/** 可缩放单图查看器（useZoomPan 的查看器包装）。
 *
 *  填满父容器（父决定总尺寸），结构与涂抹 / 裁剪页统一：
 *  视口（rounded border bg-sunken，滚轮指针中心缩放 / 左键拖拽平移 /
 *  双击 fit↔100%）+ 底部 readout 细条（zoom% / 适应窗口 / 100% /
 *  图片尺寸 / 操作提示）。src 切换自动重新 fit。
 *
 *  适用纯查看场景（FullscreenViewer / ImagePreviewModal / TagEdit 单图 /
 *  测试页出图预览）；画笔类（InpaintCanvas）直接用 useZoomPan 组合。
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
      className={'flex flex-col gap-1.5 w-full h-full min-h-0 ' + (className ?? '')}
      style={style}
    >
      <div
        ref={zp.wrapRef}
        {...zp.handlers}
        onDoubleClick={() => (zp.zoomPct === 100 ? zp.fit() : zp.reset100())}
        className="relative flex-1 min-h-0 overflow-hidden rounded border border-subtle bg-sunken"
        style={{ touchAction: 'none', cursor: 'grab' }}
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
      </div>

      {/* readout 细条（与涂抹 / 裁剪页统一版式） */}
      <div className="shrink-0 flex items-center gap-2 text-[11px] font-mono text-fg-tertiary px-1">
        <span>{zp.zoomPct}%</span>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary"
          onClick={() => zp.fit()}
        >{t('common.zoomFit')}</button>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary"
          onClick={() => zp.reset100()}
        >100%</button>
        <span className="flex-1" />
        {nat && <span>{nat.w}×{nat.h}</span>}
        <span className="text-fg-disabled">{t('common.zoomHint')}</span>
      </div>
    </div>
  )
}
