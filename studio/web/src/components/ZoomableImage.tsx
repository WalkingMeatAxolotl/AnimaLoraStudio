import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useZoomPan } from '../lib/useZoomPan'

/** 可缩放单图查看器（useZoomPan 的查看器包装）。
 *
 *  填满父容器（父决定总尺寸），结构与涂抹 / 裁剪页统一：
 *  视口（滚轮指针中心缩放 / 左键拖拽平移 / 双击 fit↔100%）+ 底部
 *  readout 细条（zoom% / 适应窗口 / 100% / 图片尺寸）。src 切换自动重新 fit。
 *
 *  两套外观：
 *  - 默认（内嵌）：视口带圆角边框 bg-sunken，readout 用主题 token，带操作提示小字。
 *  - dark（全屏 lightbox）：视口无框，readout 黑底浅字固定色（lightbox 永远黑底，
 *    不随主题走，light 主题下也可读）；不显示通用操作提示。
 *
 *  适用纯查看场景（ImagePreviewModal / TagEdit 单图 / 测试页出图预览）；
 *  画笔类（InpaintCanvas）直接用 useZoomPan 组合。
 */
export default function ZoomableImage({
  src,
  alt,
  className,
  style,
  onError,
  dark,
  fitMaxScale,
  barExtra,
  onTap,
}: {
  src: string
  alt?: string
  className?: string
  style?: React.CSSProperties
  /** img 加载失败回调（调用方切换占位 UI 用）。 */
  onError?: () => void
  /** 全屏 lightbox 外观：视口无框 + readout 黑底浅字固定色。 */
  dark?: boolean
  /** fit 的 scale 上限（lightbox 传 1 = 小图不被放大撑满）。 */
  fitMaxScale?: number
  /** 注入 readout 条右段（caption / 计数 / 快捷键提示等）；传入时
   *  隐藏通用操作提示小字，readout 条即成为调用方的唯一底 bar。 */
  barExtra?: React.ReactNode
  /** 视口内点击（未拖拽）回调；hitImage=false = 点在图外空白（modal 关闭用）。 */
  onTap?: (hitImage: boolean) => void
}) {
  const { t } = useTranslation()
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null)
  useEffect(() => { setNat(null) }, [src])
  const zp = useZoomPan({
    contentW: nat?.w ?? 0,
    contentH: nat?.h ?? 0,
    primaryButtonPans: true,
    fitMaxScale,
    onTap,
  })

  const barBtn = dark
    ? 'px-1.5 py-0.5 rounded hover:bg-white/15 hover:text-white'
    : 'px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary'

  return (
    <div
      className={'flex flex-col w-full h-full min-h-0 ' + (dark ? '' : 'gap-1.5 ') + (className ?? '')}
      style={style}
    >
      <div
        ref={zp.wrapRef}
        {...zp.handlers}
        onDoubleClick={() => (zp.zoomPct === 100 ? zp.fit() : zp.reset100())}
        className={
          'relative flex-1 min-h-0 overflow-hidden' +
          (dark ? '' : ' rounded border border-subtle bg-sunken')
        }
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

      {/* readout 细条（与涂抹 / 裁剪页统一版式）；dark 时为 lightbox 底 bar */}
      <div
        className={
          'shrink-0 flex items-center gap-2 text-[11px] font-mono ' +
          (dark
            ? 'text-slate-400 border-t border-white/10 bg-black px-4 py-1.5'
            : 'text-fg-tertiary px-1')
        }
      >
        <span>{zp.zoomPct}%</span>
        <button type="button" className={barBtn} onClick={() => zp.fit()}>
          {t('common.zoomFit')}
        </button>
        <button type="button" className={barBtn} onClick={() => zp.reset100()}>
          100%
        </button>
        {barExtra != null ? (
          <>
            {nat && <span>{nat.w}×{nat.h}</span>}
            {barExtra}
          </>
        ) : (
          <>
            <span className="flex-1" />
            {nat && <span>{nat.w}×{nat.h}</span>}
            {!dark && <span className="text-fg-disabled">{t('common.zoomHint')}</span>}
          </>
        )}
      </div>
    </div>
  )
}
