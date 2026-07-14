import type { ReactNode } from 'react'

/** Filmstrip 只依赖 name（选中键 + tooltip）；缩略图 URL 由调用方闭包提供。 */
export interface FilmstripItemBase {
  name: string
}

/** 预处理子页共用的左栏纵向导航（裁剪页抽出，PR-A 涂抹页复用）。
 *
 *  3-col vertical grid with square cover thumbs. Squaring is intentional —
 *  数据集横竖 AR 混排时方形 cover-crop 保持网格整齐；完整 AR 在主画布可见。
 *  page 特有的角标（裁剪 rect overlay / 涂抹已改 dot）由 renderOverlay 注入，
 *  本组件不知道 crop / stroke 概念。
 *
 *  空态也渲染同一容器 —— 调用方把本组件放进多列 grid 时，条件卸载会塌掉
 *  列布局（详裁剪页 264 图数据集的教训）。
 */
export default function Filmstrip<T extends FilmstripItemBase>({
  items,
  activeName,
  onSelect,
  thumbUrl,
  emptyHint,
  renderOverlay,
}: {
  items: T[]
  activeName: string | null
  onSelect: (name: string) => void
  thumbUrl: (im: T) => string
  emptyHint?: string
  renderOverlay?: (im: T) => ReactNode
}) {
  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center bg-sunken/40 border border-subtle rounded p-3 h-full text-center text-fg-tertiary text-[11px] leading-snug">
        {emptyHint ?? ''}
      </div>
    )
  }
  return (
    <div className="grid grid-cols-3 gap-1 overflow-y-auto pr-1 bg-sunken/40 border border-subtle rounded p-1.5 h-full content-start">
      {items.map((im) => {
        const isActive = im.name === activeName
        return (
          <div key={im.name} className="fs-thumb-sq-cell">
            <button
              onClick={() => onSelect(im.name)}
              className={'fs-thumb-sq ' + (isActive ? 'is-active' : '')}
              title={im.name}
            >
              {/* <img> instead of background-image: browsers honour Cache-Control
                  + ETag for <img src> reliably; CSS background-image hits the
                  in-memory decoded-image cache and can keep showing stale bytes
                  after an in-place crop output. object-fit: cover preserves the
                  original squared-thumbnail look. */}
              <img
                src={thumbUrl(im)}
                alt=""
                draggable={false}
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  objectPosition: 'center',
                  pointerEvents: 'none',
                }}
              />
              {renderOverlay?.(im)}
            </button>
          </div>
        )
      })}
    </div>
  )
}
