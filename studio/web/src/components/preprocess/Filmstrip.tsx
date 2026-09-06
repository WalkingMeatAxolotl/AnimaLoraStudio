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
  ariaLabel,
  header,
  itemLabel,
  emptyHint,
  renderOverlay,
}: {
  items: T[]
  activeName: string | null
  onSelect: (name: string) => void
  thumbUrl: (im: T) => string
  ariaLabel: string
  header?: ReactNode
  itemLabel?: (im: T) => string
  emptyHint?: string
  renderOverlay?: (im: T) => ReactNode
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className="flex flex-col bg-sunken/40 border border-subtle rounded h-full min-h-0 overflow-hidden"
    >
      {header && (
        <div className="shrink-0 border-b border-subtle p-1.5">
          {header}
        </div>
      )}
      {items.length === 0 ? (
        <div className="flex flex-1 min-h-0 items-center justify-center p-3 text-center text-fg-tertiary text-[11px] leading-snug">
          {emptyHint ?? ''}
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-1 overflow-y-auto p-1.5 pr-1 content-start min-h-0">
          {items.map((im) => {
            const isActive = im.name === activeName
            return (
              <div key={im.name} className="fs-thumb-sq-cell">
                <button
                  type="button"
                  onClick={() => onSelect(im.name)}
                  className={'fs-thumb-sq ' + (isActive ? 'is-active' : '')}
                  title={im.name}
                  aria-label={itemLabel?.(im) ?? im.name}
                  aria-pressed={isActive}
                >
                  {/* An image element instead of background-image lets browsers honour
                      Cache-Control + ETag reliably; CSS background-image hits the
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
      )}
    </div>
  )
}
