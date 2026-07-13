import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { arLabel } from '../../lib/aspectRatio'
import { useZoomPan } from '../../lib/useZoomPan'

/** A normalized [0..1] crop rectangle on an image. */
export interface CropRect {
  id: string
  x: number
  y: number
  w: number
  h: number
  label: string
  fromCluster?: boolean
}

interface ImageMeta {
  /** Filename (used for key + thumb URL). */
  id: string
  name: string
  /** Source pixel size. */
  w: number
  h: number
  /** URL for the canvas background. */
  thumbUrl: string
}

export interface FreeCropEditorProps {
  image: ImageMeta
  crops: CropRect[]
  selectedId: string | null
  /** When non-null, new + resize ops maintain this w:h aspect ratio. */
  arLock: { w: number; h: number } | null
  onSelect: (id: string | null) => void
  onChange: (id: string, rect: CropRect) => void
  onCreate: (rect: Omit<CropRect, 'id' | 'label'>) => void
}

const HANDLES: readonly ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] = [
  'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w',
]

const MIN_NORM = 0.02

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** Convert AR-lock {w,h} → normalized (h per w) ratio on the rendered image. */
function normRatio(arLock: { w: number; h: number } | null, imgW: number, imgH: number): number | null {
  if (!arLock) return null
  return (arLock.h / arLock.w) * (imgW / imgH)
}

interface DragState {
  mode: 'move' | 'resize' | 'create'
  startX: number
  startY: number
  anchorN: { x: number; y: number }
  origRect: CropRect | null
  handle: typeof HANDLES[number] | null
}

/** Apply a handle drag (dxN, dyN normalized) to a rect; respects arLock.
 *
 *  Bug history: clamping w/h independently to [0,1] after AR lock broke the
 *  AR (e.g. AR=1:1 on a 2:3 image would round-trip to 2:3 once the user
 *  dragged out of bounds — the locked rect was silently truncated to the
 *  whole canvas). Fix: when arLock is set, scale the rect uniformly toward
 *  the anchored corner so it always fits AND keeps its ratio.
 *
 *  Exported for unit testing the AR invariant.
 */
export function applyResize(
  rect: CropRect,
  handle: typeof HANDLES[number],
  dxN: number,
  dyN: number,
  arLock: { w: number; h: number } | null,
  imgW: number,
  imgH: number,
): CropRect {
  let { x, y, w, h } = rect
  if (handle.includes('w')) { x += dxN; w -= dxN }
  if (handle.includes('e')) { w += dxN }
  if (handle.includes('n')) { y += dyN; h -= dyN }
  if (handle.includes('s')) { h += dyN }
  if (w < MIN_NORM) {
    if (handle.includes('w')) x -= MIN_NORM - w
    w = MIN_NORM
  }
  if (h < MIN_NORM) {
    if (handle.includes('n')) y -= MIN_NORM - h
    h = MIN_NORM
  }
  if (arLock) {
    const r = normRatio(arLock, imgW, imgH)!
    const isCorner = handle.length === 2
    const drivenByW = isCorner
      ? Math.abs(w - rect.w) >= Math.abs(h - rect.h) * r
      : (handle === 'e' || handle === 'w')
    if (drivenByW) {
      h = w * r
      if (handle.includes('n')) y = (rect.y + rect.h) - h
    } else {
      w = h / r
      if (handle.includes('w')) x = (rect.x + rect.w) - w
    }
    // AR-preserving fit: anchored corner stays put; shrink uniformly if any
    // edge would leave the canvas. Edge handles anchor at the opposite edge;
    // corner handles anchor at the diagonally opposite corner.
    const anchorX = handle.includes('w') ? rect.x + rect.w : rect.x
    const anchorY = handle.includes('n') ? rect.y + rect.h : rect.y
    const maxByX = handle.includes('w') ? anchorX / w : (1 - anchorX) / w
    const maxByY = handle.includes('n') ? anchorY / h : (1 - anchorY) / h
    const factor = Math.min(1, maxByX, maxByY)
    if (factor < 1) {
      w *= factor
      h *= factor
      if (handle.includes('w')) x = anchorX - w
      else x = anchorX
      if (handle.includes('n')) y = anchorY - h
      else y = anchorY
    }
    // Final hygienic clamp — w/h already fit by construction, but guard
    // against rounding drift pushing x or y a hair past the edge.
    x = clamp(x, 0, 1 - w)
    y = clamp(y, 0, 1 - h)
    return { ...rect, x, y, w, h }
  }
  // Free mode: independent clamps are safe (no AR to preserve)
  x = clamp(x, 0, 1 - w)
  y = clamp(y, 0, 1 - h)
  w = clamp(w, MIN_NORM, 1 - x)
  h = clamp(h, MIN_NORM, 1 - y)
  return { ...rect, x, y, w, h }
}

/** Interactive crop editor: drag-create, click-select, 8-handle resize, AR lock.
 *
 *  Sizing model — img.naturalSize is the source of truth.
 *
 *  History: v1 computed `renderW/H` from `image.w/h` (React state) and used
 *  `background-image + background-size: cover`. After an in-place crop output
 *  overwrote `preprocess/X.png`, the new small image got `cover`-stretched
 *  into a canvas sized from stale state — what the user saw was bigger than
 *  the source rect even though the file on disk was correct.
 *
 *  v2 tried `<img maxWidth=min(100%, Npx) maxHeight=min(100%, Npx)>` with
 *  width:auto/height:auto, letting the browser pick intrinsic size. But
 *  `max-height: 100%` on inline-block inside flex containers doesn't reliably
 *  bound the image height — tall images overflowed and the flex parent's
 *  overflow-hidden cropped the bottom.
 *
 *  v3 (current): wrapper has `aspect-ratio: W/H` driven by `img.naturalSize`
 *  (filled in `onLoad`); `max-width/max-height: min(100%, props)` give upper
 *  bounds; img fills the wrapper (`width: 100%; height: 100%; object-fit:
 *  contain`). Since wrapper AR matches img AR, contain == fill — no letterbox
 *  — and the wrapper rect aligns 1:1 with what the browser drew. mouse math
 *  reads `canvasRef.getBoundingClientRect()` live. `image.w/h` is only used
 *  as a pre-load AR hint (avoid 0-size flash) and for the rect's px label
 *  ("400×600 px" = norm × source). Display = bytes, always.
 */
export default function FreeCropEditor({
  image,
  crops,
  selectedId,
  arLock,
  onSelect,
  onChange,
  onCreate,
}: FreeCropEditorProps) {
  const { t } = useTranslation()
  const canvasRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<DragState | null>(null)
  const [hoverId, setHoverId] = useState<string | null>(null)
  const [draft, setDraft] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  // Browser-reported intrinsic size from <img> onLoad. Authoritative once set
  // — supersedes the (potentially stale) image.w/h prop. Reset when thumbUrl
  // changes (user picks a different image, or in-place overwrite invalidates
  // via mtime cache-buster), so a tall->wide switch doesn't briefly render at
  // the old AR.
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  useEffect(() => { setNatural(null) }, [image.thumbUrl])

  // 视口（zoom / pan / fit）：size 模式 —— 缩放通过改 wrapper 宽高实现，
  // rect 的 % 定位自动跟随、px 单位的边框 / handle / label 不缩放；
  // 所有拖拽数学基于 getBoundingClientRect 实时读取，缩放下自动正确。
  // 左键留给画框（primaryButtonPans 缺省 false），pan = 空格 / 中键。
  const zp = useZoomPan({
    contentW: (natural ?? { w: image.w, h: image.h }).w,
    contentH: (natural ?? { w: image.w, h: image.h }).h,
    applyMode: 'size',
  })

  const onPointerDown = (
    e: React.MouseEvent,
    mode: DragState['mode'],
    rect: CropRect | null,
    handle: DragState['handle'] | null,
  ) => {
    // 空格 / 中键 = pan 手势，让路给视口（事件冒泡到视口容器处理）
    if (e.button === 1 || zp.spaceRef.current) return
    e.preventDefault()
    e.stopPropagation()
    const cv = canvasRef.current?.getBoundingClientRect()
    if (!cv || cv.width === 0 || cv.height === 0) return
    dragRef.current = {
      mode,
      startX: e.clientX,
      startY: e.clientY,
      anchorN: {
        x: clamp((e.clientX - cv.left) / cv.width, 0, 1),
        y: clamp((e.clientY - cv.top) / cv.height, 0, 1),
      },
      origRect: rect ? { ...rect } : null,
      handle: handle || null,
    }
    if (mode === 'create') {
      const a = dragRef.current.anchorN
      setDraft({ x: a.x, y: a.y, w: 0, h: 0 })
    }
  }

  // Global move/up handlers
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = dragRef.current
      if (!d) return
      const cv = canvasRef.current?.getBoundingClientRect()
      if (!cv || cv.width === 0 || cv.height === 0) return
      const dx = e.clientX - d.startX
      const dy = e.clientY - d.startY
      const dxN = dx / cv.width
      const dyN = dy / cv.height
      if (d.mode === 'move' && d.origRect) {
        const r = d.origRect
        const next: CropRect = {
          ...r,
          x: clamp(r.x + dxN, 0, 1 - r.w),
          y: clamp(r.y + dyN, 0, 1 - r.h),
        }
        onChange(r.id, next)
        // Re-anchor each frame: prevents the "magnetic stick" when the rect
        // hits a canvas edge — without this, the cursor's unclamped delta
        // accumulates while the rect stays pinned, so reversing direction
        // has to first unwind the whole accumulated delta before any visible
        // motion. Re-anchoring keeps delta = last-frame-to-now.
        d.origRect = next
        d.startX = e.clientX
        d.startY = e.clientY
      } else if (d.mode === 'resize' && d.origRect && d.handle) {
        const next = applyResize(d.origRect, d.handle, dxN, dyN, arLock, image.w, image.h)
        onChange(d.origRect.id, next)
        // Same anti-magnetic re-anchor as move. Especially important under
        // AR lock where saturation is more aggressive (one axis can cap the
        // other), so the user feels "stuck" trying to size past max.
        d.origRect = next
        d.startX = e.clientX
        d.startY = e.clientY
      } else if (d.mode === 'create') {
        const a = d.anchorN
        const curX = clamp((e.clientX - cv.left) / cv.width, 0, 1)
        const curY = clamp((e.clientY - cv.top) / cv.height, 0, 1)
        let dw = Math.abs(curX - a.x)
        let dh = Math.abs(curY - a.y)
        // AR-lock: link dw and dh, then cap by the room available from the
        // anchor in the drag direction — uniform scale, never independent
        // clamp (independent clamp would silently turn 1:1 into the source
        // image's AR when the user dragged past an edge).
        if (arLock) {
          const r = normRatio(arLock, image.w, image.h)!
          if (dw * r > dh) dh = dw * r
          else dw = dh / r
          const maxW = curX < a.x ? a.x : 1 - a.x
          const maxH = curY < a.y ? a.y : 1 - a.y
          const factor = Math.min(1, maxW > 0 ? maxW / dw : 0, maxH > 0 ? maxH / dh : 0)
          if (factor < 1) {
            dw *= factor
            dh *= factor
          }
        } else {
          dw = Math.min(dw, 1)
          dh = Math.min(dh, 1)
        }
        const sx = curX < a.x ? a.x - dw : a.x
        const sy = curY < a.y ? a.y - dh : a.y
        setDraft({
          x: clamp(sx, 0, 1 - dw),
          y: clamp(sy, 0, 1 - dh),
          w: dw,
          h: dh,
        })
      }
    }
    const onUp = () => {
      const d = dragRef.current
      if (!d) return
      if (d.mode === 'create' && draft && draft.w > MIN_NORM && draft.h > MIN_NORM) {
        onCreate(draft)
      }
      dragRef.current = null
      setDraft(null)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [draft, onChange, onCreate, arLock, image.w, image.h])

  const selectedRect = selectedId ? crops.find((c) => c.id === selectedId) : null

  // Wrapper AR — prefer browser-reported intrinsic, fall back to prop hint
  // pre-load. The fallback only matters for the first paint of each new image;
  // once onLoad fires, naturalWidth/Height takes over.
  const arSrc = natural ?? { w: image.w, h: image.h }
  const arCss = `${arSrc.w} / ${arSrc.h}`

  return (
    <div className="flex flex-col w-full h-full overflow-hidden min-w-0 min-h-0 gap-1.5">
      <div
        ref={zp.wrapRef}
        className="relative flex-1 min-h-0 overflow-hidden"
        style={{ touchAction: 'none' }}
        onPointerDown={(e) => {
          if (zp.panPointerDown(e)) e.currentTarget.setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => { zp.panPointerMove(e) }}
        onPointerUp={() => zp.endPan()}
        onPointerCancel={() => zp.endPan()}
      >
      <div
        ref={(el) => {
          canvasRef.current = el
          zp.contentRef.current = el
        }}
        className="cropper-canvas"
        style={{
          // useZoomPan size 模式控制 width/height（= 原图尺寸 × scale）+
          // translate 平移；rect overlays（% 定位）随 wrapper 缩放 1:1 对齐，
          // px 边框 / handle 不缩放。arCss 仅作 onLoad 前的首帧尺寸提示。
          position: 'absolute',
          left: 0,
          top: 0,
          aspectRatio: arCss,
        }}
        onMouseDown={(e) => {
          // 空格 / 中键 = pan（冒泡给视口容器）；create 只响应纯左键
          if (e.button !== 0 || zp.spaceRef.current) return
          // create on blank-canvas click. img has pointer-events:none so click
          // lands here; rect children stopPropagation in their own handlers.
          if (e.target === canvasRef.current || (e.target as HTMLElement).tagName === 'IMG') {
            onSelect(null)
            onPointerDown(e, 'create', null, null)
          }
        }}
      >
        <img
          src={image.thumbUrl}
          alt=""
          draggable={false}
          onLoad={(e) => {
            const t = e.currentTarget
            if (t.naturalWidth > 0 && t.naturalHeight > 0) {
              setNatural({ w: t.naturalWidth, h: t.naturalHeight })
            }
          }}
          style={{
            display: 'block',
            width: '100%',
            height: '100%',
            // Wrapper AR matches img AR (once loaded) → contain == fill, no
            // letterbox. The brief moment before onLoad uses the prop-hint AR,
            // which may differ slightly from the actual file; contain keeps
            // the image undistorted during that flash.
            objectFit: 'contain',
            userSelect: 'none',
            pointerEvents: 'none',
          }}
        />
        {/* dim outside selected rect */}
        {selectedRect && (
          <div className="cropper-dim">
            <div className="dim-piece" style={{ left: 0, top: 0, right: 0, height: `${selectedRect.y * 100}%` }} />
            <div className="dim-piece" style={{ left: 0, top: `${selectedRect.y * 100}%`, width: `${selectedRect.x * 100}%`, height: `${selectedRect.h * 100}%` }} />
            <div className="dim-piece" style={{ left: `${(selectedRect.x + selectedRect.w) * 100}%`, top: `${selectedRect.y * 100}%`, right: 0, height: `${selectedRect.h * 100}%` }} />
            <div className="dim-piece" style={{ left: 0, top: `${(selectedRect.y + selectedRect.h) * 100}%`, right: 0, bottom: 0 }} />
          </div>
        )}

        {crops.map((c, i) => {
          const isSel = c.id === selectedId
          const outW = Math.round(c.w * image.w)
          const outH = Math.round(c.h * image.h)
          const ratioLabel = arLabel(c.w * image.w, c.h * image.h)
          const cls = [
            'crop-rect',
            isSel ? 'is-selected' : '',
            hoverId === c.id ? 'is-hover' : '',
            c.fromCluster ? 'from-cluster' : '',
          ].filter(Boolean).join(' ')
          return (
            <div
              key={c.id}
              className={cls}
              style={{
                left: `${c.x * 100}%`,
                top: `${c.y * 100}%`,
                width: `${c.w * 100}%`,
                height: `${c.h * 100}%`,
              }}
              onMouseEnter={() => setHoverId(c.id)}
              onMouseLeave={() => setHoverId(null)}
              onMouseDown={(e) => {
                onSelect(c.id)
                onPointerDown(e, 'move', c, null)
              }}
            >
              <div className="crop-rect-label">
                <span className="num">#{i + 1}</span>
                <span>{c.label}</span>
                <span className="ar font-mono">{ratioLabel}</span>
              </div>
              {isSel && <div className="crop-rect-info font-mono">{outW}×{outH}</div>}
              {isSel && (
                <>
                  <div className="grid-v" style={{ left: '33.3%' }} />
                  <div className="grid-v" style={{ left: '66.6%' }} />
                  <div className="grid-h" style={{ top: '33.3%' }} />
                  <div className="grid-h" style={{ top: '66.6%' }} />
                </>
              )}
              {isSel && HANDLES.map((h) => (
                <div
                  key={h}
                  className={`handle handle-${h}`}
                  onMouseDown={(e) => onPointerDown(e, 'resize', c, h)}
                />
              ))}
            </div>
          )
        })}

        {draft && draft.w > 0 && (
          <div
            className="crop-rect is-draft"
            style={{
              left: `${draft.x * 100}%`,
              top: `${draft.y * 100}%`,
              width: `${draft.w * 100}%`,
              height: `${draft.h * 100}%`,
            }}
          >
            <div className="crop-rect-info font-mono">
              {Math.round(draft.w * image.w)}×{Math.round(draft.h * image.h)}
              <span> · {arLabel(draft.w * image.w, draft.h * image.h)}</span>
            </div>
          </div>
        )}
      </div>
      </div>

      {/* readout 细条（对齐涂抹页）：zoom / 适应 / 100% / 操作提示 */}
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
        <span>{image.w}×{image.h}</span>
        <span className="text-fg-disabled">{t('common.cropZoomHint')}</span>
      </div>
    </div>
  )
}
