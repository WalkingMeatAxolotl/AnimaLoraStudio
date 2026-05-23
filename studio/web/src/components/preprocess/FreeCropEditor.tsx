import { useCallback, useEffect, useRef, useState } from 'react'
import { arLabel } from '../../lib/aspectRatio'

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
  /** Max width (px) the canvas may render at. */
  maxWidth?: number
  /** Max height (px) the canvas may render at. */
  maxHeight?: number
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
 *  Normalized coords let the same data outlive canvas resizes; pixel sizes are
 *  computed on the fly from image.w/h.
 */
export default function FreeCropEditor({
  image,
  crops,
  selectedId,
  arLock,
  maxWidth = 1600,
  maxHeight = 1200,
  onSelect,
  onChange,
  onCreate,
}: FreeCropEditorProps) {
  const canvasRef = useRef<HTMLDivElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<DragState | null>(null)
  const [hoverId, setHoverId] = useState<string | null>(null)
  const [draft, setDraft] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  // Container-driven sizing — measure the wrapper and fit the canvas inside.
  // Fixed maxWidth/maxHeight props would either waste vertical space on tall
  // viewports or overflow on short ones. ResizeObserver lets the canvas
  // breathe with the layout. Fallback to maxWidth/maxHeight pre-measurement
  // (e.g. first render or in jsdom where RO doesn't fire) so tests still work.
  const [containerSize, setContainerSize] = useState<{ w: number; h: number }>({
    w: maxWidth, h: maxHeight,
  })
  useEffect(() => {
    const el = containerRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          setContainerSize({ w: width, h: height })
        }
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Render dims that preserve image AR within the container box (capped by props)
  const ar = image.w / image.h
  const boxW = Math.max(50, Math.min(containerSize.w, maxWidth))
  const boxH = Math.max(50, Math.min(containerSize.h, maxHeight))
  let renderW = boxW
  let renderH = renderW / ar
  if (renderH > boxH) {
    renderH = boxH
    renderW = renderH * ar
  }

  const pxToN = useCallback(
    (dx: number, dy: number) => ({ dxN: dx / renderW, dyN: dy / renderH }),
    [renderW, renderH],
  )

  const onPointerDown = (
    e: React.MouseEvent,
    mode: DragState['mode'],
    rect: CropRect | null,
    handle: DragState['handle'] | null,
  ) => {
    e.preventDefault()
    e.stopPropagation()
    const cv = canvasRef.current?.getBoundingClientRect()
    if (!cv) return
    dragRef.current = {
      mode,
      startX: e.clientX,
      startY: e.clientY,
      anchorN: {
        x: clamp((e.clientX - cv.left) / renderW, 0, 1),
        y: clamp((e.clientY - cv.top) / renderH, 0, 1),
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
      const dx = e.clientX - d.startX
      const dy = e.clientY - d.startY
      const { dxN, dyN } = pxToN(dx, dy)
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
        const cv = canvasRef.current?.getBoundingClientRect()
        if (!cv) return
        const a = d.anchorN
        const curX = clamp((e.clientX - cv.left) / renderW, 0, 1)
        const curY = clamp((e.clientY - cv.top) / renderH, 0, 1)
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
  }, [draft, onChange, onCreate, pxToN, renderW, renderH, arLock, image.w, image.h])

  const selectedRect = selectedId ? crops.find((c) => c.id === selectedId) : null

  return (
    <div ref={containerRef} className="flex items-center justify-center w-full h-full overflow-hidden">
      <div
        ref={canvasRef}
        className="cropper-canvas"
        style={{
          width: renderW,
          height: renderH,
          backgroundImage: `url(${image.thumbUrl})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
        onMouseDown={(e) => {
          if (e.target === canvasRef.current) {
            onSelect(null)
            onPointerDown(e, 'create', null, null)
          }
        }}
      >
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
  )
}
