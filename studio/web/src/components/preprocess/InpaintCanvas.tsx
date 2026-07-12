import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react'
import { useTranslation } from 'react-i18next'

/** 一笔涂抹。坐标 / 直径都是**原图像素**单位 —— 视图缩放只影响显示，
 *  笔画数据与 zoom 无关，离屏重放（保存全部）才能与画布所见一致。 */
export interface InpaintStroke {
  color: string
  /** 笔刷直径（原图 px）。 */
  size: number
  /** 1 = 实心；<1 时边缘按 size*(1-hardness)/2 的半径 blur 软化。 */
  hardness: number
  points: { x: number; y: number }[]
}

export interface InpaintCanvasHandle {
  /** 当前图 + 全部笔画合成导出 PNG。图片未加载完成时返回 null。 */
  exportBlob: () => Promise<Blob | null>
}

function strokePath(ctx: CanvasRenderingContext2D, s: InpaintStroke): void {
  ctx.strokeStyle = s.color
  ctx.fillStyle = s.color
  ctx.lineWidth = s.size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  const pts = s.points
  if (pts.length === 0) return
  if (pts.length === 1) {
    ctx.beginPath()
    ctx.arc(pts[0].x, pts[0].y, s.size / 2, 0, Math.PI * 2)
    ctx.fill()
    return
  }
  ctx.beginPath()
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
  ctx.stroke()
}

/** 重放全部笔画。软边笔画经复用的 scratch canvas + blur filter 合成。 */
export function drawStrokes(
  ctx: CanvasRenderingContext2D,
  strokes: InpaintStroke[],
  scratchRef?: { current: HTMLCanvasElement | null },
): void {
  const holder = scratchRef ?? { current: null }
  for (const s of strokes) {
    if (s.hardness >= 1 || s.points.length === 0) {
      strokePath(ctx, s)
      continue
    }
    const w = ctx.canvas.width
    const h = ctx.canvas.height
    if (!holder.current || holder.current.width !== w || holder.current.height !== h) {
      holder.current = document.createElement('canvas')
      holder.current.width = w
      holder.current.height = h
    }
    const sctx = holder.current.getContext('2d')
    if (!sctx) {
      strokePath(ctx, s)
      continue
    }
    sctx.clearRect(0, 0, w, h)
    strokePath(sctx, s)
    const blur = (s.size * (1 - s.hardness)) / 2
    ctx.save()
    ctx.filter = `blur(${blur}px)`
    ctx.drawImage(holder.current, 0, 0)
    ctx.restore()
  }
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`image load failed: ${url}`))
    img.src = url
  })
}

/** 离屏重放：加载原图 → 重放笔画 → PNG blob。「保存全部」对非活动图用这个，
 *  不需要把每张图都挂成显示 canvas。 */
export async function renderInpaintedBlob(
  imageUrl: string,
  w: number,
  h: number,
  strokes: InpaintStroke[],
): Promise<Blob> {
  const img = await loadImage(imageUrl)
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 2d context unavailable')
  ctx.drawImage(img, 0, 0, w, h)
  drawStrokes(ctx, strokes)
  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('toBlob failed'))),
      'image/png',
    )
  })
}

function toHex(r: number, g: number, b: number): string {
  const c = (v: number) => v.toString(16).padStart(2, '0')
  return `#${c(r)}${c(g)}${c(b)}`
}

interface View {
  scale: number
  tx: number
  ty: number
}

/** 涂抹主画布：原图分辨率 canvas + CSS transform 视图。
 *
 *  - 笔画为受控数据（strokes prop），绘制中在画布上增量画实心预览，
 *    pointerup 提交 onStrokeEnd 后由 strokes 变化触发全量重绘（软边生效）。
 *  - 滚轮以指针为中心缩放；空格按住 / 鼠标中键拖拽平移。
 *  - Alt+点击（或 eyedropper prop）吸管取色，取的是画布当前显示的合成色。
 *  - 视图状态放 ref 直改 DOM transform —— pan/zoom 高频路径不走 React 渲染。
 */
const InpaintCanvas = forwardRef<
  InpaintCanvasHandle,
  {
    imageUrl: string
    imageW: number
    imageH: number
    strokes: InpaintStroke[]
    brush: { color: string; size: number; hardness: number }
    eyedropper: boolean
    onStrokeEnd: (s: InpaintStroke) => void
    onPickColor: (hex: string) => void
  }
>(function InpaintCanvas(
  { imageUrl, imageW, imageH, strokes, brush, eyedropper, onStrokeEnd, onPickColor },
  ref,
) {
  const { t } = useTranslation()
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const cursorRef = useRef<HTMLDivElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const scratchRef = useRef<HTMLCanvasElement | null>(null)

  const viewRef = useRef<View>({ scale: 1, tx: 0, ty: 0 })
  const interactedRef = useRef(false)
  const spaceRef = useRef(false)
  const panRef = useRef<{ x: number; y: number } | null>(null)
  const drawingRef = useRef<InpaintStroke | null>(null)

  const [zoomPct, setZoomPct] = useState(100)
  const [loaded, setLoaded] = useState(false)
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null)

  // strokes 最新引用给 redraw 闭包（避免 effect 依赖循环）
  const strokesRef = useRef(strokes)
  strokesRef.current = strokes
  const brushRef = useRef(brush)
  brushRef.current = brush

  const applyView = useCallback(() => {
    const v = viewRef.current
    const el = canvasRef.current
    if (el) {
      el.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`
    }
    setZoomPct(Math.round(v.scale * 100))
  }, [])

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    drawStrokes(ctx, strokesRef.current, scratchRef)
  }, [])

  const fit = useCallback(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const rect = wrap.getBoundingClientRect()
    if (rect.width < 10 || rect.height < 10) return
    const scale = Math.min(rect.width / imageW, rect.height / imageH) * 0.98
    viewRef.current = {
      scale,
      tx: (rect.width - imageW * scale) / 2,
      ty: (rect.height - imageH * scale) / 2,
    }
    applyView()
  }, [imageW, imageH, applyView])

  // 图片加载（imageUrl 变化 = 换图或保存后 mtime 刷新）
  useEffect(() => {
    let cancelled = false
    setLoaded(false)
    imgRef.current = null
    loadImage(imageUrl).then(
      (img) => {
        if (cancelled) return
        imgRef.current = img
        setLoaded(true)
        interactedRef.current = false
        fit()
        redraw()
      },
      () => {
        /* 失败留空画布；filmstrip 换图可恢复 */
      },
    )
    return () => {
      cancelled = true
    }
  }, [imageUrl, fit, redraw])

  // strokes 变化全量重绘（undo / redo / 落笔提交 / 清除）
  useEffect(() => {
    redraw()
  }, [strokes, redraw])

  // 容器 resize：用户没手动动过视图时保持 fit
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const ro = new ResizeObserver(() => {
      if (!interactedRef.current) fit()
    })
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [fit])

  // wheel zoom —— React onWheel 是 passive，必须手动挂 non-passive 才能 preventDefault
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = wrap.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const v = viewRef.current
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      const next = Math.min(32, Math.max(0.02, v.scale * factor))
      // 保持指针下的图像点不动
      viewRef.current = {
        scale: next,
        tx: mx - ((mx - v.tx) * next) / v.scale,
        ty: my - ((my - v.ty) * next) / v.scale,
      }
      interactedRef.current = true
      applyView()
      updateCursor(e.clientX, e.clientY)
    }
    wrap.addEventListener('wheel', onWheel, { passive: false })
    return () => wrap.removeEventListener('wheel', onWheel)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applyView])

  // 空格 = pan 修饰键（表单元素聚焦时不劫持）
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return
      const el = e.target as HTMLElement | null
      if (
        el &&
        (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
      ) {
        return
      }
      spaceRef.current = true
      e.preventDefault()
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') spaceRef.current = false
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  useImperativeHandle(ref, () => ({
    exportBlob: async () => {
      const canvas = canvasRef.current
      if (!canvas || !imgRef.current) return null
      redraw()
      return await new Promise<Blob | null>((resolve) => {
        canvas.toBlob((b) => resolve(b), 'image/png')
      })
    },
  }), [redraw])

  const toImagePoint = useCallback((clientX: number, clientY: number) => {
    const wrap = wrapRef.current
    if (!wrap) return null
    const rect = wrap.getBoundingClientRect()
    const v = viewRef.current
    return {
      x: (clientX - rect.left - v.tx) / v.scale,
      y: (clientY - rect.top - v.ty) / v.scale,
    }
  }, [])

  /** 笔刷圆圈光标跟随（ref 直改 style，不走渲染）。 */
  const updateCursor = useCallback((clientX: number, clientY: number) => {
    const cur = cursorRef.current
    const wrap = wrapRef.current
    if (!cur || !wrap) return
    const rect = wrap.getBoundingClientRect()
    const d = brushRef.current.size * viewRef.current.scale
    cur.style.left = `${clientX - rect.left - d / 2}px`
    cur.style.top = `${clientY - rect.top - d / 2}px`
    cur.style.width = `${d}px`
    cur.style.height = `${d}px`
  }, [])

  const pickColor = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current
      const pt = toImagePoint(clientX, clientY)
      if (!canvas || !pt) return
      const x = Math.round(pt.x)
      const y = Math.round(pt.y)
      if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const d = ctx.getImageData(x, y, 1, 1).data
      onPickColor(toHex(d[0], d[1], d[2]))
    },
    [toImagePoint, onPickColor],
  )

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!loaded) return
      e.currentTarget.setPointerCapture(e.pointerId)
      // pan：空格按住 / 中键
      if (spaceRef.current || e.button === 1) {
        panRef.current = { x: e.clientX, y: e.clientY }
        return
      }
      if (e.button !== 0) return
      if (eyedropper || e.altKey) {
        pickColor(e.clientX, e.clientY)
        return
      }
      const pt = toImagePoint(e.clientX, e.clientY)
      if (!pt) return
      const b = brushRef.current
      const stroke: InpaintStroke = {
        color: b.color,
        size: b.size,
        hardness: b.hardness,
        points: [pt],
      }
      drawingRef.current = stroke
      // 单点立即可见
      const ctx = canvasRef.current?.getContext('2d')
      if (ctx) strokePath(ctx, stroke)
    },
    [loaded, eyedropper, pickColor, toImagePoint],
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      updateCursor(e.clientX, e.clientY)
      const pt = toImagePoint(e.clientX, e.clientY)
      if (pt) {
        setCursorPos({
          x: Math.max(0, Math.min(imageW, Math.round(pt.x))),
          y: Math.max(0, Math.min(imageH, Math.round(pt.y))),
        })
      }
      if (panRef.current) {
        const dx = e.clientX - panRef.current.x
        const dy = e.clientY - panRef.current.y
        panRef.current = { x: e.clientX, y: e.clientY }
        viewRef.current = {
          ...viewRef.current,
          tx: viewRef.current.tx + dx,
          ty: viewRef.current.ty + dy,
        }
        interactedRef.current = true
        const el = canvasRef.current
        if (el) {
          const v = viewRef.current
          el.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`
        }
        return
      }
      const stroke = drawingRef.current
      if (!stroke || !pt) return
      const prev = stroke.points[stroke.points.length - 1]
      stroke.points.push(pt)
      // 增量画段（实心预览；软边在提交后的全量重绘生效）
      const ctx = canvasRef.current?.getContext('2d')
      if (ctx) {
        ctx.strokeStyle = stroke.color
        ctx.lineWidth = stroke.size
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.beginPath()
        ctx.moveTo(prev.x, prev.y)
        ctx.lineTo(pt.x, pt.y)
        ctx.stroke()
      }
    },
    [updateCursor, toImagePoint, imageW, imageH],
  )

  const endStroke = useCallback(() => {
    panRef.current = null
    const stroke = drawingRef.current
    drawingRef.current = null
    if (stroke) onStrokeEnd(stroke)
  }, [onStrokeEnd])

  return (
    <div className="flex flex-col h-full min-h-0 gap-1.5">
      <div
        ref={wrapRef}
        className="relative flex-1 min-h-0 overflow-hidden rounded border border-subtle bg-sunken"
        style={{ touchAction: 'none', cursor: 'none' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endStroke}
        onPointerCancel={endStroke}
        onPointerLeave={() => {
          const cur = cursorRef.current
          if (cur) {
            cur.style.width = '0px'
            cur.style.height = '0px'
          }
          setCursorPos(null)
        }}
      >
        <canvas
          ref={canvasRef}
          width={imageW}
          height={imageH}
          style={{ position: 'absolute', left: 0, top: 0, transformOrigin: '0 0' }}
        />
        {/* 笔刷圆圈光标（吸管态换成方框） */}
        <div
          ref={cursorRef}
          className="absolute pointer-events-none rounded-full"
          style={{
            border: '1.5px solid rgba(255,255,255,0.9)',
            outline: '1px solid rgba(0,0,0,0.6)',
            borderRadius: eyedropper ? '2px' : '9999px',
          }}
        />
        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center text-fg-tertiary text-sm">
            {t('preprocessInpaint.canvasLoading')}
          </div>
        )}
      </div>

      {/* readout 细条：zoom / 视图操作 / 光标像素坐标 */}
      <div className="shrink-0 flex items-center gap-2 text-[11px] font-mono text-fg-tertiary px-1">
        <span>{zoomPct}%</span>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary"
          onClick={() => {
            interactedRef.current = false
            fit()
          }}
        >{t('preprocessInpaint.zoomFit')}</button>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary"
          onClick={() => {
            const wrap = wrapRef.current
            if (!wrap) return
            const rect = wrap.getBoundingClientRect()
            const v = viewRef.current
            // 以视口中心为锚点回 100%
            const cx = rect.width / 2
            const cy = rect.height / 2
            viewRef.current = {
              scale: 1,
              tx: cx - ((cx - v.tx) * 1) / v.scale,
              ty: cy - ((cy - v.ty) * 1) / v.scale,
            }
            interactedRef.current = true
            applyView()
          }}
        >100%</button>
        <span className="flex-1" />
        {cursorPos && (
          <span>{cursorPos.x}, {cursorPos.y}</span>
        )}
        <span>{imageW}×{imageH}</span>
        <span className="text-fg-disabled">{t('preprocessInpaint.canvasHint')}</span>
      </div>
    </div>
  )
})

export default InpaintCanvas
