import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react'
import { useTranslation } from 'react-i18next'

/** 一笔涂抹 / mask 笔画。坐标 / 直径都是**原图像素**单位 —— 视图缩放只影响
 *  显示，笔画数据与 zoom 无关，离屏重放（保存全部）才能与画布所见一致。 */
export interface InpaintStroke {
  color: string
  /** 笔刷直径（原图 px）。 */
  size: number
  /** 1 = 实心；<1 时边缘按 size*(1-hardness)/2 的半径 blur 软化。 */
  hardness: number
  /** mask 模式的橡皮擦（destination-out）。paint 模式不使用。 */
  erase?: boolean
  points: { x: number; y: number }[]
}

export type InpaintMode = 'paint' | 'mask'

export interface InpaintCanvasHandle {
  /** 当前图 + 全部涂抹笔画合成导出 PNG。图片未加载完成时返回 null。 */
  exportBlob: () => Promise<Blob | null>
  /** mask 层导出灰度 PNG（255=学 0=不学）+ 覆盖率。
   *  mask 为空（全学）→ null（调用方应 DELETE 而不是写全白文件）。 */
  exportMaskBlob: () => Promise<{ blob: Blob; coverage: number } | null>
}

/** mask 在画布上的显示色（导出只看 alpha，色值无所谓）。 */
const MASK_COLOR = '#ff2d2d'
const MASK_VIEW_ALPHA = 0.45

function strokePath(ctx: CanvasRenderingContext2D, s: InpaintStroke, color?: string): void {
  const c = color ?? s.color
  ctx.strokeStyle = c
  ctx.fillStyle = c
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

type ScratchRef = { current: HTMLCanvasElement | null }

/** 画一笔（含软边：经复用 scratch canvas + blur filter 合成）。 */
function drawOneStroke(
  ctx: CanvasRenderingContext2D,
  s: InpaintStroke,
  color: string,
  holder: ScratchRef,
): void {
  if (s.hardness >= 1 || s.points.length === 0) {
    strokePath(ctx, s, color)
    return
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
    strokePath(ctx, s, color)
    return
  }
  sctx.clearRect(0, 0, w, h)
  strokePath(sctx, s, color)
  const blur = (s.size * (1 - s.hardness)) / 2
  ctx.save()
  ctx.filter = `blur(${blur}px)`
  ctx.drawImage(holder.current, 0, 0)
  ctx.restore()
}

/** 重放涂抹笔画（场景 1，覆盖像素）。 */
export function drawStrokes(
  ctx: CanvasRenderingContext2D,
  strokes: InpaintStroke[],
  scratchRef?: ScratchRef,
): void {
  const holder = scratchRef ?? { current: null }
  for (const s of strokes) drawOneStroke(ctx, s, s.color, holder)
}

/** 重放 mask 笔画到 mask 层（erase 走 destination-out）。 */
function drawMaskStrokes(
  ctx: CanvasRenderingContext2D,
  strokes: InpaintStroke[],
  scratchRef: ScratchRef,
): void {
  for (const s of strokes) {
    ctx.save()
    ctx.globalCompositeOperation = s.erase ? 'destination-out' : 'source-over'
    drawOneStroke(ctx, s, MASK_COLOR, scratchRef)
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

/** 服务器灰度 mask（255=学 0=不学）→ 画布 mask 层位图（红色 + alpha=不学度）。 */
async function loadMaskBase(url: string, w: number, h: number): Promise<HTMLCanvasElement | null> {
  let img: HTMLImageElement
  try {
    img = await loadImage(url)
  } catch {
    return null // 404 = 无 mask
  }
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')
  if (!ctx) return null
  ctx.drawImage(img, 0, 0, w, h)
  const data = ctx.getImageData(0, 0, w, h)
  const px = data.data
  for (let i = 0; i < px.length; i += 4) {
    const v = px[i] // 灰度值（R 通道）
    px[i] = 255
    px[i + 1] = 45
    px[i + 2] = 45
    px[i + 3] = 255 - v
  }
  ctx.putImageData(data, 0, 0)
  return c
}

/** 重建 mask 层：底图（服务器已有 mask）+ 本地笔画。 */
function rebuildMaskLayer(
  layer: HTMLCanvasElement,
  base: HTMLCanvasElement | null,
  strokes: InpaintStroke[],
  scratchRef: ScratchRef,
): void {
  const ctx = layer.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, layer.width, layer.height)
  if (base) ctx.drawImage(base, 0, 0)
  drawMaskStrokes(ctx, strokes, scratchRef)
}

/** mask 层 → 灰度 PNG（255=学 0=不学）+ 覆盖率。全空 → null。 */
async function maskLayerToGray(
  layer: HTMLCanvasElement,
): Promise<{ blob: Blob; coverage: number } | null> {
  const ctx = layer.getContext('2d')
  if (!ctx) return null
  const data = ctx.getImageData(0, 0, layer.width, layer.height)
  const px = data.data
  let sum = 0
  for (let i = 0; i < px.length; i += 4) {
    const a = px[i + 3]
    sum += a
    const v = 255 - a
    px[i] = v
    px[i + 1] = v
    px[i + 2] = v
    px[i + 3] = 255
  }
  const n = px.length / 4
  const coverage = sum / 255 / n
  if (sum === 0) return null
  const out = document.createElement('canvas')
  out.width = layer.width
  out.height = layer.height
  const octx = out.getContext('2d')
  if (!octx) return null
  octx.putImageData(data, 0, 0)
  const blob = await new Promise<Blob | null>((resolve) => {
    out.toBlob((b) => resolve(b), 'image/png')
  })
  return blob ? { blob, coverage } : null
}

/** 离屏重建 mask 并导出（「保存全部」对非活动图用）。null = mask 为空。 */
export async function renderMaskBlob(
  maskBaseUrl: string | null,
  w: number,
  h: number,
  strokes: InpaintStroke[],
): Promise<{ blob: Blob; coverage: number } | null> {
  const layer = document.createElement('canvas')
  layer.width = w
  layer.height = h
  const base = maskBaseUrl ? await loadMaskBase(maskBaseUrl, w, h) : null
  rebuildMaskLayer(layer, base, strokes, { current: null })
  return await maskLayerToGray(layer)
}

/** 离屏重放涂抹：加载原图 → 重放笔画 → PNG blob。 */
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
 *  两个数据面（双桶受控）：
 *  - 涂抹笔画（strokes）：直接覆盖像素，重绘顺序 img → strokes。
 *  - mask 笔画（maskStrokes）+ 服务器底图（maskBaseUrl）：合成到独立
 *    maskLayer（红色 alpha 位图），主画布最后以半透明叠加显示。
 *
 *  绘制中在主画布增量画预览段（mask 橡皮擦以半透明白示意），pointerup
 *  提交后由 props 变化触发全量重绘校正。视图状态放 ref 直改 DOM
 *  transform —— pan/zoom 高频路径不走 React 渲染。
 */
const InpaintCanvas = forwardRef<
  InpaintCanvasHandle,
  {
    imageUrl: string
    imageW: number
    imageH: number
    mode: InpaintMode
    strokes: InpaintStroke[]
    maskStrokes: InpaintStroke[]
    /** 服务器已有 mask 的 URL；null = 无底图（含本地「清除 mask」后）。 */
    maskBaseUrl: string | null
    brush: { color: string; size: number; hardness: number }
    /** mask 模式当前是否橡皮擦。 */
    maskErase: boolean
    onStrokeEnd: (s: InpaintStroke) => void
    onMaskStrokeEnd: (s: InpaintStroke) => void
    onPickColor: (hex: string) => void
    /** mask 层变化后的覆盖率回调（0..1，缩样估算）。 */
    onMaskCoverage?: (pct: number) => void
  }
>(function InpaintCanvas(
  {
    imageUrl, imageW, imageH, mode, strokes, maskStrokes, maskBaseUrl,
    brush, maskErase, onStrokeEnd, onMaskStrokeEnd, onPickColor, onMaskCoverage,
  },
  ref,
) {
  const { t } = useTranslation()
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const cursorRef = useRef<HTMLDivElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const scratchRef = useRef<HTMLCanvasElement | null>(null)
  const maskLayerRef = useRef<HTMLCanvasElement | null>(null)
  const maskBaseRef = useRef<HTMLCanvasElement | null>(null)

  const viewRef = useRef<View>({ scale: 1, tx: 0, ty: 0 })
  const interactedRef = useRef(false)
  const spaceRef = useRef(false)
  const panRef = useRef<{ x: number; y: number } | null>(null)
  // 落笔时锁定归属（paint / mask），松手按此提交 —— 不事后按 mode 猜
  const drawingRef = useRef<{ stroke: InpaintStroke; target: InpaintMode } | null>(null)

  const [zoomPct, setZoomPct] = useState(100)
  const [loaded, setLoaded] = useState(false)
  const [maskBaseTick, setMaskBaseTick] = useState(0)
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null)

  const strokesRef = useRef(strokes)
  strokesRef.current = strokes
  const maskStrokesRef = useRef(maskStrokes)
  maskStrokesRef.current = maskStrokes
  const brushRef = useRef(brush)
  brushRef.current = brush
  const modeRef = useRef(mode)
  modeRef.current = mode
  const maskEraseRef = useRef(maskErase)
  maskEraseRef.current = maskErase

  const ensureMaskLayer = useCallback((): HTMLCanvasElement => {
    if (
      !maskLayerRef.current ||
      maskLayerRef.current.width !== imageW ||
      maskLayerRef.current.height !== imageH
    ) {
      const c = document.createElement('canvas')
      c.width = imageW
      c.height = imageH
      maskLayerRef.current = c
    }
    return maskLayerRef.current
  }, [imageW, imageH])

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
    const layer = maskLayerRef.current
    if (layer) {
      ctx.save()
      ctx.globalAlpha = MASK_VIEW_ALPHA
      ctx.drawImage(layer, 0, 0)
      ctx.restore()
    }
  }, [])

  /** 覆盖率估算：mask 层缩样到 64×64 读 alpha 均值（每笔一次，便宜）。 */
  const reportCoverage = useCallback(() => {
    if (!onMaskCoverage) return
    const layer = maskLayerRef.current
    if (!layer) {
      onMaskCoverage(0)
      return
    }
    const tiny = document.createElement('canvas')
    tiny.width = 64
    tiny.height = 64
    const tctx = tiny.getContext('2d')
    if (!tctx) return
    tctx.drawImage(layer, 0, 0, 64, 64)
    const px = tctx.getImageData(0, 0, 64, 64).data
    let sum = 0
    for (let i = 3; i < px.length; i += 4) sum += px[i]
    onMaskCoverage(sum / 255 / (64 * 64))
  }, [onMaskCoverage])

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

  // mask 底图加载（URL 变化 = 换图 / 保存后刷新 / 本地清除→null）
  useEffect(() => {
    let cancelled = false
    maskBaseRef.current = null
    if (!maskBaseUrl) {
      setMaskBaseTick((v) => v + 1)
      return
    }
    void loadMaskBase(maskBaseUrl, imageW, imageH).then((base) => {
      if (cancelled) return
      maskBaseRef.current = base
      setMaskBaseTick((v) => v + 1)
    })
    return () => {
      cancelled = true
    }
  }, [maskBaseUrl, imageW, imageH])

  // mask 层重建（底图 / 笔画变化）→ 主画布重绘 + 覆盖率
  useEffect(() => {
    const layer = ensureMaskLayer()
    rebuildMaskLayer(layer, maskBaseRef.current, maskStrokes, scratchRef)
    redraw()
    reportCoverage()
  }, [maskStrokes, maskBaseTick, ensureMaskLayer, redraw, reportCoverage])

  // 涂抹笔画变化全量重绘（undo / redo / 落笔提交 / 清除）
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
      const img = imgRef.current
      if (!canvas || !img) return null
      // 导出不含 mask overlay：干净重绘 img + strokes
      const out = document.createElement('canvas')
      out.width = canvas.width
      out.height = canvas.height
      const ctx = out.getContext('2d')
      if (!ctx) return null
      ctx.drawImage(img, 0, 0, out.width, out.height)
      drawStrokes(ctx, strokesRef.current, scratchRef)
      return await new Promise<Blob | null>((resolve) => {
        out.toBlob((b) => resolve(b), 'image/png')
      })
    },
    exportMaskBlob: async () => {
      const layer = ensureMaskLayer()
      rebuildMaskLayer(layer, maskBaseRef.current, maskStrokesRef.current, scratchRef)
      return await maskLayerToGray(layer)
    },
  }), [ensureMaskLayer])

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
      if (spaceRef.current || e.button === 1) {
        panRef.current = { x: e.clientX, y: e.clientY }
        return
      }
      if (e.button !== 0) return
      if (e.altKey && modeRef.current === 'paint') {
        pickColor(e.clientX, e.clientY)
        return
      }
      const pt = toImagePoint(e.clientX, e.clientY)
      if (!pt) return
      const b = brushRef.current
      const isMask = modeRef.current === 'mask'
      const stroke: InpaintStroke = {
        color: isMask ? MASK_COLOR : b.color,
        size: b.size,
        hardness: b.hardness,
        ...(isMask && maskEraseRef.current ? { erase: true } : {}),
        points: [pt],
      }
      drawingRef.current = { stroke, target: isMask ? 'mask' : 'paint' }
      // 单点立即可见（预览；mask 橡皮以半透明白示意，松手后全量重绘校正）
      const ctx = canvasRef.current?.getContext('2d')
      if (ctx) {
        ctx.save()
        if (isMask) {
          ctx.globalAlpha = MASK_VIEW_ALPHA
          strokePath(ctx, stroke, stroke.erase ? '#ffffff' : MASK_COLOR)
        } else {
          strokePath(ctx, stroke)
        }
        ctx.restore()
      }
    },
    [loaded, pickColor, toImagePoint],
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
      const drawing = drawingRef.current
      if (!drawing || !pt) return
      const stroke = drawing.stroke
      const prev = stroke.points[stroke.points.length - 1]
      stroke.points.push(pt)
      const ctx = canvasRef.current?.getContext('2d')
      if (ctx) {
        ctx.save()
        const isMask = drawing.target === 'mask'
        ctx.strokeStyle = isMask
          ? (stroke.erase ? '#ffffff' : MASK_COLOR)
          : stroke.color
        if (isMask) ctx.globalAlpha = MASK_VIEW_ALPHA
        ctx.lineWidth = stroke.size
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.beginPath()
        ctx.moveTo(prev.x, prev.y)
        ctx.lineTo(pt.x, pt.y)
        ctx.stroke()
        ctx.restore()
      }
    },
    [updateCursor, toImagePoint, imageW, imageH],
  )

  const endStroke = useCallback(() => {
    panRef.current = null
    const drawing = drawingRef.current
    drawingRef.current = null
    if (!drawing) return
    if (drawing.target === 'mask') onMaskStrokeEnd(drawing.stroke)
    else onStrokeEnd(drawing.stroke)
  }, [onStrokeEnd, onMaskStrokeEnd])

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
        {/* 笔刷圆圈光标（mask 模式描红 / 橡皮描白） */}
        <div
          ref={cursorRef}
          className="absolute pointer-events-none rounded-full"
          style={{
            border: mode === 'mask'
              ? `1.5px solid ${maskErase ? 'rgba(255,255,255,0.95)' : 'rgba(255,45,45,0.95)'}`
              : '1.5px solid rgba(255,255,255,0.9)',
            outline: '1px solid rgba(0,0,0,0.6)',
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
