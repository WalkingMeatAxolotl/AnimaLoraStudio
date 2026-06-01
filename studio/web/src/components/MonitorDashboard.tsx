/**
 * MonitorDashboard — native React training monitor
 * Replaces the monitor_smooth.html iframe.
 * Data source: GET /api/state?task_id=N 拉降采样快照 + SSE monitor_progress
 * 走 useMonitorProgress hook 做 delta merge（PR #37 增量协议）。
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import ImagePreviewModal from './ImagePreviewModal'

// ── helpers ────────────────────────────────────────────────────────────────

function fmtSec(sec: number): string {
  if (!sec || sec < 0) return '--'
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`
  return `${s}s`
}

function calcEMA(data: number[], alpha = 0.02): number[] {
  if (!data.length) return []
  const out = [data[0]]
  for (let i = 1; i < data.length; i++) out.push(alpha * data[i] + (1 - alpha) * out[i - 1])
  return out
}

function downsample<T>(arr: T[], n: number): T[] {
  if (arr.length <= n) return arr
  return Array.from({ length: n }, (_, i) => arr[Math.round((i * (arr.length - 1)) / (n - 1))])
}

// ── StatCard ───────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, tone }: {
  label: string
  value: string
  sub?: string
  tone?: 'accent' | 'ok' | 'warn'
}) {
  const colorCls = tone === 'accent' ? 'text-accent' : tone === 'ok' ? 'text-ok' : tone === 'warn' ? 'text-warn' : 'text-fg-primary'
  return (
    <div className="bg-surface border border-subtle rounded-md px-[18px] py-[14px]">
      <div className="text-xs text-fg-tertiary font-mono uppercase tracking-[0.04em] mb-1.5">
        {label}
      </div>
      <div className={`text-3xl font-semibold font-mono tabular-nums tracking-[-0.02em] leading-[1.1] ${colorCls}`}>
        {value}
      </div>
      {sub && (
        <div className="text-xs text-fg-tertiary font-mono mt-1">
          {sub}
        </div>
      )}
    </div>
  )
}

// ── SmoothControl ──────────────────────────────────────────────────────────
// EMA slider；alpha = 1 表示"不平滑"（SeriesChart 内部据此跳过 EMA）。

function SmoothControl({ alpha, setAlpha, min, max, step }: {
  alpha: number
  setAlpha: (v: number) => void
  min: number
  max: number
  step: number
}) {
  return (
    <label className="flex items-center gap-1 cursor-pointer text-xs text-fg-tertiary">
      smooth
      <input
        type="range" min={min} max={max} step={step} value={alpha}
        onChange={(e) => setAlpha(parseFloat(e.target.value))}
        style={{ width: 60, accentColor: 'var(--accent)' }}
      />
      <span className="font-mono w-[4ch] text-right">
        {alpha >= 0.999 ? 'off' : alpha.toFixed(alpha < 0.1 ? 3 : 2)}
      </span>
    </label>
  )
}

// ── SeriesChart (pure SVG) ─────────────────────────────────────────────────
// 通用的 step×value 折线图：raw + EMA smooth 双线 + xy 轴 tick。
// loss / lr / d 都复用：传 rawColor/smoothColor 自定义配色，传 yFormat 控制
// y 轴数字格式（科学计数法 vs 定点）。

function SeriesChart({ data, rawColor, smoothColor, fillColor, emaAlpha, yFormat, height, minHeight, axes = true }: {
  data: Array<{ step: number; value: number }>
  rawColor: string
  smoothColor: string
  fillColor?: string
  emaAlpha: number
  yFormat: (v: number) => string
  /** 固定像素高度（用于次要图，e.g. d value） */
  height?: number
  /** flex 模式下的最低像素高度；视口足够高时随父高度自动拉伸（用于主图，e.g. loss / lr） */
  minHeight?: number
  /** 是否绘制坐标轴 + tick label + 网格线；false 时退化为纯 sparkline 适合小高度图（d value） */
  axes?: boolean
}) {
  // ResizeObserver 测真实像素尺寸，viewBox 用真实尺寸 → SVG 1:1 渲染，
  // 文本/线宽不会被 preserveAspectRatio 非等比缩放扭曲。
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)

  useLayoutEffect(() => {
    const el = wrapperRef.current
    if (!el) return
    const measure = (w: number, h: number) => {
      if (w <= 0 || h <= 0) return
      setSize((prev) =>
        prev && Math.abs(prev.w - w) < 1 && Math.abs(prev.h - h) < 1 ? prev : { w, h },
      )
    }
    const rect = el.getBoundingClientRect()
    measure(rect.width, rect.height)
    const ro = new ResizeObserver(([entry]) => {
      const { width, height: h } = entry.contentRect
      measure(width, h)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const wrapperStyle: React.CSSProperties = height != null
    ? { height, width: '100%' }
    : { flex: 1, minHeight: minHeight ?? 0, width: '100%' }

  return (
    <div ref={wrapperRef} style={wrapperStyle}>
      {!data.length ? (
        <div className="grid place-items-center text-fg-tertiary text-sm h-full">
          等待数据…
        </div>
      ) : size ? (
        <ChartSvg
          data={data}
          W={size.w}
          H={size.h}
          rawColor={rawColor}
          smoothColor={smoothColor}
          fillColor={fillColor}
          emaAlpha={emaAlpha}
          yFormat={yFormat}
          axes={axes}
        />
      ) : null}
    </div>
  )
}

function ChartSvg({ data, W, H, rawColor, smoothColor, fillColor, emaAlpha, yFormat, axes }: {
  data: Array<{ step: number; value: number }>
  W: number
  H: number
  rawColor: string
  smoothColor: string
  fillColor?: string
  emaAlpha: number
  yFormat: (v: number) => string
  axes: boolean
}) {
  const pts = downsample(data, 600)
  const raw = pts.map((p) => p.value)
  // alpha = 1 → 跳过 EMA，纯 raw（avoid 双重曲线视觉冗余）
  const smooth = emaAlpha >= 0.999 ? raw : calcEMA(raw, emaAlpha)
  const steps = pts.map((p) => p.step)

  // sparkline 模式（axes=false）省掉 y label 左侧空间，path 填满；
  // 带 axes 时 PX 留 48 给 y tick（13pt 字宽约 7-8px × "0.0796" 6字 ≈ 46）；
  // PY 留 18 给 x tick（13pt 字高 ≈ 14，再留 4px 呼吸）。
  const PX = axes ? 48 : 0
  const PY = axes ? 18 : 2
  const RX = axes ? 8 : 0  // 右侧留白
  // y 范围按 smooth 算（无 smooth 时退化为 raw）—— raw 尖刺超出顶部会被裁掉，
  // 这是有意的：换取 smooth 信号占满高度、趋势可读。原 LossChart 同款行为。
  const refVals = emaAlpha >= 0.999 ? raw : smooth
  const minV = Math.min(...refVals), maxV = Math.max(...refVals)
  const range = maxV - minV || Math.max(Math.abs(maxV), 1e-9) * 1e-3 || 1e-9
  const x = (i: number) => PX + (i / Math.max(1, pts.length - 1)) * (W - PX - RX)
  const y = (v: number) => PY + (1 - (v - minV) / range) * (H - PY - PY)

  const smoothPath = smooth.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  const areaPath = fillColor
    ? smoothPath + ` L${x(smooth.length - 1).toFixed(1)},${H - PY} L${PX},${H - PY}Z`
    : null
  const rawPath = raw.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')

  const yTicks = [minV, (minV + maxV) / 2, maxV].map((v) => ({
    v, y: y(v), label: yFormat(v),
  }))
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
    const i = Math.round(t * Math.max(1, pts.length - 1))
    return { x: x(i), label: String(steps[i] ?? '') }
  })

  const lastY = y(smooth[smooth.length - 1])
  const showSmoothLayer = emaAlpha < 0.999

  // viewBox 与真实尺寸 1:1，省掉 preserveAspectRatio="none" 的非等比缩放——
  // 文字/线宽在真实像素下渲染，不再被父容器宽高比扭曲。
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: '100%', display: 'block' }}>
      {axes && (
        <>
          {/* axis lines */}
          <line x1={PX} y1={PY} x2={PX} y2={H - PY} stroke="var(--border-subtle)" />
          <line x1={PX} y1={H - PY} x2={W - RX} y2={H - PY} stroke="var(--border-subtle)" />
          {/* grid */}
          {[0.25, 0.5, 0.75].map((t) => (
            <line key={t} x1={PX} y1={PY + t * (H - 2 * PY)} x2={W - RX} y2={PY + t * (H - 2 * PY)}
              stroke="var(--border-subtle)" strokeDasharray="3 3" />
          ))}
        </>
      )}
      {/* area (smooth fill, optional) */}
      {areaPath && <path d={areaPath} fill={fillColor} opacity="0.5" />}
      {/* raw —— smooth 模式下淡显，无 smooth 模式下当主线 */}
      <path
        d={rawPath}
        stroke={showSmoothLayer ? rawColor : smoothColor}
        strokeWidth={showSmoothLayer ? 1 : 2}
        strokeOpacity={showSmoothLayer ? 0.45 : 1}
        fill="none"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {/* smooth */}
      {showSmoothLayer && (
        <path d={smoothPath} stroke={smoothColor} strokeWidth="2" fill="none" strokeLinejoin="round" strokeLinecap="round" />
      )}
      {/* last point */}
      <circle cx={x(smooth.length - 1)} cy={lastY} r="4" fill={smoothColor} stroke="var(--bg-surface)" strokeWidth="2" />
      {axes && (
        <>
          {/* y axis labels —— y offset +4.5 = fontSize/2 + 准基线微调，把字垂直居中到 tick */}
          {yTicks.map(({ v, y: yt, label }) => (
            <text key={v} x={PX - 4} y={yt + 4.5} fontSize="13" fill="var(--fg-tertiary)"
              fontFamily="var(--font-mono)" textAnchor="end">{label}</text>
          ))}
          {/* x axis labels —— 首/末两 tick 在 SVG 边缘上，middle 锚点会让一半字宽溢出被裁，
              改 start/end 锚点把字往内推；中间 tick 维持 middle 居中。 */}
          {xTicks.map(({ x: xt, label }, i, arr) => {
            const anchor = i === 0 ? 'start' : i === arr.length - 1 ? 'end' : 'middle'
            return (
              <text key={label} x={xt} y={H - 3} fontSize="13" fill="var(--fg-tertiary)"
                fontFamily="var(--font-mono)" textAnchor={anchor}>{label}</text>
            )
          })}
        </>
      )}
    </svg>
  )
}

// ── SampleViewer（单图 + 左右切换） ──────────────────────────────────────

function SampleViewer({ samples, taskId }: {
  samples: Array<{ path: string; step?: number }>
  taskId: number
}) {
  // 按数组原顺序铺（最新在末尾，对应训练时间轴）。多 prompt 同 step 就是相邻
  // 几个相同 step 的项，下标重复，视觉上自己传达「同一步不同 prompt」。
  const list = samples
  const [active, setActive] = useState(list.length - 1)
  const [zoomOpen, setZoomOpen] = useState(false)
  const stripRef = useRef<HTMLDivElement | null>(null)

  // 初次有图 / 新增 sample 时，仅当用户当前选中是「最末或之后」（即跟随末尾）
  // 才把 active 跟到新末尾；用户回头看早期图时不打断。
  const prevLenRef = useRef(0)
  useEffect(() => {
    if (list.length === 0) {
      setActive(0)
      prevLenRef.current = 0
      return
    }
    if (active >= prevLenRef.current - 1) {
      setActive(list.length - 1)
    }
    prevLenRef.current = list.length
  }, [list.length, active])

  // active 变化时把 strip 滚到对应缩略图（仅水平方向，不影响外层）
  useEffect(() => {
    const strip = stripRef.current
    if (!strip) return
    const target = strip.children[active] as HTMLElement | undefined
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' })
    }
  }, [active])

  if (!list.length) return (
    <div className="grid place-items-center h-[300px] text-fg-tertiary text-sm">
      等待采样图…
    </div>
  )

  const cur = list[active]
  const filename = cur.path.split(/[\\/]/).pop() ?? cur.path
  const fullUrl = api.sampleImageUrl(filename, taskId)

  return (
    <div className="flex flex-col gap-2.5 w-full flex-1">
      {/* 顶部缩略图条 —— 横向滚动，按数组原顺序铺 */}
      <div
        ref={stripRef}
        className="flex gap-1.5 overflow-x-auto pb-1 shrink-0"
        style={{ scrollbarWidth: 'thin' }}
      >
        {list.map((s, i) => {
          const fn = s.path.split(/[\\/]/).pop() ?? s.path
          const thumbUrl = api.sampleImageUrl(fn, taskId, 128)
          const isActive = i === active
          return (
            <button
              key={`${fn}-${i}`}
              onClick={() => setActive(i)}
              className={[
                'shrink-0 rounded-sm overflow-hidden border transition-colors relative',
                isActive ? 'border-accent ring-2 ring-accent-soft' : 'border-subtle hover:border-bold',
                'cursor-pointer p-0 bg-sunken',
              ].join(' ')}
              title={s.step != null ? `step ${s.step}` : fn}
              style={{ width: 64, height: 64 }}
            >
              <img
                src={thumbUrl}
                alt=""
                loading="lazy"
                className="w-full h-full object-cover block"
              />
              {s.step != null && (
                <span className="absolute bottom-0 inset-x-0 bg-black/55 text-white text-[10px] font-mono text-center leading-tight py-0.5">
                  {s.step.toLocaleString()}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* 大图 —— 当前选中
          img 用 absolute inset-0 脱离 flow，避免 sample 图原始分辨率(1024×*)
          顶起父容器 min-content；letterbox 由 object-contain 处理。
          minHeight 220 是底线（letterbox 视觉勉强够），父 row 高度够时由 flex-1 撑满。 */}
      <div
        className="bg-sunken rounded-sm overflow-hidden relative flex-1 min-h-0"
        style={{ minHeight: 220 }}
      >
        <img
          key={fullUrl}
          src={fullUrl}
          alt="sample preview"
          loading="lazy"
          onClick={() => setZoomOpen(true)}
          className="absolute inset-0 w-full h-full object-contain cursor-zoom-in"
        />
        {cur.step != null && (
          <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 border border-subtle rounded-sm px-2.5 py-0.5 text-xs font-mono text-fg-secondary bg-surface/85">
            step <strong className="text-accent">{cur.step.toLocaleString()}</strong>
            <span className="text-fg-tertiary ml-2">{active + 1} / {list.length}</span>
          </div>
        )}
      </div>

      {/* 点击大图放大（参考下载页 ImagePreviewModal）；← / → 在采样序列里前后切 */}
      {zoomOpen && (
        <ImagePreviewModal
          src={fullUrl}
          caption={cur.step != null
            ? `step ${cur.step.toLocaleString()} · ${active + 1} / ${list.length} · ${filename}`
            : `${filename} · ${active + 1} / ${list.length}`}
          hasPrev={active > 0}
          hasNext={active < list.length - 1}
          onClose={() => setZoomOpen(false)}
          onPrev={() => setActive((i) => Math.max(0, i - 1))}
          onNext={() => setActive((i) => Math.min(list.length - 1, i + 1))}
        />
      )}
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function MonitorDashboard({ taskId }: { taskId: number }) {
  const { state, connected } = useMonitorProgress(taskId)
  const [emaAlpha, setEmaAlpha] = useState(0.02)
  // LR / d 默认不做 EMA（数据本身已是 EMA 派生量），slider 拉到 < 1 才平滑
  const [lrAlpha, setLrAlpha] = useState(1)
  const [dAlpha, setDAlpha] = useState(1)

  // Derived stats
  const losses = useMemo(() => state?.losses ?? [], [state?.losses])
  const lrHistory = useMemo(() => state?.lr_history ?? [], [state?.lr_history])
  const optimizerMetricsHistory = useMemo(
    () => state?.optimizer_metrics_history ?? [],
    [state?.optimizer_metrics_history],
  )
  const samples = useMemo(() => state?.samples ?? [], [state?.samples])
  const step = state?.step ?? 0
  const totalSteps = state?.total_steps ?? 0
  const speed = state?.speed ?? 0
  const eta = speed > 0 && totalSteps > step ? fmtSec((totalSteps - step) / speed) : '--'
  const progress = totalSteps > 0 ? Math.min(100, (step / totalSteps) * 100) : 0
  const elapsed = state?.start_time ? fmtSec(Date.now() / 1000 - state.start_time) : '--'

  // Recent loss vs previous (windowed comparison)
  const lossInfo = useMemo(() => {
    if (!losses.length) return null
    const WINDOW = Math.min(50, Math.floor(losses.length / 3)) || losses.length
    const raw = losses.map((l) => l.loss)
    const recent = raw.slice(-WINDOW)
    const prev = raw.length > WINDOW ? raw.slice(-WINDOW * 2, -WINDOW) : null
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length
    if (!prev || prev.length === 0) return { val: recentAvg, delta: null }
    const prevAvg = prev.reduce((a, b) => a + b, 0) / prev.length
    return { val: recentAvg, delta: recentAvg - prevAvg }
  }, [losses])

  // Average loss (raw)
  const avgLoss = useMemo(() => {
    if (!losses.length) return null
    const raw = losses.map((l) => l.loss)
    return raw.reduce((a, b) => a + b, 0) / raw.length
  }, [losses])

  // Current LR
  const lastLr = lrHistory.length ? lrHistory[lrHistory.length - 1].lr : null
  const lastOptimizerMetrics = optimizerMetricsHistory.length
    ? optimizerMetricsHistory[optimizerMetricsHistory.length - 1]
    : null
  const lastD = lastOptimizerMetrics?.d ?? null
  const fmtLr = (v: number | null) => {
    if (v === null) return '--'
    if (v < 0.0001) return v.toExponential(1)
    return v.toFixed(5).replace(/0+$/, '').replace(/\.$/, '')
  }
  const fmtMetric = (v: number | null) => {
    if (v === null) return '--'
    if (Math.abs(v) < 0.0001 || Math.abs(v) >= 10000) return v.toExponential(2)
    return v.toFixed(5).replace(/0+$/, '').replace(/\.$/, '')
  }

  const vram = state?.vram_used_gb
  const vramTotal = state?.vram_total_gb
  const vramTone = vram && vramTotal ? (vram / vramTotal > 0.85 ? 'warn' : 'ok') as 'ok' | 'warn' : undefined

  if (!state && !connected) {
    return (
      <div className="grid place-items-center h-[200px] text-fg-tertiary text-sm">
        等待训练数据…
      </div>
    )
  }

  // 全量 raw series（不再 slice(-60)）— SeriesChart 内部会均匀降采样到 600 渲染
  const lrSeries = lrHistory.map((l) => ({ step: l.step, value: l.lr }))
  const dSeries = optimizerMetricsHistory
    .map((m) => ({ step: m.step, d: m.d }))
    .filter((m): m is { step: number; d: number } => typeof m.d === 'number')
    .map((m) => ({ step: m.step, value: m.d }))

  return (
    <div className="flex flex-col gap-3.5 p-4 h-full overflow-y-auto">
      {/* Connection status + progress */}
      <div className="flex items-center gap-2.5 text-xs text-fg-tertiary font-mono shrink-0">
        <span className={`w-[7px] h-[7px] rounded-full inline-block shrink-0 ${connected ? 'bg-ok animate-pulse' : 'bg-err'}`} />
        {connected ? '实时' : '已断开'}
        {totalSteps > 0 && (
          <>
            <span className="text-dim">·</span>
            <span>{step.toLocaleString()} / {totalSteps.toLocaleString()} steps</span>
            <span className="text-dim">·</span>
            <span>{progress.toFixed(1)}%</span>
            <div className="flex-1 h-1 bg-overlay rounded overflow-hidden">
              <div
                className="h-full bg-accent rounded transition-[width] duration-[1s] ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span>已用 {elapsed}</span>
            {eta !== '--' && (
              <>
                <span className="text-dim">·</span>
                <span>剩余 {eta}</span>
              </>
            )}
          </>
        )}
        <span className="flex-1" />
        <a
          href={`/tools/monitor?task=${taskId}`}
          target="_blank"
          rel="noopener"
          className="text-fg-tertiary no-underline hover:text-fg-primary transition-colors"
        >
          独立监控 ↗
        </a>
      </div>

      {/* 6 stat cards */}
      <div className="grid grid-cols-6 gap-2.5">
        <StatCard label="step" value={step ? step.toLocaleString() : '--'}
          sub={totalSteps ? `of ${totalSteps.toLocaleString()}` : undefined} tone="accent" />
        <StatCard
          label="loss"
          value={lossInfo ? lossInfo.val.toFixed(4) : '--'}
          sub={lossInfo?.delta != null
            ? `recent avg, ${lossInfo.delta > 0 ? '↑' : '↓'}${Math.abs(lossInfo.delta).toFixed(4)}`
            : losses.length > 0 ? 'recent avg' : 'awaiting'}
          tone={lossInfo?.delta != null ? (lossInfo.delta < 0 ? 'ok' : 'warn') : undefined}
        />
        <StatCard label="avg loss" value={avgLoss != null ? avgLoss.toFixed(4) : '--'}
          sub={losses.length ? `${losses.length} pts raw mean` : 'awaiting'} />
        <StatCard label="lr" value={fmtLr(lastLr)}
          sub={lastD != null ? `actual · d ${fmtMetric(lastD)}` : lrHistory.length ? 'learning rate' : undefined} />
        <StatCard
          label={vram ? 'vram' : 'speed'}
          value={vram ? `${vram.toFixed(1)} GB` : speed ? `${speed.toFixed(2)} it/s` : '--'}
          sub={vramTotal ? `of ${vramTotal.toFixed(0)} GB · ${((vram! / vramTotal) * 100).toFixed(0)}%` : undefined}
          tone={vramTone}
        />
        <StatCard label="eta" value={eta} sub={speed ? `${speed.toFixed(2)} it/s` : undefined} />
      </div>

      {/* 左：采样图（竖） / 右：loss → LR
          gridTemplateRows: '1fr' → row 跟随 flex-1 撑满，避免 row 默认 auto 在大屏留空白；
          右卡 minHeight 形成下界，flex-1 在 row 高度 > 3*min+gap 时均分扩展；
          总 min 超视口时由外层 overflow-y-auto 滚 */}
      <div
        className="grid grid-cols-[1fr_1.5fr] gap-3.5 flex-1"
        style={{ gridTemplateRows: '1fr' }}
      >
        {/* 左：采样图 */}
        <div className="card p-0 overflow-hidden flex flex-col min-h-0">
          <div className="px-3.5 py-2.5 border-b border-subtle flex items-center justify-between shrink-0">
            <span className="text-sm font-semibold">采样</span>
            <span className="text-xs text-fg-tertiary font-mono">{samples.length} 张</span>
          </div>
          <div className="flex-1 p-3 flex flex-col min-h-0">
            <SampleViewer samples={samples} taskId={taskId} />
          </div>
        </div>

        {/* 右：loss / lr / d 三卡（d 可选），flex-1 等高平分但夹在 [140, 300] 之间。
            每张卡同结构：header 单行 + 占满 flex-1 的 chart。LR 不再夹带任何 d 信息
            （avoid 之前 d-block 作为 LR 内 shrink-0 死成本顶起 LR card min 的问题）。
            minHeight 140 = 可读下界（再小 chart 不易读，触发外层滚动条而非继续压缩）；
            maxHeight 300 = 防止 4K / 大屏上卡片被拉到失衡的高度（剩余空间留给左列采样图）。 */}
        <div className="flex flex-col gap-3.5 min-h-0">
          <div className="card p-4 flex-1 flex flex-col" style={{ minHeight: 140, maxHeight: 300 }}>
            <div className="flex items-center justify-between mb-2 shrink-0">
              <span className="text-sm font-semibold">loss</span>
              <SmoothControl alpha={emaAlpha} setAlpha={setEmaAlpha} min={0.001} max={0.3} step={0.001} />
            </div>
            <SeriesChart
              data={losses.map((l) => ({ step: l.step, value: l.loss }))}
              rawColor="rgba(74,71,64,0.35)"
              smoothColor="var(--accent)"
              fillColor="var(--accent-soft)"
              emaAlpha={emaAlpha}
              yFormat={(v) => v.toFixed(4)}
              minHeight={60}
            />
          </div>

          <div className="card p-4 flex-1 flex flex-col" style={{ minHeight: 140, maxHeight: 300 }}>
            <div className="flex items-center justify-between mb-2 shrink-0">
              <span className="text-sm font-semibold">learning rate</span>
              <SmoothControl alpha={lrAlpha} setAlpha={setLrAlpha} min={0.005} max={1} step={0.005} />
            </div>
            <SeriesChart
              data={lrSeries}
              rawColor="rgba(224,162,58,0.35)"
              smoothColor="var(--warn)"
              emaAlpha={lrAlpha}
              yFormat={fmtLr}
              minHeight={60}
            />
          </div>

          {dSeries.length >= 2 && (
            <div className="card p-4 flex-1 flex flex-col" style={{ minHeight: 140, maxHeight: 300 }}>
              <div className="flex items-center justify-between mb-2 shrink-0">
                <div className="flex items-baseline gap-2">
                  <span className="text-sm font-semibold">d</span>
                  <span className="text-xs font-mono text-fg-tertiary tabular-nums">
                    {fmtMetric(lastD)}
                  </span>
                </div>
                <SmoothControl alpha={dAlpha} setAlpha={setDAlpha} min={0.005} max={1} step={0.005} />
              </div>
              <SeriesChart
                data={dSeries}
                rawColor="rgba(237,107,58,0.30)"
                smoothColor="var(--accent)"
                emaAlpha={dAlpha}
                yFormat={fmtMetric}
                minHeight={60}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
