/**
 * MonitorDashboard — native React training monitor
 * Replaces the monitor_smooth.html iframe.
 * Data source: GET /api/state?task_id=N 拉降采样快照 + SSE monitor_progress
 * 走 useMonitorProgress hook 做 delta merge（PR #37 增量协议）。
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api, type EvalJobInfo, type EvalMetricResult, type EvalMetricState, type LoraCkpt, type MonitorState } from '../api/client'
import { evalProgressFromResults } from '../lib/useEvalProgress'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { InfoButton } from './InfoButton'
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

function SeriesChart({ data, rawColor, smoothColor, fillColor, emaAlpha, yFormat, height, minHeight, axes = true, refLine }: {
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
  /** 可选的水平参考线（eval：纯底模 baseline 值）；y 范围会纳入它。 */
  refLine?: number
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
          refLine={refLine}
        />
      ) : null}
    </div>
  )
}

function ChartSvg({ data, W, H, rawColor, smoothColor, fillColor, emaAlpha, yFormat, axes, refLine }: {
  data: Array<{ step: number; value: number }>
  W: number
  H: number
  rawColor: string
  smoothColor: string
  fillColor?: string
  emaAlpha: number
  yFormat: (v: number) => string
  axes: boolean
  refLine?: number
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
  const hasRef = typeof refLine === 'number' && Number.isFinite(refLine)
  // y 范围纳入 baseline 参考线，保证它落在可视区（曲线在 base 线上/下方一目了然）。
  const minV = Math.min(...refVals, ...(hasRef ? [refLine as number] : []))
  const maxV = Math.max(...refVals, ...(hasRef ? [refLine as number] : []))
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
  // 点少时（eval 只有几个 checkpoint）5 个分位会 round 到重复索引（如 3 点 →
  // 0,1,1,2,2），导致标签 "20 20 40 40" 叠在同一 x 上重叠。按索引去重。
  const xTicks = [...new Set(
    [0, 0.25, 0.5, 0.75, 1].map((t) => Math.round(t * Math.max(1, pts.length - 1))),
  )].map((i) => ({ i, x: x(i), label: String(steps[i] ?? '') }))

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
      {/* baseline 参考线（纯底模）+ "base" 标注：曲线在它上方=优于底模，下方=不如底模 */}
      {hasRef && (
        <>
          <line
            x1={PX} y1={y(refLine as number)} x2={W - RX} y2={y(refLine as number)}
            stroke="var(--fg-tertiary)" strokeWidth="1" strokeDasharray="4 3" opacity="0.75"
          />
          <text
            x={W - RX - 1} y={y(refLine as number) - 3} fontSize="10"
            fill="var(--fg-tertiary)" fontFamily="var(--font-mono)" textAnchor="end"
          >base</text>
        </>
      )}
      {axes && (
        <>
          {/* y axis labels —— y offset +4.5 = fontSize/2 + 准基线微调，把字垂直居中到 tick */}
          {yTicks.map(({ v, y: yt, label }) => (
            <text key={v} x={PX - 4} y={yt + 4.5} fontSize="13" fill="var(--fg-tertiary)"
              fontFamily="var(--font-mono)" textAnchor="end">{label}</text>
          ))}
          {/* x axis labels —— 首/末两 tick 在 SVG 边缘上，middle 锚点会让一半字宽溢出被裁，
              改 start/end 锚点把字往内推；中间 tick 维持 middle 居中。 */}
          {xTicks.map(({ i: idx, x: xt, label }, i, arr) => {
            const anchor = i === 0 ? 'start' : i === arr.length - 1 ? 'end' : 'middle'
            return (
              <text key={idx} x={xt} y={H - 3} fontSize="13" fill="var(--fg-tertiary)"
                fontFamily="var(--font-mono)" textAnchor={anchor}>{label}</text>
            )
          })}
        </>
      )}
    </svg>
  )
}

// ── EvalMetricsPanel ──────────────────────────────────────────────────────

const EVAL_METRIC_KEYS = ['clip_t', 'clip_i', 'dino_i', 'ccip_i', 'tag_recall'] as const
type EvalMetricKey = typeof EVAL_METRIC_KEYS[number]
// 核心指标常显；动漫域新指标默认关，只在算过（状态非 not_run）时才显示卡片/列。
const CORE_METRIC_KEYS = new Set<EvalMetricKey>(['clip_t', 'clip_i', 'dino_i'])

const EVAL_LABELS: Record<EvalMetricKey, string> = {
  clip_t: 'CLIP-T',
  clip_i: 'CLIP-I',
  dino_i: 'DINO-I',
  ccip_i: 'CCIP-I',
  tag_recall: 'Tag-Recall',
}

// 每个指标一种线色（并排区分）。深色背景上高对比、可辨。
const EVAL_COLORS: Record<EvalMetricKey, string> = {
  clip_t: '#3fb950',
  clip_i: '#58a6ff',
  dino_i: '#bc8cff',
  ccip_i: '#f778ba',
  tag_recall: '#e3b341',
}

const EVAL_DESCRIPTIONS: Record<EvalMetricKey, string> = {
  clip_t: '生成图和 prompt 文本的 CLIP 相似度，用来看 prompt following；越高越好。',
  clip_i: '生成图和参考图的 CLIP 图像相似度，用来看整体视觉相似度；越高越好。',
  dino_i: '生成图和参考图的 DINO 图像特征相似度，用来看主体或风格特征是否学到；越高越好。',
  ccip_i: '生成图被参考集判为同一动漫角色的比例（CCIP 动漫域角色身份保真）；仅单角色角色 LoRA 有意义；越高越好。',
  tag_recall: '对生成图回标，prompt 里 booru tag 的召回率（动漫原生 prompt following）；仅 booru-tag caption 有意义；越高越好。',
}

function checkpointSortValue(result: EvalMetricResult, index: number): number {
  const value = result.checkpoint?.value
  if (typeof value === 'number') return value
  return result.updated_at ?? result.created_at ?? index
}

function checkpointLabel(result: EvalMetricResult): string {
  return result.checkpoint?.label
    || result.checkpoint?.path?.split(/[\\/]/).pop()
    || result.run_id
}

function metricState(result: EvalMetricResult, key: EvalMetricKey): EvalMetricState | undefined {
  return result.metric_states?.[key]
}

function metricValue(result: EvalMetricResult, key: EvalMetricKey): number | null {
  const stateValue = metricState(result, key)?.value
  if (typeof stateValue === 'number' && Number.isFinite(stateValue)) return stateValue
  const raw = result.metrics?.[key]
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : null
}

function formatEvalValue(value: number | null, state?: EvalMetricState): string {
  if (value != null) return value.toFixed(4)
  const status = state?.status || 'not_run'
  if (status === 'pending') return 'pending'
  if (status === 'running') return 'running'
  if (status === 'failed') return 'failed'
  if (status === 'unavailable') return 'n/a'
  return '--'
}

function stateTone(state?: EvalMetricState, value?: number | null): 'ok' | 'warn' | 'err' | 'muted' {
  if (state?.status === 'failed') return 'err'
  if (state?.status === 'pending' || state?.status === 'running') return 'warn'
  if (value != null || state?.status === 'done') return 'ok'
  return 'muted'
}

function toneClass(tone: 'ok' | 'warn' | 'err' | 'muted'): string {
  if (tone === 'ok') return 'text-ok'
  if (tone === 'warn') return 'text-warn'
  if (tone === 'err') return 'text-err'
  return 'text-fg-tertiary'
}

// 一行 checkpoint 的统一进度文案 + 色调（不管 inline/训练后/手动触发，都从同一份
// run.json/metrics.json 状态推：先出图（sample_run.summary done/total），再算指标。
function evalRowStatus(result: EvalMetricResult): { text: string; tone: 'ok' | 'warn' | 'err' | 'muted' } {
  const s = result.sample_run?.summary
  const total = s?.total ?? 0
  const sampleDone = (s?.done ?? 0) + (s?.failed ?? 0)
  const samplingActive = total > 0 && sampleDone < total &&
    (result.status === 'running' || result.status === 'pending' || (s?.running ?? 0) > 0 || (s?.pending ?? 0) > 0)
  if (samplingActive) return { text: `出图 ${s?.done ?? 0}/${total}`, tone: 'warn' }
  const metricActive = EVAL_METRIC_KEYS.some((k) => {
    const st = metricState(result, k)?.status
    return st === 'pending' || st === 'running'
  })
  if (metricActive) return { text: '算指标…', tone: 'warn' }
  if (result.status === 'failed') return { text: '失败', tone: 'err' }
  if (result.status === 'done') return { text: '完成', tone: 'ok' }
  return { text: result.status, tone: 'muted' }
}

export function EvalMetricsPanel({ state, connected, taskId }: {
  state: MonitorState | null
  connected: boolean
  taskId?: number
}) {
  const pid = state?.project_id
  const vid = state?.version_id
  const [payload, setPayload] = useState<Awaited<ReturnType<typeof api.listEvalMetrics>> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // 训练后/手动评估 job（统一日志区取原始日志用）。
  const [evalJobs, setEvalJobs] = useState<EvalJobInfo[]>([])

  const load = useCallback(async (quiet = false) => {
    if (!pid || !vid) return
    if (!quiet) setLoading(true)
    try {
      const next = await api.listEvalMetrics(pid, vid, taskId)
      setPayload(next)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (!quiet) setLoading(false)
    }
    if (taskId) {
      try {
        const ej = await api.listTaskEvalJobs(pid, vid, taskId)
        setEvalJobs(ej.jobs)
      } catch {
        // 日志关联是辅助信息，拉失败不打扰
      }
    }
  }, [pid, vid, taskId])

  useEffect(() => {
    setPayload(null)
    setError(null)
    if (!pid || !vid) return
    void load()
  }, [pid, vid, load])

  const results = useMemo(() => {
    return [...(payload?.results ?? [])]
      .sort((a, b) => checkpointSortValue(a, 0) - checkpointSortValue(b, 0))
  }, [payload?.results])

  // baseline run（纯底模对照）不作为 checkpoint 展示——只用来给各 checkpoint 算 Δ
  // （后端已挂在 result.delta）。每次「运行评估」会自动清空上一轮，所以这里永远只有
  // 这次 run 的结果，每个 checkpoint 一条，无需跨轮去重。
  const displayResults = useMemo(
    () => results.filter((r) => !r.baseline),
    [results],
  )

  const hasActiveMetric = useMemo(() => {
    return results.some((result) =>
      EVAL_METRIC_KEYS.some((key) => {
        const status = metricState(result, key)?.status
        return status === 'pending' || status === 'running'
      }),
    )
  }, [results])

  // 还有评估 job 在跑（含「出图」阶段——此时 metric 状态还是 not_run，hasActiveMetric
  // 抓不到）。重跑评估在已 done 的 task 上时，靠这个让轮询继续，新 run 进度/日志才刷新。
  const hasActiveJob = useMemo(
    () => evalJobs.some((j) => j.status === 'pending' || j.status === 'running'),
    [evalJobs],
  )

  // 核心指标常显；动漫域新指标默认关，只有算过（状态非 not_run）才显示，避免空卡。
  const displayKeys = useMemo<EvalMetricKey[]>(
    () => EVAL_METRIC_KEYS.filter((k) =>
      CORE_METRIC_KEYS.has(k) ||
      displayResults.some((r) => {
        const s = metricState(r, k)?.status
        return s != null && s !== 'not_run'
      }),
    ),
    [displayResults],
  )

  // 训练结束后评估进度：复用现有 results 聚合「评估中 done/total」，给面板头部用
  const evalAgg = useMemo(() => evalProgressFromResults(results), [results])

  useEffect(() => {
    if (!pid || !vid) return
    if (!connected && !hasActiveMetric && !hasActiveJob) return
    const id = window.setInterval(() => void load(true), 5000)
    return () => window.clearInterval(id)
  }, [connected, hasActiveMetric, hasActiveJob, load, pid, vid])

  // ── 手动评估：选 checkpoint → POST /eval/run（task-scoped） ──────────────
  const [pickerOpen, setPickerOpen] = useState(false)
  const [ckpts, setCkpts] = useState<LoraCkpt[]>([])
  const [ckptsLoading, setCkptsLoading] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [running, setRunning] = useState(false)
  const [runMsg, setRunMsg] = useState<string | null>(null)

  const loadCkpts = useCallback(async () => {
    if (!pid || !vid) return
    setCkptsLoading(true)
    try {
      const items = await api.listVersionLoraCkpts(pid, vid)
      setCkpts(items)
    } catch (err) {
      setRunMsg(err instanceof Error ? err.message : String(err))
    } finally {
      setCkptsLoading(false)
    }
  }, [pid, vid])

  const togglePicker = useCallback(() => {
    setPickerOpen((open) => {
      const next = !open
      if (next && ckpts.length === 0) void loadCkpts()
      return next
    })
  }, [ckpts.length, loadCkpts])

  const toggleCkpt = useCallback((path: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }, [])

  const runEval = useCallback(async () => {
    if (!pid || !vid || !taskId || selected.size === 0) return
    setRunning(true)
    setRunMsg(null)
    try {
      const r = await api.runTaskEval(pid, vid, {
        task_id: taskId,
        checkpoints: [...selected],
      })
      setRunMsg(`已排队 ${r.queued} 个 checkpoint 的评估`)
      setSelected(new Set())
      setPickerOpen(false)
      void load(true)
    } catch (err) {
      setRunMsg(err instanceof Error ? err.message : String(err))
    } finally {
      setRunning(false)
    }
  }, [pid, vid, taskId, selected, load])

  const latestByKey = useMemo(() => {
    const out: Partial<Record<EvalMetricKey, { result: EvalMetricResult; value: number | null; state?: EvalMetricState }>> = {}
    for (const key of EVAL_METRIC_KEYS) {
      for (let i = displayResults.length - 1; i >= 0; i--) {
        const result = displayResults[i]
        const state = metricState(result, key)
        const value = metricValue(result, key)
        if (value != null || state?.status) {
          out[key] = { result, value, state }
          break
        }
      }
    }
    return out
  }, [displayResults])

  const seriesByKey = useMemo(() => {
    const out = Object.fromEntries(
      EVAL_METRIC_KEYS.map((k) => [k, [] as Array<{ x: number; value: number }>]),
    ) as Record<EvalMetricKey, Array<{ x: number; value: number }>>
    displayResults.forEach((result, index) => {
      const x = checkpointSortValue(result, index)
      for (const key of EVAL_METRIC_KEYS) {
        const value = metricValue(result, key)
        if (value != null) out[key].push({ x, value })
      }
    })
    return out
  }, [displayResults])

  // 各指标的纯底模 baseline 值（画成图上的水平参考线）。后端给每条非 baseline 结果
  // 都挂了相同的 baseline_metrics，取任一即可。
  const baselineByKey = useMemo(() => {
    const out: Partial<Record<EvalMetricKey, number>> = {}
    const bm = displayResults.find(
      (r) => r.baseline_metrics && Object.keys(r.baseline_metrics).length,
    )?.baseline_metrics
    if (bm) {
      for (const key of EVAL_METRIC_KEYS) {
        const v = bm[key]
        if (typeof v === 'number' && Number.isFinite(v)) out[key] = v
      }
    }
    return out
  }, [displayResults])

  if (!pid || !vid) {
    return (
      <div className="card px-4 py-3 text-sm text-fg-tertiary">
        当前任务未绑定项目版本，暂不能读取指标。
      </div>
    )
  }

  return (
    <div className="card p-4 flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold">指标</div>
          <div className="text-xs text-fg-tertiary font-mono truncate">
            {state?.project_slug ?? `project ${pid}`} · {state?.version_label ?? `version ${vid}`}
          </div>
        </div>
        <span className="flex-1" />
        {evalAgg.active && (
          <span
            className="badge badge-accent text-xs"
            title="训练结束后正在用验证集对各 checkpoint 算指标（出图 + CLIP / DINO），完成后消失"
          >
            <span className="dot dot-running" />
            评估中 {evalAgg.done}/{evalAgg.total}
          </span>
        )}
        {loading && <span className="text-xs text-fg-tertiary">读取中…</span>}
        {taskId != null && (
          <button
            type="button"
            onClick={togglePicker}
            className={`btn btn-sm ${pickerOpen ? 'btn-primary' : 'btn-secondary'}`}
          >
            运行评估
          </button>
        )}
        <button
          type="button"
          onClick={() => void load()}
          className="btn btn-secondary btn-sm"
        >
          刷新
        </button>
      </div>

      {taskId != null && pickerOpen && (
        <div className="rounded-md border border-subtle bg-overlay px-3 py-2.5 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold">选择 checkpoint 评估</span>
            <span className="text-[11px] text-fg-tertiary">
              选多个可横向对比；样本数 / 模型用 Settings 默认
            </span>
            <span className="flex-1" />
            {ckpts.length > 0 && (
              <button
                type="button"
                className="text-[11px] text-fg-tertiary hover:text-fg underline"
                onClick={() =>
                  setSelected((prev) =>
                    prev.size === ckpts.length ? new Set() : new Set(ckpts.map((c) => c.path)),
                  )
                }
              >
                {selected.size === ckpts.length ? '清空' : '全选'}
              </button>
            )}
            <button
              type="button"
              onClick={() => setPickerOpen(false)}
              className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
              title="关闭"
              aria-label="关闭 checkpoint 选择"
            >
              ×
            </button>
          </div>
          {ckptsLoading ? (
            <div className="text-xs text-fg-tertiary py-1">读取 checkpoint…</div>
          ) : ckpts.length === 0 ? (
            <div className="text-xs text-fg-tertiary py-1">output/ 下没有 LoRA checkpoint。</div>
          ) : (
            // chip 网格（与测试页 LoRA 选择器同款）：auto-fill 等宽列、+/✓ 标记、
            // 选中态 accent-soft 高亮。
            <div
              className="grid gap-1.5 overflow-y-auto"
              style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', maxHeight: 200, padding: 2 }}
            >
              {ckpts.map((c) => {
                const isPicked = selected.has(c.path)
                return (
                  <button
                    key={c.path}
                    type="button"
                    onClick={() => toggleCkpt(c.path)}
                    className="font-mono flex items-center gap-1 min-w-0"
                    style={{
                      fontSize: 11,
                      padding: '4px 8px',
                      borderRadius: 'var(--r-md)',
                      border: isPicked ? '1px solid transparent' : '1px solid var(--border-subtle)',
                      background: isPicked ? 'var(--accent-soft)' : 'var(--bg-sunken)',
                      color: isPicked ? 'var(--accent)' : 'var(--fg-secondary)',
                      cursor: 'pointer',
                    }}
                    title={c.path}
                  >
                    <span className="shrink-0">{isPicked ? '✓' : '+'}</span>
                    <span className="truncate flex-1 text-left">{c.label}</span>
                  </button>
                )
              })}
            </div>
          )}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={running || selected.size === 0}
              onClick={() => void runEval()}
              className="btn btn-primary btn-sm"
            >
              {running ? '排队中…' : `运行评估${selected.size ? ` (${selected.size})` : ''}`}
            </button>
            {runMsg && <span className="text-[11px] text-fg-tertiary">{runMsg}</span>}
          </div>
        </div>
      )}

      {error ? (
        <div className="rounded-md border border-err bg-err-soft px-3 py-2 text-sm text-err">
          评估指标读取失败：{error}
        </div>
      ) : results.length === 0 ? (
        <div className="rounded-md border border-dashed border-subtle px-3 py-3 text-sm text-fg-tertiary">
          暂无 eval 结果。点「运行评估」选 checkpoint 手动评估，或在训练配置开启「训练后指标评估」，训练结束后自动用验证集算 CLIP-T、CLIP-I、DINO-I。
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
            {displayKeys.map((key) => {
              const latest = latestByKey[key]
              const tone = stateTone(latest?.state, latest?.value)
              const series = seriesByKey[key].map((p) => ({ step: p.x, value: p.value }))
              return (
                <div key={key} className="card p-4 flex flex-col min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1 shrink-0">
                    <span className="inline-flex items-center gap-1.5 text-sm font-semibold">
                      {EVAL_LABELS[key]}
                      <InfoButton ariaLabel={`${EVAL_LABELS[key]} 指标说明`}>
                        <p>{EVAL_DESCRIPTIONS[key]}</p>
                      </InfoButton>
                    </span>
                    <span className={`text-xs font-mono ${toneClass(tone)}`}>
                      {latest?.state?.status ?? 'not_run'}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-2 shrink-0 mb-1.5">
                    <span className={`text-2xl font-semibold font-mono tabular-nums ${toneClass(tone)}`}>
                      {formatEvalValue(latest?.value ?? null, latest?.state)}
                    </span>
                    {(() => {
                      const d = latest?.result.delta?.[key]
                      return d != null ? (
                        <span
                          className={`text-xs font-mono tabular-nums shrink-0 ${d >= 0 ? 'text-ok' : 'text-err'}`}
                          title="相对纯底模 baseline 的净增益 Δ"
                        >
                          {d >= 0 ? '+' : ''}{d.toFixed(4)}
                        </span>
                      ) : null
                    })()}
                    <span className="text-[11px] text-fg-tertiary truncate">
                      {latest ? checkpointLabel(latest.result) : '等待指标'}
                    </span>
                  </div>
                  <SeriesChart
                    data={series}
                    rawColor={EVAL_COLORS[key]}
                    smoothColor={EVAL_COLORS[key]}
                    emaAlpha={1}
                    yFormat={(v) => v.toFixed(4)}
                    height={132}
                    refLine={baselineByKey[key]}
                  />
                </div>
              )
            })}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-fg-tertiary">
                <tr className="border-b border-subtle">
                  <th className="text-left font-medium py-1.5 pr-3">checkpoint</th>
                  {displayKeys.map((key) => (
                    <th key={key} className="text-right font-medium py-1.5 px-2">
                      {EVAL_LABELS[key]}
                    </th>
                  ))}
                  <th className="text-right font-medium py-1.5 pl-3">状态</th>
                </tr>
              </thead>
              <tbody>
                {displayResults.slice(-8).reverse().map((result) => {
                  const rowStatus = evalRowStatus(result)
                  return (
                    <tr key={result.run_id} className="border-b border-subtle last:border-0">
                      <td className="py-1.5 pr-3 max-w-[220px] truncate font-mono" title={checkpointLabel(result)}>
                        {checkpointLabel(result)}
                      </td>
                      {displayKeys.map((key) => {
                        const state = metricState(result, key)
                        const value = metricValue(result, key)
                        const tone = stateTone(state, value)
                        const d = result.delta?.[key]
                        return (
                          <td key={key} className={`py-1.5 px-2 text-right font-mono tabular-nums ${toneClass(tone)}`}>
                            {formatEvalValue(value, state)}
                            {d != null && value != null && (
                              <span
                                className={`ml-1 text-[10px] ${d >= 0 ? 'text-ok' : 'text-err'}`}
                                title="相对纯底模 baseline 的 Δ"
                              >
                                {d >= 0 ? '+' : ''}{d.toFixed(4)}
                              </span>
                            )}
                          </td>
                        )
                      })}
                      <td className="py-1.5 pl-3 text-right font-mono">
                        <span className={toneClass(rowStatus.tone)}>{rowStatus.text}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

// ── SampleViewer（单图 + 左右切换） ──────────────────────────────────────

// monitor 给每张图都记了触发那一刻的 global_step，所以 step 始终能显示；epoch 只有
// 按 epoch 采样（文件名 epoch_N.png）的图才有，从文件名解析。两个都返回 → 角标 / 标题
// 「step 一直显示、ep 有就附加」，不用点开看文件名才知道是第几 epoch。
function sampleMarks(s: { path: string; step?: number }): { step: number | null; epoch: number | null } {
  const fn = s.path.split(/[\\/]/).pop() ?? s.path
  const ep = /^epoch_(\d+)/i.exec(fn)
  const st = /^step_(\d+)/i.exec(fn)
  return {
    epoch: ep ? Number(ep[1]) : null,
    step: st ? Number(st[1]) : (s.step != null ? s.step : null),
  }
}

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
  const curM = sampleMarks(cur)
  const markText = [
    curM.epoch != null ? `ep ${curM.epoch.toLocaleString()}` : null,
    curM.step != null ? `step ${curM.step.toLocaleString()}` : null,
  ].filter(Boolean).join(' · ')

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
          const m = sampleMarks(s)
          const thumbTitle = [
            m.epoch != null ? `ep ${m.epoch}` : null,
            m.step != null ? `step ${m.step}` : null,
          ].filter(Boolean).join(' · ') || fn
          // 角标放缩略图下面一行，不压在 64px 小图上（盖住看不清）。
          const thumbCaption = [
            m.epoch != null ? `ep${m.epoch}` : null,
            m.step != null ? `${m.step}` : null,
          ].filter(Boolean).join('·')
          return (
            <button
              key={`${fn}-${i}`}
              onClick={() => setActive(i)}
              className="shrink-0 flex flex-col items-center gap-0.5 p-0 bg-transparent border-none cursor-pointer"
              title={thumbTitle}
            >
              <div
                className={[
                  'rounded-sm overflow-hidden border transition-colors bg-sunken',
                  isActive ? 'border-accent ring-2 ring-accent-soft' : 'border-subtle hover:border-bold',
                ].join(' ')}
                style={{ width: 64, height: 64 }}
              >
                <img
                  src={thumbUrl}
                  alt=""
                  loading="lazy"
                  className="w-full h-full object-cover block"
                />
              </div>
              {thumbCaption && (
                <span className={`text-[10px] font-mono leading-tight text-center ${isActive ? 'text-fg-primary' : 'text-fg-tertiary'}`}>
                  {thumbCaption}
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
        {(curM.epoch != null || curM.step != null) && (
          <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 border border-subtle rounded-sm px-2.5 py-0.5 text-xs font-mono text-fg-secondary bg-surface/85">
            {curM.epoch != null && (
              <>ep <strong className="text-accent">{curM.epoch.toLocaleString()}</strong>{curM.step != null && ' · '}</>
            )}
            {curM.step != null && (
              <>step <strong className="text-accent">{curM.step.toLocaleString()}</strong></>
            )}
            <span className="text-fg-tertiary ml-2">{active + 1} / {list.length}</span>
          </div>
        )}
      </div>

      {/* 点击大图放大（参考下载页 ImagePreviewModal）；← / → 在采样序列里前后切 */}
      {zoomOpen && (
        <ImagePreviewModal
          src={fullUrl}
          caption={[markText, filename].filter(Boolean).join(' · ')}
          index={active}
          total={list.length}
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
