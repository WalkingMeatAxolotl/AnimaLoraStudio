/**
 * MonitorDashboard — native React training monitor
 * Replaces the monitor_smooth.html iframe.
 * Data source: GET /api/state?task_id=N  +  SSE monitor_state_updated
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, type MonitorState } from '../api/client'
import { useEventStream } from '../lib/useEventStream'

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

// ── LossChart (pure SVG) ───────────────────────────────────────────────────

function LossChart({ losses, emaAlpha }: {
  losses: Array<{ step: number; loss: number }>
  emaAlpha: number
}) {
  if (!losses.length) return (
    <div className="h-60 grid place-items-center text-fg-tertiary text-sm">
      等待数据…
    </div>
  )

  const pts = downsample(losses, 600)
  const raw = pts.map((p) => p.loss)
  const smooth = calcEMA(raw, emaAlpha)
  const steps = pts.map((p) => p.step)

  const W = 760, H = 220, PX = 36, PY = 14
  const minV = Math.min(...smooth), maxV = Math.max(...smooth)
  const range = maxV - minV || 0.001
  const x = (i: number) => PX + (i / (pts.length - 1)) * (W - PX - 8)
  const y = (v: number) => PY + (1 - (v - minV) / range) * (H - PY - PY)

  const smoothPath = smooth.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  const areaPath = smoothPath + ` L${x(smooth.length - 1).toFixed(1)},${H - PY} L${PX},${H - PY}Z`
  const rawPath = raw.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')

  // y axis labels
  const yTicks = [minV, (minV + maxV) / 2, maxV].map((v) => ({
    v, y: y(v), label: v.toFixed(4),
  }))
  // x axis labels (5 evenly)
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
    const i = Math.round(t * (pts.length - 1))
    return { x: x(i), label: String(steps[i] ?? '') }
  })

  const lastY = y(smooth[smooth.length - 1])

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: 240, display: 'block' }}>
      {/* grid */}
      {[0.25, 0.5, 0.75].map((t) => (
        <line key={t} x1={PX} y1={PY + t * (H - 2 * PY)} x2={W - 8} y2={PY + t * (H - 2 * PY)}
          stroke="var(--border-subtle)" strokeDasharray="3 3" />
      ))}
      {/* area */}
      <path d={areaPath} fill="var(--accent-soft)" opacity="0.5" />
      {/* raw (faint) */}
      <path d={rawPath} stroke="rgba(74,71,64,0.18)" strokeWidth="1" fill="none" />
      {/* smooth */}
      <path d={smoothPath} stroke="var(--accent)" strokeWidth="2" fill="none" strokeLinejoin="round" strokeLinecap="round" />
      {/* last point */}
      <circle cx={x(smooth.length - 1)} cy={lastY} r="4" fill="var(--accent)" stroke="var(--bg-surface)" strokeWidth="2" />
      {/* y axis labels */}
      {yTicks.map(({ v, y: yt, label }) => (
        <text key={v} x={PX - 4} y={yt + 3.5} fontSize="9" fill="var(--fg-tertiary)"
          fontFamily="var(--font-mono)" textAnchor="end">{label}</text>
      ))}
      {/* x axis labels */}
      {xTicks.map(({ x: xt, label }) => (
        <text key={label} x={xt} y={H - 2} fontSize="9" fill="var(--fg-tertiary)"
          fontFamily="var(--font-mono)" textAnchor="middle">{label}</text>
      ))}
    </svg>
  )
}

// ── Sparkline ─────────────────────────────────────────────────────────────

function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return <div className="h-[50px]" />
  const W = 200, H = 50
  const min = Math.min(...values), max = Math.max(...values)
  const range = max - min || 0.001
  const x = (i: number) => (i / (values.length - 1)) * W
  const y = (v: number) => H - ((v - min) / range) * H
  const path = values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full mt-2 block" style={{ height: 50 }}>
      <path d={path} stroke={color} strokeWidth="1.5" fill="none" />
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
    <div className="flex flex-col gap-2.5 w-full">
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

      {/* 大图 —— 当前选中 */}
      <div
        className="bg-sunken rounded-sm overflow-hidden relative flex items-center justify-center flex-1 min-h-0"
        style={{ minHeight: 320 }}
      >
        <img
          key={fullUrl}
          src={fullUrl}
          alt="sample preview"
          loading="lazy"
          className="max-w-full max-h-[480px] object-contain block"
        />
        {cur.step != null && (
          <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 border border-subtle rounded-sm px-2.5 py-0.5 text-xs font-mono text-fg-secondary bg-surface/85">
            step <strong className="text-accent">{cur.step.toLocaleString()}</strong>
            <span className="text-fg-tertiary ml-2">{active + 1} / {list.length}</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function MonitorDashboard({ taskId }: { taskId: number }) {
  const [state, setState] = useState<MonitorState | null>(null)
  const [connected, setConnected] = useState(false)
  const [emaAlpha, setEmaAlpha] = useState(0.02)
  const lastUpdateRef = useRef(0)

  const load = useCallback(async () => {
    try {
      const s = await api.getMonitorState(taskId)
      setState(s)
      setConnected(true)
      lastUpdateRef.current = Date.now()
    } catch {
      if (Date.now() - lastUpdateRef.current > 5000) setConnected(false)
    }
  }, [taskId])

  // mount 时拉一次冷启动；之后 SSE 推送（monitor_state_updated 带全量 state
  // payload）。SSE 重连后 onOpen 也补一次冷启动防漏事件。删了原来的 5s 轮询
  // fallback——SSE 通了就够用，断了 EventSource 自动重连 + onOpen 触发重 fetch。
  useEffect(() => {
    void load()
  }, [load])

  useEventStream(
    (evt) => {
      if (evt.type === 'monitor_state_updated' && String(evt.task_id) === String(taskId)) {
        if (evt.state) {
          setState(evt.state as MonitorState)
          setConnected(true)
          lastUpdateRef.current = Date.now()
        } else {
          // 兜底：极少数情况 SSE 不带 state（旧 backend 行为），退回 fetch
          void load()
        }
      }
    },
    { onOpen: () => void load() },
  )

  // Derived stats
  const losses = useMemo(() => state?.losses ?? [], [state?.losses])
  const lrHistory = useMemo(() => state?.lr_history ?? [], [state?.lr_history])
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
  const fmtLr = (v: number | null) => {
    if (v === null) return '--'
    if (v < 0.0001) return v.toExponential(1)
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

  const lrSparkline = lrHistory.slice(-60).map((l) => l.lr)

  return (
    <div className="flex flex-col gap-3.5 p-4 overflow-y-auto">
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
          sub={lrHistory.length ? 'learning rate' : undefined} />
        <StatCard
          label={vram ? 'vram' : 'speed'}
          value={vram ? `${vram.toFixed(1)} GB` : speed ? `${speed.toFixed(2)} it/s` : '--'}
          sub={vramTotal ? `of ${vramTotal.toFixed(0)} GB · ${((vram! / vramTotal) * 100).toFixed(0)}%` : undefined}
          tone={vramTone}
        />
        <StatCard label="eta" value={eta} sub={speed ? `${speed.toFixed(2)} it/s` : undefined} />
      </div>

      {/* 左：采样图（竖） / 右：loss → LR */}
      <div className="grid grid-cols-[1fr_1.5fr] gap-3.5">
        {/* 左：采样图 — 撑满全高 */}
        <div className="card p-0 overflow-hidden flex flex-col">
          <div className="px-3.5 py-2.5 border-b border-subtle flex items-center justify-between shrink-0">
            <span className="text-sm font-semibold">采样</span>
            <span className="text-xs text-fg-tertiary font-mono">{samples.length} 张</span>
          </div>
          <div className="flex-1 p-3 min-h-0 flex items-center justify-center">
            <SampleViewer samples={samples} taskId={taskId} />
          </div>
        </div>

        {/* 右：loss + lr */}
        <div className="flex flex-col gap-3.5">
          <div className="card p-4">
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-sm font-semibold">loss</span>
              <div className="flex items-center gap-2 text-xs text-fg-tertiary">
                <label className="flex items-center gap-1 cursor-pointer">
                  smooth
                  <input type="range" min="0.001" max="0.3" step="0.001" value={emaAlpha}
                    onChange={(e) => setEmaAlpha(parseFloat(e.target.value))}
                    style={{ width: 60, accentColor: 'var(--accent)' }}
                  />
                  <span className="font-mono w-[3ch]">{emaAlpha.toFixed(2)}</span>
                </label>
              </div>
            </div>
            <LossChart losses={losses} emaAlpha={emaAlpha} />
          </div>

          <div className="card p-4">
            <div className="text-sm font-semibold mb-1.5">learning rate</div>
            <div className="text-2xl font-semibold font-mono tabular-nums text-warn">
              {fmtLr(lastLr)}
            </div>
            <Sparkline values={lrSparkline} color="var(--warn)" />
          </div>
        </div>
      </div>
    </div>
  )
}
