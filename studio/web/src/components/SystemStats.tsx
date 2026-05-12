import { useEffect, useState } from 'react'
import { api, type SystemStats as SystemStatsData } from '../api/client'
import { useEventStream } from '../lib/useEventStream'

function toneClasses(pct: number): { text: string; bg: string } {
  if (pct >= 90) return { text: 'text-err', bg: 'bg-err-soft' }
  if (pct >= 70) return { text: 'text-warn', bg: 'bg-warn-soft' }
  return { text: 'text-fg-primary', bg: 'bg-accent-soft' }
}

function fmtGb(used: number, total: number): string {
  return `${used.toFixed(1)}/${Math.round(total)}G`
}

interface PillProps {
  label: string
  value: string
  pct: number
  tooltip: string
}

/** 进度条胶囊 — 整个 pill 背景按占用百分比填色 (>=70% warn, >=90% err)，
 *  高度与 topbar 上其他元素 (搜索 icon 32px) 一致。
 *
 *  `min-w-[96px]` + `justify-between` 让 4 个 pill 视觉等宽：CPU/GPU 只占 3-4
 *  字符（"CPU 13%"），MEM/VRAM 占 11 字符（"MEM 35.6/63G"），auto-width 下
 *  宽度差近 1 倍。固定下界 96px (够 "VRAM 80.0/128G" 之类最长情况)，label 左
 *  value 右两端对齐，bg 填充自然居于中间。 */
function Pill({ label, value, pct, tooltip }: PillProps) {
  const tone = toneClasses(pct)
  const clamped = Math.min(100, Math.max(0, pct))
  return (
    <div
      className="relative flex items-center justify-between gap-1.5 h-8 min-w-[96px] px-2 rounded-md border border-dim bg-surface overflow-hidden shrink-0"
      title={tooltip}
    >
      <div
        aria-hidden
        className={`absolute inset-y-0 left-0 ${tone.bg} transition-[width] duration-500 ease-out`}
        style={{ width: `${clamped}%` }}
      />
      <span className="relative z-10 text-2xs uppercase tracking-wider text-fg-tertiary">{label}</span>
      <span className={`relative z-10 font-mono text-xs tabular-nums ${tone.text}`}>{value}</span>
    </div>
  )
}

export default function SystemStats() {
  const [stats, setStats] = useState<SystemStatsData | null>(null)

  // mount 时拉一次冷启动 (避免空白等 2.5s 首个 SSE 事件)，之后纯靠后端
  // sampler 通过 SSE 推送。SSE 重连时 onOpen 也补一次冷启动，防漏。
  useEffect(() => {
    let cancelled = false
    api.systemStats().then((s) => {
      if (!cancelled) setStats(s)
    }).catch(() => {/* 首次失败：等 SSE 第一帧就行 */})
    return () => { cancelled = true }
  }, [])

  useEventStream(
    (evt) => {
      if (evt.type !== 'system_stats_updated') return
      const payload = evt.payload as SystemStatsData | undefined
      if (payload) setStats(payload)
    },
    {
      onOpen: () => {
        // SSE 重连：补一次冷启动；服务端 sampler 仍在跑，下次 tick 会自然推
        // 上来，但这一次显式 GET 让 UI 立刻刷新
        api.systemStats().then((s) => setStats(s)).catch(() => {})
      },
    },
  )

  if (!stats) return null

  const gpu0 = stats.gpu && stats.gpu.length > 0 ? stats.gpu[0] : null
  const ramPct = stats.ram_total_gb > 0 ? (stats.ram_used_gb / stats.ram_total_gb) * 100 : 0
  const vramPct = gpu0 && gpu0.vram_total_gb > 0 ? (gpu0.vram_used_gb / gpu0.vram_total_gb) * 100 : 0

  const gpuExtra = stats.gpu && stats.gpu.length > 1
    ? ` (+${stats.gpu.length - 1} more)`
    : ''
  const gpuTempText = gpu0?.temp_c != null ? ` · ${gpu0.temp_c}°C` : ''
  const gpuLabel = gpu0 ? `${gpu0.name}${gpuTempText}${gpuExtra}` : ''

  return (
    <div className="hidden md:flex items-center gap-2 shrink-0">
      <Pill
        label="CPU"
        value={`${stats.cpu_pct.toFixed(0)}%`}
        pct={stats.cpu_pct}
        tooltip={`CPU 占用 ${stats.cpu_pct.toFixed(1)}%`}
      />
      <Pill
        label="MEM"
        value={fmtGb(stats.ram_used_gb, stats.ram_total_gb)}
        pct={ramPct}
        tooltip={`内存 ${stats.ram_used_gb.toFixed(1)} / ${stats.ram_total_gb.toFixed(1)} GB (${ramPct.toFixed(0)}%)`}
      />
      {gpu0 && (
        <>
          <Pill
            label="GPU"
            value={`${gpu0.util_pct}%`}
            pct={gpu0.util_pct}
            tooltip={`GPU 利用率 · ${gpuLabel}`}
          />
          <Pill
            label="VRAM"
            value={fmtGb(gpu0.vram_used_gb, gpu0.vram_total_gb)}
            pct={vramPct}
            tooltip={`显存 ${gpu0.vram_used_gb.toFixed(1)} / ${gpu0.vram_total_gb.toFixed(1)} GB (${vramPct.toFixed(0)}%) · ${gpuLabel}`}
          />
        </>
      )}
    </div>
  )
}
