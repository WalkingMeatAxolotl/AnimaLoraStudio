import { useEffect, useState } from 'react'
import { api, type SystemStats as SystemStatsData } from '../api/client'

const POLL_MS = 2500

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
 *  高度与 topbar 上其他元素 (搜索 icon 32px) 一致。 */
function Pill({ label, value, pct, tooltip }: PillProps) {
  const tone = toneClasses(pct)
  const clamped = Math.min(100, Math.max(0, pct))
  return (
    <div
      className="relative flex items-center gap-1.5 h-8 px-2 rounded-md border border-dim bg-surface overflow-hidden shrink-0"
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

  useEffect(() => {
    let cancelled = false
    let firstFetchDone = false

    const tick = async () => {
      try {
        const s = await api.systemStats()
        if (!cancelled) {
          setStats(s)
          firstFetchDone = true
        }
      } catch {
        // 单次失败保留上次的数据；后端临时挂掉时 topbar 不闪烁
        if (!firstFetchDone && !cancelled) {
          // 首次拉就失败：组件保持不可见，避免空白 pill 占位
        }
      }
    }
    void tick()
    const id = setInterval(tick, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

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
