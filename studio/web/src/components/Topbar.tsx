import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useProjectCtx } from '../context/ProjectContext'
import { api, type Task } from '../api/client'
import { useEventStream, type StudioEvent } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import CommandPalette from './CommandPalette'
import SystemStats from './SystemStats'

const SearchIcon = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" />
  </svg>
)

const QueueIcon = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    <rect x="2" y="3" width="20" height="14" rx="2" />
    <path d="M8 21h8M12 17v4" />
  </svg>
)

// ── 格式化工具 ──────────────────────────────────────────────────────────────

function formatETA(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}分`
  const h = Math.floor(seconds / 3600)
  const m = Math.round((seconds % 3600) / 60)
  return m > 0 ? `${h}时${m}分` : `${h}时`
}

function formatElapsed(from: number): string {
  const s = Math.max(0, (Date.now() / 1000) - from)
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}分`
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return m > 0 ? `${h}时${m}分` : `${h}时`
}

// ── breadcrumb ──────────────────────────────────────────────────────────────

interface Crumb { label: string; mono?: boolean; to?: string }

function useBreadcrumbs(): Crumb[] {
  const { pathname } = useLocation()
  const ctx = useProjectCtx()
  const parts = pathname.split('/').filter(Boolean)

  if (parts.length === 0) return [{ label: '项目', to: '/' }]

  if (parts[0] === 'queue') {
    if (parts.length === 1) return [{ label: '队列', to: '/queue' }]
    return [{ label: '队列', to: '/queue' }, { label: `#${parts[1]}`, mono: true }]
  }

  if (parts[0] === 'tools') {
    const labels: Record<string, string> = { presets: '预设', monitor: '监控', settings: '设置', generate: '测试' }
    return [{ label: labels[parts[1]] ?? parts[1] }]
  }

  if (parts[0] === 'projects') {
    const crumbs: Crumb[] = [{ label: '项目', to: '/' }]

    const projectLabel = ctx?.project?.title ?? (parts[1] ? `#${parts[1]}` : null)
    const projectId = parts[1]
    if (projectLabel) crumbs.push({ label: projectLabel, to: projectId ? `/projects/${projectId}` : undefined })

    const vIdx = parts.indexOf('v')
    if (vIdx !== -1 && parts[vIdx + 1]) {
      const versionLabel = ctx?.activeVersion?.label ?? `v${parts[vIdx + 1]}`
      const vid = parts[vIdx + 1]
      // version 节点没有独立页面，不可跳；step 才有路由。
      crumbs.push({ label: versionLabel, mono: true })
      const stepLabels: Record<string, string> = {
        curate: '筛选', tag: '打标', edit: '标签编辑', reg: '正则集', train: '训练',
      }
      const step = parts[vIdx + 2]
      if (step && stepLabels[step]) {
        crumbs.push({ label: stepLabels[step], to: `/projects/${projectId}/v/${vid}/${step}` })
      }
    } else if (parts[2] === 'download') {
      crumbs.push({ label: '下载', to: `/projects/${projectId}/download` })
    }
    return crumbs
  }

  return [{ label: pathname }]
}

// ── Topbar ──────────────────────────────────────────────────────────────────

export default function Topbar() {
  const crumbs = useBreadcrumbs()
  const navigate = useNavigate()
  const ctx = useProjectCtx()
  const [paletteOpen, setPaletteOpen] = useState(false)
  const searchBtnRef = useRef<HTMLButtonElement>(null)

  // 队列详细状态
  const [runningTask, setRunningTask] = useState<Task | null>(null)
  const [pendingCount, setPendingCount] = useState(0)

  // monitor 状态走 useMonitorProgress hook (PR #37 增量协议)：自动管理
  // /api/state 冷启动 + monitor_progress delta 合并 + 重连补拉，本组件只
  // 关心结果的 step/total_steps/speed/start_time 几个 scalar 字段。
  const { state: monitor } = useMonitorProgress(runningTask?.id ?? null)

  const refreshQueue = useCallback(async () => {
    try {
      const [running, pending] = await Promise.all([
        api.listQueue('running'),
        api.listQueue('pending'),
      ])
      setRunningTask(running.length > 0 ? running[0] : null)
      setPendingCount(pending.length)
    } catch {
      // 忽略
    }
  }, [])

  // 队列任务列表走 SSE task_state_changed；mount + 重连各拉一次冷启动。
  // 之前 3-10s setInterval 在 1 个 running task 下会攒成每秒 7+ 请求的死
  // 循环 bug，根因是 useEffect deps 用了 Task 对象。
  useEffect(() => {
    void refreshQueue()
  }, [refreshQueue])

  useEventStream(
    (evt: StudioEvent) => {
      if (evt.type === 'task_state_changed') {
        void refreshQueue()
      }
    },
    { onOpen: () => void refreshQueue() },
  )

  // ⌘K / Ctrl+K
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setPaletteOpen((p) => !p)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // ── 计算进度 ──
  const progress = (() => {
    if (!monitor || !runningTask) return null

    if (monitor.step != null && monitor.total_steps != null && monitor.total_steps > 0) {
      const pct = Math.round((monitor.step / monitor.total_steps) * 100)
      let eta = ''
      if (monitor.speed && monitor.speed > 0) {
        const remaining = (monitor.total_steps - monitor.step) / monitor.speed
        eta = `~${formatETA(remaining)}`
      }
      return { pct, current: monitor.step, total: monitor.total_steps, eta, unit: 'step' as const }
    }

    if (monitor.epoch != null && monitor.total_epochs != null && monitor.total_epochs > 0) {
      const pct = Math.round((monitor.epoch / monitor.total_epochs) * 100)
      return { pct, current: monitor.epoch, total: monitor.total_epochs, unit: 'epoch' as const }
    }

    if (monitor.start_time) {
      const elapsed = formatElapsed(monitor.start_time)
      return { currentUnit: `运行 ${elapsed}` } as const
    }

    return null
  })()

  // ── 训练胶囊文本 ──
  const projectLabel = (ctx && runningTask?.project_id != null && runningTask.project_id === ctx.project.id)
    ? ctx.project.title
    : null
  const configName = runningTask ? (runningTask.config_name || runningTask.name || '—') : ''
  const taskLabel = projectLabel ? `${projectLabel} / ${configName}` : configName

  const progressSuffix = (() => {
    if (!progress) return runningTask?.started_at ? ` · 运行 ${formatElapsed(runningTask.started_at)}` : ''
    if ('pct' in progress) {
      const p = progress as { current: number; total: number; unit: string; eta?: string }
      const nums = `${p.current.toLocaleString()} / ${p.total.toLocaleString()}`
      return ` · ${p.unit} ${nums}${p.eta ? ` ${p.eta}` : ''}`
    }
    if ('currentUnit' in progress) return ` · ${(progress as { currentUnit: string }).currentUnit}`
    return ''
  })()

  // ── 渲染 ──
  return (
    <>
      <header
        className="flex items-center gap-3 border-b border-subtle bg-canvas shrink-0 px-5"
        style={{ height: 'var(--topbar-h)' }}
      >
        {/* breadcrumb — 非最后一节且有 to 字段时渲染成 Link 可点击。 */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {crumbs.map((b, i) => {
            const isLast = i === crumbs.length - 1
            const cls =
              `text-sm ${b.mono ? 'font-mono' : ''} ` +
              (isLast
                ? 'text-fg-primary font-semibold'
                : 'text-fg-secondary hover:text-fg-primary transition-colors')
            return (
              <span key={i} className="flex items-center gap-2">
                {i > 0 && <span className="text-fg-tertiary select-none">/</span>}
                {!isLast && b.to ? (
                  <Link to={b.to} className={cls}>{b.label}</Link>
                ) : (
                  <span className={cls}>{b.label}</span>
                )}
              </span>
            )
          })}
        </div>

        {/* 实时系统监控 (CPU / RAM / GPU / VRAM)，每 ~2.5s 轮询；无 NVIDIA 时只显示 CPU/RAM。 */}
        <SystemStats />

        {/* 搜索按钮 — 缩成 icon-only，⌘K 仍可弹出完整 palette (含跨图标签搜索)。 */}
        <button
          ref={searchBtnRef}
          onClick={() => setPaletteOpen(true)}
          title="搜索 (⌘K)"
          aria-label="搜索"
          className="flex items-center justify-center text-fg-tertiary bg-surface border border-dim rounded-md cursor-pointer w-8 h-8 hover:border-bold hover:text-fg-secondary transition-colors shrink-0"
        >
          {SearchIcon}
        </button>

        {/* 运行中训练胶囊 */}
        {runningTask && (
          <button
            onClick={() => navigate(`/queue/${runningTask.id}`)}
            className="flex items-center gap-2 px-3 py-[5px] rounded-md border border-warn bg-warn-soft cursor-pointer hover:bg-warn/10 transition-colors shrink-0 max-w-xs"
            title={`任务 #${runningTask.id}`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-warn animate-pulse shrink-0" />
            <span className="text-xs font-mono text-warn overflow-hidden text-ellipsis whitespace-nowrap">
              训练中 · {taskLabel}{progressSuffix}
            </span>
          </button>
        )}

        {/* 排队（无运行中时） */}
        {!runningTask && pendingCount > 0 && (
          <button
            onClick={() => navigate('/queue')}
            className="flex items-center gap-1.5 px-2.5 py-[5px] rounded-md text-xs font-mono text-warn bg-warn-soft border border-warn cursor-pointer hover:bg-warn/10 transition-colors shrink-0"
          >
            {QueueIcon}
            <span>{pendingCount} 排队中</span>
          </button>
        )}
      </header>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        anchorEl={searchBtnRef.current}
      />
    </>
  )
}
