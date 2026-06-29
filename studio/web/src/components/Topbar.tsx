import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useProjectCtx } from '../context/ProjectContext'
import { api, type Task } from '../api/client'
import { useEventStream, type StudioEvent } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { useAnnouncements } from '../lib/Announcements'
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

const BellIcon = (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
  </svg>
)

// ── 格式化工具 ──────────────────────────────────────────────────────────────

function formatETA(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  const h = Math.floor(seconds / 3600)
  const m = Math.round((seconds % 3600) / 60)
  return m > 0 ? `${h}h${m}m` : `${h}h`
}

function formatElapsed(from: number): string {
  const s = Math.max(0, (Date.now() / 1000) - from)
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return m > 0 ? `${h}h${m}m` : `${h}h`
}

// ── breadcrumb ──────────────────────────────────────────────────────────────

interface Crumb { label: string; mono?: boolean; to?: string }

function useBreadcrumbs(): Crumb[] {
  const { t } = useTranslation()
  const { pathname } = useLocation()
  const ctx = useProjectCtx()
  const parts = pathname.split('/').filter(Boolean)

  if (parts.length === 0) return [{ label: t('breadcrumb.projects'), to: '/' }]

  if (parts[0] === 'queue') {
    if (parts.length === 1) return [{ label: t('breadcrumb.queue'), to: '/queue' }]
    return [{ label: t('breadcrumb.queue'), to: '/queue' }, { label: `#${parts[1]}`, mono: true }]
  }

  if (parts[0] === 'tools') {
    const labels: Record<string, string> = {
      presets: t('breadcrumb.presets'),
      monitor: t('breadcrumb.monitor'),
      settings: t('breadcrumb.settings'),
      generate: t('breadcrumb.generate'),
    }
    return [{ label: labels[parts[1]] ?? parts[1] }]
  }

  if (parts[0] === 'projects') {
    const crumbs: Crumb[] = [{ label: t('breadcrumb.projects'), to: '/' }]

    const projectLabel = ctx?.project?.title ?? (parts[1] ? `#${parts[1]}` : null)
    const projectId = parts[1]
    if (projectLabel) crumbs.push({ label: projectLabel, to: projectId ? `/projects/${projectId}` : undefined })

    const vIdx = parts.indexOf('v')
    if (vIdx !== -1 && parts[vIdx + 1]) {
      const versionLabel = ctx?.activeVersion?.label ?? `v${parts[vIdx + 1]}`
      const vid = parts[vIdx + 1]
      crumbs.push({ label: versionLabel, mono: true })
      const stepLabels: Record<string, string> = {
        curate: t('breadcrumb.curate'),
        tag: t('breadcrumb.tag'),
        edit: t('breadcrumb.tagEdit'),
        reg: t('breadcrumb.reg'),
        train: t('breadcrumb.train'),
      }
      const step = parts[vIdx + 2]
      if (step && stepLabels[step]) {
        crumbs.push({ label: stepLabels[step], to: `/projects/${projectId}/v/${vid}/${step}` })
      }
    } else if (parts[2] === 'download') {
      crumbs.push({ label: t('breadcrumb.download'), to: `/projects/${projectId}/download` })
    }
    return crumbs
  }

  return [{ label: pathname }]
}

// ── Topbar ──────────────────────────────────────────────────────────────────

export default function Topbar() {
  const { t } = useTranslation()
  const crumbs = useBreadcrumbs()
  const navigate = useNavigate()
  const ctx = useProjectCtx()
  const { unreadCount, updateInfo, openCenter } = useAnnouncements()
  const [paletteOpen, setPaletteOpen] = useState(false)
  const searchBtnRef = useRef<HTMLButtonElement>(null)

  const [runningTask, setRunningTask] = useState<Task | null>(null)
  const [pendingCount, setPendingCount] = useState(0)

  // 「有可用更新」检查已迁到 lib/Announcements（D8：公告栏铃铛统一承载）。

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

  useEffect(() => {
    void refreshQueue()
  }, [refreshQueue])

  useEventStream(
    (evt: StudioEvent) => {
      if (evt.type === 'task_state_changed') void refreshQueue()
    },
    { onOpen: () => void refreshQueue() },
  )

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
      return { currentUnit: t('topbar.running', { elapsed }) } as const
    }

    return null
  })()

  const projectLabel = (ctx && runningTask?.project_id != null && runningTask.project_id === ctx.project.id)
    ? ctx.project.title
    : null
  const configName = runningTask ? (runningTask.config_name || runningTask.name || '—') : ''
  const taskLabel = projectLabel ? `${projectLabel} / ${configName}` : configName

  const progressSuffix = (() => {
    if (!progress) return runningTask?.started_at ? ` · ${t('topbar.running', { elapsed: formatElapsed(runningTask.started_at) })}` : ''
    if ('pct' in progress) {
      const p = progress as { current: number; total: number; unit: string; eta?: string }
      const nums = `${p.current.toLocaleString()} / ${p.total.toLocaleString()}`
      return ` · ${p.unit} ${nums}${p.eta ? ` ${p.eta}` : ''}`
    }
    if ('currentUnit' in progress) return ` · ${(progress as { currentUnit: string }).currentUnit}`
    return ''
  })()

  return (
    <>
      <header
        className="flex items-center gap-3 border-b border-subtle bg-canvas shrink-0 px-5"
        style={{ height: 'var(--topbar-h)' }}
      >
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

        <SystemStats />

        {/* 版本更新入口已并入右侧公告栏铃铛（D8） */}

        {runningTask && (
          <button
            onClick={() => navigate(`/queue/${runningTask.id}`)}
            className="flex items-center gap-2 px-3 py-[5px] rounded-md border border-warn bg-warn-soft cursor-pointer hover:bg-warn/10 transition-colors shrink-0 max-w-xs"
            title={t('topbar.taskId', { id: runningTask.id })}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-warn animate-pulse shrink-0" />
            <span className="text-xs font-mono text-warn overflow-hidden text-ellipsis whitespace-nowrap">
              {t('topbar.trainingCapsule', { label: taskLabel, suffix: progressSuffix })}
            </span>
          </button>
        )}

        {!runningTask && pendingCount > 0 && (
          <button
            onClick={() => navigate('/queue')}
            className="flex items-center gap-1.5 px-2.5 py-[5px] rounded-md text-xs font-mono text-warn bg-warn-soft border border-warn cursor-pointer hover:bg-warn/10 transition-colors shrink-0"
          >
            {QueueIcon}
            <span>{t('topbar.pendingCount', { n: pendingCount })}</span>
          </button>
        )}

        <button
          onClick={openCenter}
          title={t('announcements.bellTitle')}
          aria-label={t('announcements.bellTitle')}
          className="relative flex items-center justify-center text-fg-tertiary bg-surface border border-dim rounded-md cursor-pointer w-8 h-8 hover:border-bold hover:text-fg-secondary transition-colors shrink-0"
          data-testid="announcement-bell"
        >
          {BellIcon}
          {(unreadCount > 0 || updateInfo?.has_update) && (
            <span
              className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-err border border-canvas"
              data-testid="announcement-bell-dot"
            />
          )}
        </button>

        <button
          ref={searchBtnRef}
          onClick={() => setPaletteOpen(true)}
          title={t('topbar.search')}
          aria-label={t('topbar.searchAriaLabel')}
          className="flex items-center justify-center text-fg-tertiary bg-surface border border-dim rounded-md cursor-pointer w-8 h-8 hover:border-bold hover:text-fg-secondary transition-colors shrink-0"
        >
          {SearchIcon}
        </button>
      </header>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        anchorEl={searchBtnRef.current}
      />
    </>
  )
}
