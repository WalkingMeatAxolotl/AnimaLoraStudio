import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import {
  api,
  type Task,
  type TaskOutputs,
  type TaskStatus,
} from '../api/client'
import { PauseProgressModal } from '../components/PauseProgressModal'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'
import MonitorDashboard from '../components/MonitorDashboard'

type Tab = 'overview' | 'log' | 'monitor' | 'outputs' | 'snapshot'

const STATUS_BADGE: Record<TaskStatus, string> = {
  pending: 'badge badge-neutral',
  running: 'badge badge-accent',
  done: 'badge badge-ok',
  failed: 'badge badge-err',
  canceled: 'badge badge-neutral',
  paused: 'badge badge-warn',
}

const TERMINAL: ReadonlyArray<TaskStatus> = ['done', 'failed', 'canceled']

function fmtTime(ts: number | null | undefined): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false })
}

function fmtDuration(start?: number | null, end?: number | null): string {
  if (!start) return '—'
  const e = end ?? Date.now() / 1000
  const sec = Math.max(0, e - start)
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = Math.floor(sec / 60); const s = Math.floor(sec % 60)
  if (m < 60) return `${m}m ${s}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

// ── StatCard ────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, mono, large, tone }: {
  label: string
  value: string
  sub?: string
  mono?: boolean
  large?: boolean
  tone?: 'accent' | 'ok' | 'warn' | 'err' | 'neutral'
}) {
  const toneClass = tone ? `text-${tone}` : 'text-fg-primary'
  return (
    <div className="flex flex-col gap-1 px-[18px] py-3.5 bg-surface rounded-md border border-subtle">
      <span className="text-xs text-fg-tertiary font-mono tracking-widest uppercase">
        {label}
      </span>
      <span
        className={`${large ? 'text-3xl' : 'text-xl'} font-semibold ${mono ? 'font-mono' : 'font-sans'} tabular-nums ${toneClass}`}
        style={{ letterSpacing: '-0.02em', lineHeight: 1.1 }}
      >
        {value}
      </span>
      {sub && (
        <span className="text-xs text-fg-tertiary font-mono">
          {sub}
        </span>
      )}
    </div>
  )
}

// ── Page ────────────────────────────────────────────────────────────────────
export default function QueueDetailPage() {
  const { id } = useParams<{ id: string }>()
  const taskId = Number(id)
  const { t } = useTranslation()
  const navigate = useNavigate()
  const location = useLocation()
  const { toast } = useToast()

  const [task, setTask] = useState<Task | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState<Tab>(() => {
    if (typeof window === 'undefined') return 'overview'
    const v = window.location.hash.replace(/^#/, '')
    return (['overview', 'log', 'monitor', 'outputs', 'snapshot'] as const).includes(v as Tab) ? (v as Tab) : 'overview'
  })
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [pauseModalOpen, setPauseModalOpen] = useState(false)

  // tab → hash 写回（点 tab 按钮时同步 URL，replaceState 不触发 router 重渲）
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const h = `#${tab}`
      if (window.location.hash !== h) window.history.replaceState(null, '', h)
    }
  }, [tab])

  // hash → tab 同步（用户已在本页 navigate 到同一 task 但换 hash 时切 tab）：
  // 例如 Overview 的「查看输出」点击 navigate(`/queue/${id}#outputs`)。
  // react-router 的 useLocation 会反映 navigate 改的 hash；上面的 replaceState
  // 写回不会更新 router state，所以两条 effect 不会 ping-pong。
  useEffect(() => {
    const v = location.hash.replace(/^#/, '')
    if ((['overview', 'log', 'monitor', 'outputs', 'snapshot'] as const).includes(v as Tab)) {
      setTab((prev) => (prev === v ? prev : (v as Tab)))
    }
  }, [location.hash])

  const reload = useCallback(async () => {
    if (!Number.isFinite(taskId)) return
    try { const t = await api.getTask(taskId); setTask(t); setError(null) }
    catch (e) { setError(String(e)) }
  }, [taskId])

  useEffect(() => { void reload() }, [reload])

  useEventStream((evt) => {
    if (evt.type === 'task_state_changed' && evt.task_id === taskId) void reload()
  })

  useEffect(() => {
    if (task?.status !== 'running') return
    const tick = window.setInterval(() => setTask((t) => (t ? { ...t } : t)), 2000)
    return () => window.clearInterval(tick)
  }, [task?.status])

  if (!Number.isFinite(taskId)) return <p className="text-err">{t('queueDetail.invalidId')}</p>

  const status = task?.status
  const isLive = status === 'running' || status === 'pending'
  const isTerminal = !!status && TERMINAL.includes(status)

  const cancel = async () => {
    if (!task) return
    setBusy(true)
    try { await api.cancelTask(task.id); toast(t('queueDetail.cancelSent'), 'success'); void reload() }
    catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  const retry = async () => {
    if (!task) return
    setBusy(true)
    try { const newTask = await api.retryTask(task.id); toast(t('queueDetail.retryQueued', { id: newTask.id }), 'success'); navigate(`/queue/${newTask.id}`) }
    catch (e) { toast(String(e), 'error'); setBusy(false) }
    finally { setBusy(true) }
  }

  const remove = async () => {
    if (!task) return
    setBusy(true)
    try { await api.deleteTask(task.id); toast(t('queueDetail.deleted'), 'success'); navigate('/queue') }
    catch (e) { toast(String(e), 'error'); setBusy(false); setConfirmDelete(false) }
  }

  // ADR 0006 PR-4: 暂停 / 恢复 / 取消 paused 三连。
  const pauseRunning = async () => {
    if (!task) return
    setPauseModalOpen(true)
    try {
      await api.pauseTask(task.id)
      toast(t('queue.pauseSent'), 'success')
    } catch (e) {
      toast(t('queue.pauseFailed', { reason: String(e) }), 'error')
      setPauseModalOpen(false)
    }
  }

  const resumePaused = async () => {
    if (!task) return
    setBusy(true)
    try {
      await api.resumeTask(task.id)
      toast(t('queue.resumeSent', { id: task.id }), 'success')
      void reload()
    } catch (e) {
      const msg = String(e)
      if (msg.toLowerCase().includes('missing')) toast(t('queue.resumeFailedMissing'), 'error')
      else toast(t('queue.resumeFailed', { reason: msg }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const STATUS_LABEL: Record<TaskStatus, string> = {
    pending: t('status.pending'),
    running: t('status.running'),
    done: t('status.done'),
    failed: t('status.failed'),
    canceled: t('status.canceled'),
    paused: t('status.paused'),
  }

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: 'overview', label: t('queueDetail.tabOverview') },
    { key: 'log',      label: t('queueDetail.tabLogs') },
    { key: 'monitor',  label: t('queueDetail.tabMonitor') },
    { key: 'outputs',  label: t('queueDetail.tabOutputs') },
    { key: 'snapshot', label: t('queueDetail.tabSnapshot') },
  ]

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      {/* Header */}
      <header className="px-6 py-4 border-b border-subtle flex flex-col gap-2 shrink-0 bg-canvas">
        <div className="flex items-center gap-2.5 flex-wrap">
          <Link to="/queue" className="btn btn-ghost btn-sm no-underline"
          >{t('queueDetail.backToQueue')}</Link>
          <span className="text-fg-tertiary">/</span>
          <h1 className="m-0 text-xl font-semibold font-mono">
            #{taskId}
          </h1>
          {task && (
            <>
              <span className="text-fg-secondary text-md">{task.name}</span>
              <code className="text-xs text-fg-tertiary font-mono">{task.config_name}.yaml</code>
            </>
          )}
          {status && (
            <span className={STATUS_BADGE[status]}>
              {status === 'running' && <span className="dot dot-running" />}
              {STATUS_LABEL[status]}
            </span>
          )}
          <span className="flex-1" />
          {isLive && status === 'running' && task?.is_pausable && (
            <button onClick={pauseRunning} disabled={busy || pauseModalOpen} className="btn btn-sm"
              data-testid="detail-pause-btn"
              title={t('queue.pauseHint')}
            >{t('queue.pause')}</button>
          )}
          {isLive && (
            <button onClick={cancel} disabled={busy} className="btn btn-sm bg-warn-soft border border-warn text-warn"
            >{t('queueDetail.cancelTask')}</button>
          )}
          {status === 'paused' && (
            <>
              <button onClick={resumePaused} disabled={busy} className="btn btn-primary btn-sm"
                data-testid="detail-resume-btn"
                title={t('queue.resumeHint')}
              >{t('queue.resume')}</button>
              <button onClick={cancel} disabled={busy}
                className="btn btn-sm bg-err-soft border border-err text-err"
                title={t('queue.cancelPausedHint')}
              >{t('queue.cancelPaused')}</button>
            </>
          )}
          {isTerminal && (
            <>
              <button onClick={retry} disabled={busy} className="btn btn-primary btn-sm">{t('common.retry')}</button>
              <button onClick={() => setConfirmDelete(true)} disabled={busy}
                className="btn btn-sm bg-err-soft border border-err text-err"
              >{t('queueDetail.deleteRecord')}</button>
            </>
          )}
        </div>

        {error && (
          <div className="px-3 py-2 rounded-md bg-err-soft border border-err text-err text-xs font-mono">
            {error}
          </div>
        )}

        {/* Stat cards for running tasks */}
        {task && task.status === 'running' && (
          <div className="grid grid-cols-4 gap-2.5 mt-1">
            <StatCard label={t('queueDetail.duration')} value={fmtDuration(task.started_at, null)} mono large tone="accent" />
            <StatCard label={t('queueDetail.startTime')} value={fmtTime(task.started_at)} mono />
            <StatCard label="Config" value={task.config_name} mono />
            <StatCard label="PID" value={task.pid ? String(task.pid) : '—'} mono />
          </div>
        )}
      </header>

      {/* Tabs */}
      <nav className="flex items-center gap-0 border-b border-subtle shrink-0 px-6">
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`py-2 px-[18px] text-sm border-0 bg-transparent -mb-px cursor-pointer transition-colors ${tab === key ? 'font-semibold text-accent border-b-2 border-accent' : 'font-normal text-fg-tertiary hover:text-fg-primary border-b-2 border-transparent hover:border-default'}`}
          >
            {label}
          </button>
        ))}
      </nav>

      {/* Tab body */}
      <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
        {tab === 'overview' && task && <OverviewTab task={task} />}
        {tab === 'overview' && !task && (
          <div className="p-6 text-center text-fg-tertiary text-sm">
            {t('common.loading')}
          </div>
        )}
        {tab === 'log' && <LogTab taskId={taskId} />}
        {tab === 'monitor' && <MonitorTab taskId={taskId} />}
        {tab === 'outputs' && <OutputsTab taskId={taskId} />}
        {tab === 'snapshot' && <SnapshotConfigTab task={task} />}
      </div>


      {confirmDelete && task && (
        <ConfirmDialog
          title={t('queueDetail.deleteTitle')}
          message={
            <>
              {t('queueDetail.deleteDesc')}{' '}
              <code className="text-fg-primary font-mono">#{task.id} {task.name}</code>{' '}
              <br />
              <span className="text-fg-tertiary text-xs">
                {t('queueDetail.deleteNote')}
              </span>
            </>
          }
          confirmLabel={t('common.delete')}
          danger
          onConfirm={remove}
          onCancel={() => setConfirmDelete(false)}
          busy={busy}
        />
      )}

      {/* ADR §4.3 暂停过程 modal — 跟 Queue.tsx 同组件，UI 锁屏让用户看进度。 */}
      {pauseModalOpen && task && (
        <PauseProgressModal
          taskId={task.id}
          taskName={task.name}
          onClose={() => setPauseModalOpen(false)}
        />
      )}
    </div>
  )
}

// ── OverviewTab ─────────────────────────────────────────────────────────────

function OverviewTab({ task }: { task: Task }) {
  const { t } = useTranslation()
  const statusLabel: Record<string, string> = {
    pending: t('status.pending'), running: t('status.running'), done: t('status.done'),
    failed: t('status.failed'), canceled: t('status.canceled'), paused: t('status.paused'),
  }
  const items: Array<{ label: string; value: React.ReactNode; mono?: boolean }> = [
    { label: 'ID',     value: <code className="font-mono">{task.id}</code> },
    { label: t('common.name'), value: task.name },
    { label: 'Config', value: <code className="font-mono">{task.config_name}.yaml</code> },
    { label: t('common.status'), value: <span className={STATUS_BADGE[task.status]}>{task.status === 'running' && <span className="dot dot-running" />}{statusLabel[task.status]}</span> },
    { label: t('queueDetail.priority'), value: task.priority, mono: true },
    { label: t('queueDetail.enqueuedAt'), value: fmtTime(task.created_at) },
    { label: t('queueDetail.startedAt'), value: fmtTime(task.started_at) },
    { label: t('queueDetail.finishedAt'), value: fmtTime(task.finished_at) },
    { label: t('queueDetail.duration'), value: fmtDuration(task.started_at, task.finished_at), mono: true },
    { label: t('queueDetail.exitCode'),   value: task.exit_code ?? '—', mono: true },
    { label: 'PID',     value: task.pid ?? '—', mono: true },
  ]

  if (task.project_id || task.version_id) {
    items.push({
      label: t('queueDetail.source'),
      value: task.project_id && task.version_id ? (
        <Link to={`/projects/${task.project_id}?version=${task.version_id}`}
          className="text-accent font-mono text-sm"
        >{t('queueDetail.sourceLink', { projectId: task.project_id, versionId: task.version_id })}</Link>
      ) : '—',
    })
  }
  if (task.config_path) {
    items.push({ label: t('queueDetail.configPath'), value: <code className="font-mono text-xs break-all">{task.config_path}</code> })
  }
  if (task.monitor_state_path) {
    items.push({ label: t('queueDetail.monitorFile'), value: <code className="font-mono text-xs break-all">{task.monitor_state_path}</code> })
  }
  if (task.error_msg) {
    items.push({ label: t('common.error'), value: <code className="font-mono text-xs break-all text-err">{task.error_msg}</code> })
  }

  return (
    <div className="flex-1 min-h-0 overflow-y-auto p-5">
      <div className="card overflow-hidden p-0" style={{ maxWidth: 720 }}>
        {items.map((row, i) => (
          <div
            key={row.label}
            className={`grid gap-3 items-center px-[18px] py-2.5 ${i < items.length - 1 ? 'border-b border-subtle' : 'border-b-0'}`}
            style={{ gridTemplateColumns: '140px 1fr' }}
          >
            <span className="text-sm text-fg-tertiary font-normal">
              {row.label}
            </span>
            <span className={`text-sm text-fg-primary ${row.mono ? 'font-mono' : ''}`}>
              {row.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── LogTab ──────────────────────────────────────────────────────────────────

function LogTab({ taskId }: { taskId: number }) {
  const { t } = useTranslation()
  const [content, setContent] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const preRef = useRef<HTMLPreElement>(null)
  const contentRef = useRef('')

  const setBoth = useCallback((s: string) => { contentRef.current = s; setContent(s) }, [])

  const refresh = useCallback(async () => {
    try { const log = await api.getLog(taskId); setBoth(log.content); setError(null) }
    catch (e) { setError(String(e)) }
  }, [taskId, setBoth])

  useEffect(() => { setBoth(''); void refresh() }, [taskId, refresh, setBoth])

  useEventStream((evt) => {
    if (evt.task_id !== taskId) return
    if (evt.type === 'task_log_appended') {
      const text = typeof evt.text === 'string' ? evt.text : ''
      const prev = contentRef.current
      const sep = prev && !prev.endsWith('\n') ? '\n' : ''
      setBoth(prev + sep + text + '\n')
    } else if (evt.type === 'task_state_changed') {
      void refresh()
    }
  })

  useEffect(() => {
    if (autoScroll && preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight
  }, [content, autoScroll])

  return (
    <div className="flex flex-col flex-1 min-h-0 p-4">
      <div className="flex items-center gap-3 text-xs pb-2.5 shrink-0">
        <label className="text-fg-tertiary flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)}
            style={{ width: 14, height: 14, accentColor: 'var(--accent)' }} />
          {t('queueDetail.autoScroll')}
        </label>
        <span className="flex-1" />
        <button onClick={() => void refresh()} className="btn btn-ghost btn-sm">{t('common.refresh')}</button>
      </div>
      {error && (
        <div className="mb-2.5 p-2.5 rounded-md bg-err-soft border border-err text-err text-xs font-mono">{error}</div>
      )}
      <pre ref={preRef} className="flex-1 min-h-0 overflow-auto bg-sunken border border-subtle rounded-md p-3.5 text-xs font-mono text-fg-secondary whitespace-pre-wrap break-all m-0" style={{ lineHeight: 1.6 }}>
        {content || <span className="text-fg-tertiary">{t('queueDetail.noLogs')}</span>}
      </pre>
    </div>
  )
}

// ── MonitorTab ──────────────────────────────────────────────────────────────

function MonitorTab({ taskId }: { taskId: number }) {
  return (
    <div className="flex-1 min-h-0 overflow-hidden">
      <MonitorDashboard taskId={taskId} />
    </div>
  )
}

// ── OutputsTab ──────────────────────────────────────────────────────────────

export function OutputsTab({ taskId }: { taskId: number }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [data, setData] = useState<TaskOutputs | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [zipping, setZipping] = useState(false)
  const [exportingOutputs, setExportingOutputs] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)
  const [selectMode, setSelectMode] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(() => new Set())
  const [downloadDialog, setDownloadDialog] = useState<null | { destination: 'download' | 'data_exports' }>(null)

  useEffect(() => {
    let alive = true
    void api.getTaskOutputs(taskId).then((r) => alive && setData(r)).catch((e) => alive && setError(String(e)))
    return () => { alive = false }
  }, [taskId, refreshKey])

  // 压缩中状态：点 "下载全部 / 下载所选" 时 setZipping(true)，浏览器直链接管下载，
  // 后端打包完 publish task_outputs_zip_ready → SSE 清状态。
  // 60s 兜底防止事件丢失 / 后端失败时按钮卡死。
  useEventStream((evt) => {
    if (evt.task_id !== taskId) return
    if (evt.type === 'task_outputs_zip_ready') {
      setZipping(false)
    } else if (evt.type === 'task_outputs_zip_failed') {
      setZipping(false)
      toast(t('queueDetail.compressionFailed', { error: typeof evt.error === 'string' ? evt.error : '?' }), 'error')
    }
  })

  useEffect(() => {
    if (!zipping) return
    const tid = window.setTimeout(() => {
      setZipping(false)
      toast(t('queueDetail.compressionTimeout'), 'info')
    }, 60_000)
    return () => window.clearTimeout(tid)
  }, [zipping, toast, t])

  // 列排序：默认按 mtime desc（最新的在上，和之前行为一致）。点表头同 key
  // 切方向，换 key 切到该 key 的默认方向（name=asc / size,mtime=desc）。
  type SortKey = 'name' | 'size' | 'mtime'
  const [sortKey, setSortKey] = useState<SortKey>('mtime')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const sortedFiles = useMemo(() => {
    if (!data) return []
    const sign = sortDir === 'asc' ? 1 : -1
    return [...data.files].sort((a, b) => {
      if (sortKey === 'name') {
        // numeric 让 ep_002 排在 ep_010 之前，避免字典序的 ep_10 < ep_2
        return (a.path || a.name).localeCompare(b.path || b.name, undefined, { numeric: true }) * sign
      }
      if (sortKey === 'size') return (a.size - b.size) * sign
      return (a.mtime - b.mtime) * sign
    })
  }, [data, sortKey, sortDir])
  const stateFiles = useMemo(
    () => sortedFiles.filter((f) => f.kind === 'training_state' || f.kind === 'pause_state' || f.kind === 'auto_epoch_state'),
    [sortedFiles]
  )
  const regularFiles = useMemo(
    () => sortedFiles.filter((f) => f.kind !== 'training_state' && f.kind !== 'pause_state' && f.kind !== 'auto_epoch_state'),
    [sortedFiles]
  )

  const onHeaderClick = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'name' ? 'asc' : 'desc')
    }
  }
  const sortArrow = (key: SortKey) =>
    sortKey === key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''

  // 刷新后剔除选中里已不存在的文件路径
  useEffect(() => {
    if (selected.size === 0) return
    const names = new Set(sortedFiles.map((f) => f.path))
    let dropped = false
    const next = new Set<string>()
    for (const n of selected) {
      if (names.has(n)) next.add(n); else dropped = true
    }
    if (dropped) setSelected(next)
  }, [sortedFiles, selected])

  const selectedSize = useMemo(() => {
    let total = 0
    for (const f of sortedFiles) if (selected.has(f.path)) total += f.size
    return total
  }, [sortedFiles, selected])

  const allSelected = sortedFiles.length > 0 && selected.size === sortedFiles.length
  const noneSelected = selected.size === 0
  const partialSelected = !allSelected && !noneSelected

  const toggleSelectAll = () => {
    setSelected(allSelected ? new Set() : new Set(sortedFiles.map((f) => f.path)))
  }
  const toggleOne = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name); else next.add(name)
      return next
    })
  }
  const toggleSelectMode = () => {
    setSelectMode((m) => {
      if (m) setSelected(new Set())  // 退出批量时清空选中
      return !m
    })
  }

  const openFolder = async () => {
    setBusy(true)
    try { const r = await api.openTaskFolder(taskId); toast(t('queueDetail.folderOpened', { path: r.opened }), 'success') }
    catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  const outputSelection = () => {
    const partial = selectMode && selected.size > 0
    if (selectMode && !partial) return null
    return partial ? Array.from(selected) : undefined
  }

  const handleDownloadZip = () => {
    if (zipping) return
    const files = outputSelection()
    if (selectMode && files === null) return
    setZipping(true)
    // 优先用后端给的 archive_basename ({slug}-{label})，老任务没 project/version
    // 时 fallback task_{id}。download 属性是兜底 —— 浏览器优先用响应头
    // Content-Disposition.filename，所以最终下载名以后端为准。
    const baseName = data?.archive_basename ?? `task_${taskId}`
    const zipName = files ? `${baseName}_outputs_selected.zip` : `${baseName}_outputs.zip`
    const a = document.createElement('a')
    a.href = api.taskOutputsZipUrl(taskId, files ?? undefined)
    a.download = zipName
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const handleExportOutputs = async () => {
    if (exportingOutputs) return
    const files = outputSelection()
    if (selectMode && files === null) return
    setExportingOutputs(true)
    try {
      const result = await api.exportTaskOutputs(taskId, files ?? undefined)
      toast(t('queueDetail.exportedToDataExports', { filename: result.filename, path: result.path }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setExportingOutputs(false)
    }
  }

  const copyPath = async () => {
    if (!data?.output_dir) return
    try { await navigator.clipboard.writeText(data.output_dir); toast(t('queueDetail.pathCopied'), 'success') }
    catch { toast(t('queueDetail.copyFailed'), 'error') }
  }

  const renderFileRows = (files: typeof sortedFiles) => files.map((f) => {
    const isSel = selected.has(f.path)
    return (
      <div
        key={f.path}
        onClick={selectMode ? () => toggleOne(f.path) : undefined}
        className={`grid gap-2 px-4 py-2 items-center border-b border-subtle text-xs transition-colors ${selectMode ? `cursor-pointer ${isSel ? 'bg-accent-soft' : 'hover:bg-overlay'}` : 'hover:bg-overlay'}`}
        style={{ gridTemplateColumns: '1fr 100px 160px 80px' }}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <code className="font-mono text-fg-primary overflow-hidden text-ellipsis whitespace-nowrap">{f.path || f.name}</code>
          {f.is_lora && <span className="badge badge-ok">LoRA</span>}
          {f.kind === 'training_state' && <span className="badge badge-warn">State</span>}
          {f.kind === 'pause_state' && <span className="badge badge-warn">Pause</span>}
          {f.kind === 'auto_epoch_state' && <span className="badge badge-warn">Auto</span>}
        </div>
        <span className="text-right font-mono text-fg-tertiary">{fmtBytes(f.size)}</span>
        <span className="text-right font-mono text-fg-tertiary">{fmtTime(f.mtime)}</span>
        <span className="text-right">
          {selectMode ? (
            <input
              type="checkbox"
              checked={isSel}
              onChange={() => toggleOne(f.path)}
              onClick={(e) => e.stopPropagation()}
              style={{ width: 14, height: 14, accentColor: 'var(--accent)', cursor: 'pointer' }}
              aria-label={`${t('common.select')} ${f.path || f.name}`}
            />
          ) : (
            <a href={api.taskOutputDownloadUrl(taskId, f.path)} download={f.name}
              className="text-accent no-underline hover:underline text-xs"
            >{t('queueDetail.downloadFile')}</a>
          )}
        </span>
      </div>
    )
  })

  const renderFileTable = (files: typeof sortedFiles) => (
    <div className="card overflow-hidden p-0">
      <div
        className="grid gap-2 px-4 py-2 text-xs text-fg-tertiary border-b border-subtle font-mono"
        style={{ gridTemplateColumns: '1fr 100px 160px 80px' }}
      >
        <button
          onClick={() => onHeaderClick('name')}
          className="text-left bg-transparent border-0 p-0 text-xs font-mono text-fg-tertiary hover:text-fg-primary cursor-pointer"
        >{t('common.file')}{sortArrow('name')}</button>
        <button
          onClick={() => onHeaderClick('size')}
          className="text-right bg-transparent border-0 p-0 text-xs font-mono text-fg-tertiary hover:text-fg-primary cursor-pointer"
        >{t('common.size')}{sortArrow('size')}</button>
        <button
          onClick={() => onHeaderClick('mtime')}
          className="text-right bg-transparent border-0 p-0 text-xs font-mono text-fg-tertiary hover:text-fg-primary cursor-pointer"
        >{t('queueDetail.modifiedTime')}{sortArrow('mtime')}</button>
        <span className="text-right">
          {selectMode ? (
            <input
              type="checkbox"
              checked={allSelected}
              ref={(el) => { if (el) el.indeterminate = partialSelected }}
              onChange={toggleSelectAll}
              style={{ width: 14, height: 14, accentColor: 'var(--accent)', cursor: 'pointer' }}
              aria-label={t('common.selectAll')}
            />
          ) : null}
        </span>
      </div>
      {renderFileRows(files)}
    </div>
  )

  return (
    <div className="flex flex-col flex-1 min-h-0 p-4 gap-2.5">
      {data?.output_dir ? (
        <div className="flex items-center gap-2 text-xs shrink-0 border-b border-subtle pb-2.5">
          <span className="text-fg-tertiary shrink-0">{t('common.directory')}</span>
          <code className="flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-fg-primary font-mono">{data.output_dir}</code>
          <button onClick={copyPath} className="btn btn-ghost btn-sm">{t('common.copy')}</button>
          {data.supports_open_folder ? (
            <button onClick={openFolder} disabled={busy || !data.exists}
              className="btn btn-secondary btn-sm"
            >{t('common.open')}</button>
          ) : (
            <span className="text-xs text-fg-tertiary shrink-0">{t('common.remote')}</span>
          )}
          <button onClick={() => setRefreshKey((k) => k + 1)} className="btn btn-ghost btn-sm">{t('common.refresh')}</button>
          {data.exists && data.files.length > 0 && (
            <>
              <button
                onClick={toggleSelectMode}
                className={selectMode ? 'btn btn-secondary btn-sm' : 'btn btn-ghost btn-sm'}
              >
                {selectMode ? t('queueDetail.exitBatchMode') : t('queueDetail.batchMode')}
              </button>
              <button
                onClick={() => setDownloadDialog({ destination: 'download' })}
                disabled={zipping || exportingOutputs || (selectMode && noneSelected)}
                className="btn btn-primary btn-sm"
              >
                {zipping
                  ? t('queueDetail.compressing')
                  : exportingOutputs
                    ? t('queueDetail.exportingOutputs')
                    : selectMode
                      ? (noneSelected ? t('queueDetail.downloadSelectedEmpty') : t('queueDetail.downloadSelected', { n: selected.size, size: fmtBytes(selectedSize) }))
                      : t('queueDetail.downloadAll')}
              </button>
            </>
          )}
        </div>
      ) : data && !data.output_dir ? (
        <div className="text-fg-tertiary text-sm shrink-0 py-2">
          {t('queueDetail.noProjectAssoc')}
        </div>
      ) : null}

      <div className="flex-1 min-h-0 overflow-y-auto">
        {error ? (
          <div className="p-2.5 rounded-md bg-err-soft border border-err text-err font-mono text-xs">{error}</div>
        ) : !data ? (
          <div className="text-fg-tertiary text-sm text-center p-5">{t('common.loading')}</div>
        ) : !data.exists ? (
          <div className="text-warn text-sm text-center p-5">{t('queueDetail.dirNotExist')}</div>
        ) : sortedFiles.length === 0 ? (
          <div className="text-fg-tertiary text-sm text-center p-5">{t('queueDetail.dirEmpty')}</div>
        ) : (
          <div className="flex flex-col gap-3">
            {regularFiles.length > 0 && (
              <section className="flex flex-col gap-1.5">
                <div className="px-1 text-xs font-semibold text-fg-secondary">输出文件</div>
                {renderFileTable(regularFiles)}
              </section>
            )}
            {stateFiles.length > 0 && (
              <section className="flex flex-col gap-1.5">
                <div className="px-1 text-xs font-semibold text-warn">训练状态</div>
                {renderFileTable(stateFiles)}
              </section>
            )}
          </div>
        )}
      </div>

      {downloadDialog && (
        <OutputsDownloadDialog
          destination={downloadDialog.destination}
          onDestinationChange={(d) => setDownloadDialog({ destination: d })}
          busy={zipping || exportingOutputs}
          onCancel={() => setDownloadDialog(null)}
          onConfirm={() => {
            const dest = downloadDialog.destination
            setDownloadDialog(null)
            if (dest === 'download') handleDownloadZip()
            else void handleExportOutputs()
          }}
        />
      )}
    </div>
  )
}

// ── OutputsDownloadDialog ───────────────────────────────────────────────────

function OutputsDownloadDialog({
  destination, onDestinationChange, busy, onConfirm, onCancel,
}: {
  destination: 'download' | 'data_exports'
  onDestinationChange: (d: 'download' | 'data_exports') => void
  busy: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onCancel() }}
    >
      <div className="bg-elevated border border-subtle rounded-lg shadow-lg w-full max-w-[420px]">
        <header className="px-[18px] py-3.5 border-b border-subtle">
          <h3 className="m-0 text-md font-semibold text-fg-primary">{t('queueDetail.downloadDialogTitle')}</h3>
        </header>
        <div className="px-[18px] py-3.5 flex flex-col gap-2">
          <div className="text-sm text-fg-secondary">{t('queueDetail.downloadDialogHint')}</div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="outputs-download-destination"
              checked={destination === 'download'}
              onChange={() => onDestinationChange('download')}
              disabled={busy}
            />
            <span className="text-sm text-fg-primary">{t('queueDetail.downloadDestinationLocal')}</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="outputs-download-destination"
              checked={destination === 'data_exports'}
              onChange={() => onDestinationChange('data_exports')}
              disabled={busy}
            />
            <span className="text-sm text-fg-primary">{t('queueDetail.downloadDestinationDataExports')}</span>
          </label>
        </div>
        <footer className="px-[18px] py-3 border-t border-subtle flex items-center gap-2 justify-end">
          <button onClick={onCancel} disabled={busy} className="btn btn-ghost btn-sm">{t('common.cancel')}</button>
          <button onClick={onConfirm} disabled={busy} className="btn btn-primary btn-sm">
            {busy ? '...' : t('common.confirm')}
          </button>
        </footer>
      </div>
    </div>
  )
}

// ── SnapshotConfigTab (ADR-0007 §11.7 / §11.8-D) ──────────────────────────

/** task 启动时冻结的 training config 只读展示 + "套用此配置" 流程。
 *
 *  心智分离（§11.7 设计）：snapshot 是历史，不点 task 跳 version config 编辑页；
 *  按钮 "套用此配置" → confirm → PUT version config → navigate train phase 页。
 *  user 在 train 页可编辑后点 "开始训练" → 创建新 task（同 version 多 task）。
 */
export function SnapshotConfigTab({ task }: { task: Task | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const [data, setData] = useState<{ yaml: string; config: Record<string, unknown> } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [applying, setApplying] = useState(false)
  const [confirmApply, setConfirmApply] = useState(false)

  // 父组件每 2s 浅 clone task 让 elapsed time 走表（QueueDetailPage:133），
  // 用 [task] 作 deps 会让 snapshot config 也跟着 2s 重拉，浏览器闪烁卡顿。
  // snapshot 是 task 启动时冻结的不可变数据 —— 只在 id 变 / pending→running
  // 时拉一次即可（started_at null→number 那一刻 snapshot 才落盘）。
  const taskId = task?.id ?? null
  const startedAt = task?.started_at ?? null
  useEffect(() => {
    if (taskId == null) { setLoading(false); return }
    let cancelled = false
    setLoading(true)
    void api.getTaskSnapshotConfig(taskId)
      .then((r) => { if (!cancelled) { setData(r); setError(null) } })
      .catch((e) => { if (!cancelled) setError(String(e)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [taskId, startedAt])

  const apply = async () => {
    if (!task || !data) return
    const pid = task.project_id, vid = task.version_id
    if (!pid || !vid) {
      toast(t('snapshot.noVersionLink'), 'error')
      return
    }
    setApplying(true)
    try {
      await api.putVersionConfig(pid, vid, data.config as Parameters<typeof api.putVersionConfig>[2])
      setConfirmApply(false)
      navigate(`/projects/${pid}/v/${vid}/train`)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setApplying(false)
    }
  }

  if (loading) return <div className="p-6 text-fg-tertiary text-sm">{t('common.loading')}</div>
  if (error) {
    return (
      <div className="p-6">
        <p className="m-0 text-sm text-err">{error}</p>
        <p className="m-0 text-xs text-fg-tertiary mt-2">{t('snapshot.notFoundHint')}</p>
      </div>
    )
  }
  if (!data) return <div className="p-6 text-fg-tertiary text-sm italic">{t('snapshot.empty')}</div>

  const canApply = !!(task?.project_id && task?.version_id)

  return (
    <div className="p-6 flex flex-col gap-4 flex-1 min-h-0">
      <div className="flex items-start gap-3 shrink-0">
        <div className="flex-1">
          <h3 className="m-0 text-md font-semibold">{t('snapshot.title')}</h3>
          <p className="m-0 mt-1 text-xs text-fg-tertiary">{t('snapshot.subtitle')}</p>
        </div>
        <button
          className="btn btn-primary btn-sm"
          onClick={() => setConfirmApply(true)}
          disabled={!canApply || applying}
          title={canApply ? undefined : t('snapshot.noVersionLink')}
        >
          {t('snapshot.applyBtn')}
        </button>
      </div>
      <pre className="m-0 p-4 rounded-md border border-subtle bg-sunken text-xs font-mono overflow-auto whitespace-pre flex-1 min-h-0">{data.yaml}</pre>

      {confirmApply && (
        <ConfirmDialog
          title={t('snapshot.applyConfirmTitle')}
          message={t('snapshot.applyConfirmDesc')}
          confirmLabel={t('snapshot.applyBtn')}
          onConfirm={apply}
          onCancel={() => { if (!applying) setConfirmApply(false) }}
          busy={applying}
        />
      )}
    </div>
  )
}

// ── ConfirmDialog ───────────────────────────────────────────────────────────

function ConfirmDialog({
  title, message, confirmLabel = '确认', cancelLabel = '取消', danger = false, busy = false,
  onConfirm, onCancel,
}: {
  title: string; message: React.ReactNode; confirmLabel?: string; cancelLabel?: string
  danger?: boolean; busy?: boolean; onConfirm: () => void; onCancel: () => void
}) {
  return (
    <div onClick={onCancel} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60">
      <div onClick={(e) => e.stopPropagation()} className="bg-elevated border border-subtle rounded-lg shadow-lg w-full max-w-[420px]">
        <header className="px-[18px] py-3.5 border-b border-subtle">
          <h3 className="m-0 text-md font-semibold text-fg-primary">{title}</h3>
        </header>
        <div className="px-[18px] py-3.5 text-sm text-fg-secondary">{message}</div>
        <footer className="px-[18px] py-3 border-t border-subtle flex items-center gap-2 justify-end">
          <button onClick={onCancel} disabled={busy} className="btn btn-ghost btn-sm">{cancelLabel}</button>
          <button onClick={onConfirm} disabled={busy}
            className={danger ? 'btn btn-sm bg-err border border-err text-fg-inverse' : 'btn btn-primary btn-sm'}
          >{busy ? '...' : confirmLabel}</button>
        </footer>
      </div>
    </div>
  )
}
