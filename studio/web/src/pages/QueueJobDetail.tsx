/** 0.17 P-G — 数据作业详情页（/queue/jobs/:jid）。
 *
 *  样式与结构对齐 task 详情页（QueueDetail）：同款 header（返回 / #id / 名称 /
 *  状态徽章 / 右侧操作）+ 同款 tab 导航，两个 tab：
 *  - 概览：meta 行卡（对齐 OverviewTab 的 140px 标签列行式卡片）+ 参数行卡
 *    （全字段——用户在原生页面配置的内容一个不藏；param key 有映射用人话标签、
 *    没映射退回原 key）
 *  - 日志：对齐 LogTab（自动滚动开关 + 刷新 + bg-sunken pre），running 时吃
 *    job_log_appended SSE 实时追加
 *
 *  路由与 /queue/:id 无冲突：React Router 静态段（jobs）ranking 高于动态段。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, type Job, type JobStatus } from '../api/client'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'
import {
  JOB_STATUS_TONE, fmtJobDuration, fmtJobTime, fmtParamValue, jobJumpPath,
  paramLabel,
} from './queue/jobUtils'

type Tab = 'overview' | 'log'

export default function QueueJobDetailPage() {
  const { t } = useTranslation()
  const { jid: jidStr } = useParams()
  const jid = Number(jidStr)
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState<Tab>('overview')
  const [projectTitle, setProjectTitle] = useState<string | null>(null)

  const reload = useCallback(async () => {
    try {
      setJob(await api.getJob(jid))
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [jid])
  const reloadRef = useRef(reload); reloadRef.current = reload

  useEffect(() => { void reload() }, [reload])

  useEffect(() => {
    if (!job?.project_id) return
    let cancelled = false
    api.listProjects()
      .then((items) => {
        if (cancelled) return
        setProjectTitle(items.find((p) => p.id === job.project_id)?.title ?? null)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [job?.project_id])

  useEventStream((evt) => {
    if (evt.type === 'job_state_changed' && evt.job_id === jid) {
      void reloadRef.current()
    }
  })

  // running 时「已运行」每 2s 刷新。
  const isRunning = job?.status === 'running'
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!isRunning) return
    const id = window.setInterval(() => setTick((n) => n + 1), 2000)
    return () => window.clearInterval(id)
  }, [isRunning])

  if (!Number.isFinite(jid)) {
    return <p className="text-err">{t('queueDetail.invalidId')}</p>
  }

  const isLive = job?.status === 'running' || job?.status === 'pending'
  const jump = job ? jobJumpPath(job) : null

  const cancel = async () => {
    if (!job) return
    const ok = await confirm(
      t('queue.jobs.cancelConfirm', { id: job.id }),
      { okText: t('queue.jobs.cancelOk') },
    )
    if (!ok) return
    setBusy(true)
    try {
      await api.cancelJob(job.id)
      toast(t('queueDetail.cancelSent'), 'success')
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const STATUS_LABEL: Record<JobStatus, string> = {
    pending: t('status.queued'), running: t('status.running'), done: t('status.done'),
    failed: t('status.failed'), canceled: t('status.canceled'),
  }

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: 'overview', label: t('queueDetail.tabOverview') },
    { key: 'log',      label: t('queueDetail.tabLogs') },
  ]

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      {/* Header — 对齐 QueueDetail */}
      <header className="px-6 py-4 border-b border-subtle flex flex-col gap-2 shrink-0 bg-canvas">
        <div className="flex items-center gap-2.5 flex-wrap">
          <Link to="/queue" className="btn btn-ghost btn-sm no-underline">
            {t('queueDetail.backToQueue')}
          </Link>
          <span className="text-fg-tertiary">/</span>
          <h1 className="m-0 text-xl font-semibold font-mono">#{jid}</h1>
          {job && (
            <>
              <span className="text-fg-secondary text-md">
                {t(`queue.jobs.kind.${job.kind}`)}
              </span>
              <span className={`badge badge-${JOB_STATUS_TONE[job.status]}`}>
                {job.status === 'running' && <span className="dot dot-running" />}
                {STATUS_LABEL[job.status]}
              </span>
            </>
          )}
          <span className="flex-1" />
          {jump && (
            <button
              onClick={() => navigate(jump)}
              className="btn btn-secondary btn-sm"
              data-testid="job-detail-jump"
            >
              {t('queue.jobs.jump')} →
            </button>
          )}
          {isLive && (
            <button
              onClick={() => void cancel()}
              disabled={busy}
              className="btn btn-sm bg-warn-soft border border-warn text-warn"
              data-testid="job-detail-cancel"
            >
              {t('queue.jobs.cancel')}
            </button>
          )}
        </div>
        {error && (
          <div className="px-3 py-2 rounded-md bg-err-soft border border-err text-err text-xs font-mono">
            {error}
          </div>
        )}
      </header>

      {/* Tabs — 对齐 QueueDetail */}
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
        {tab === 'overview' && job && (
          <OverviewTab job={job} projectTitle={projectTitle} statusLabel={STATUS_LABEL} />
        )}
        {tab === 'overview' && !job && (
          <div className="p-6 text-center text-fg-tertiary text-sm">
            {t('common.loading')}
          </div>
        )}
        {tab === 'log' && <LogTab jobId={jid} />}
      </div>
    </div>
  )
}

// ── OverviewTab ─────────────────────────────────────────────────────────────
// 对齐 QueueDetail OverviewTab：140px 标签列的行式卡片。meta 卡 + 参数卡。

function OverviewTab({
  job, projectTitle, statusLabel,
}: {
  job: Job
  projectTitle: string | null
  statusLabel: Record<JobStatus, string>
}) {
  const { t } = useTranslation()

  const items: Array<{ label: string; value: React.ReactNode; mono?: boolean }> = [
    { label: 'ID', value: <code className="font-mono">{job.id}</code> },
    { label: t('queue.typeFilterLabel'), value: t(`queue.jobs.kind.${job.kind}`) },
    { label: t('common.status'), value: (
      <span className={`badge badge-${JOB_STATUS_TONE[job.status]}`}>
        {job.status === 'running' && <span className="dot dot-running" />}
        {statusLabel[job.status]}
      </span>
    ) },
    { label: t('queue.jobs.project'), value: projectTitle ?? `#${job.project_id}` },
    ...(job.version_id
      ? [{ label: t('queue.jobs.version'), value: `v${job.version_id}` }]
      : []),
    { label: t('queueDetail.enqueuedAt'), value: fmtJobTime(job.created_at) },
    { label: t('queueDetail.startedAt'), value: fmtJobTime(job.started_at) },
    { label: t('queueDetail.finishedAt'), value: fmtJobTime(job.finished_at) },
    { label: t('queueDetail.duration'), value: fmtJobDuration(job.started_at, job.finished_at), mono: true },
    { label: 'PID', value: job.pid ?? '—', mono: true },
  ]
  if (job.error_msg) {
    items.push({
      label: t('common.error'),
      value: <code className="font-mono text-xs break-all text-err">{job.error_msg}</code>,
    })
  }

  const params = job.params_decoded && typeof job.params_decoded === 'object'
    ? Object.entries(job.params_decoded)
    : []

  const renderRows = (
    rows: Array<{ label: string; value: React.ReactNode; mono?: boolean }>,
  ) => rows.map((row, i) => (
    <div
      key={row.label}
      className={`grid gap-3 items-center px-[18px] py-2.5 ${i < rows.length - 1 ? 'border-b border-subtle' : 'border-b-0'}`}
      style={{ gridTemplateColumns: '140px 1fr' }}
    >
      <span className="text-sm text-fg-tertiary font-normal">{row.label}</span>
      <span className={`text-sm text-fg-primary ${row.mono ? 'font-mono' : ''}`}>
        {row.value}
      </span>
    </div>
  ))

  return (
    <div className="flex-1 min-h-0 overflow-y-auto p-5 flex flex-col gap-4">
      <div className="card overflow-hidden p-0" style={{ maxWidth: 720 }}>
        {renderRows(items)}
      </div>

      {/* 参数：全字段（用户在原生页面配置的内容），映射人话标签，未映射退回原 key */}
      {params.length > 0 && (
        <div style={{ maxWidth: 720 }} data-testid="job-detail-params">
          <h3 className="m-0 mb-2 text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
            {t('queue.jobs.paramsSection')}
          </h3>
          <div className="card overflow-hidden p-0">
            {renderRows(params.map(([k, v]) => ({
              label: paramLabel(k, t),
              value: fmtParamValue(v, t),
              mono: true,
            })))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── LogTab ──────────────────────────────────────────────────────────────────
// 对齐 QueueDetail LogTab：自动滚动开关 + 刷新 + bg-sunken pre + SSE 增量。

function LogTab({ jobId }: { jobId: number }) {
  const { t } = useTranslation()
  const [content, setContent] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const preRef = useRef<HTMLPreElement>(null)
  const contentRef = useRef('')

  const setBoth = useCallback((s: string) => { contentRef.current = s; setContent(s) }, [])

  const refresh = useCallback(async () => {
    try {
      const log = await api.getJobLog(jobId)
      setBoth(log.content)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [jobId, setBoth])

  useEffect(() => { setBoth(''); void refresh() }, [jobId, refresh, setBoth])

  useEventStream((evt) => {
    if (evt.job_id !== jobId) return
    if (evt.type === 'job_log_appended') {
      const text = typeof evt.text === 'string' ? evt.text : ''
      const prev = contentRef.current
      const sep = prev && !prev.endsWith('\n') ? '\n' : ''
      setBoth(prev + sep + text + '\n')
    } else if (evt.type === 'job_state_changed') {
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
      <pre ref={preRef} className="flex-1 min-h-0 overflow-auto bg-sunken border border-subtle rounded-md p-3.5 text-xs font-mono text-fg-secondary whitespace-pre-wrap break-all m-0" style={{ lineHeight: 1.6 }} data-testid="job-detail-log">
        {content || <span className="text-fg-tertiary">{t('queueDetail.noLogs')}</span>}
      </pre>
    </div>
  )
}
