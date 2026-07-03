/** 0.17 P-G — 数据作业详情页（/queue/jobs/:jid）。
 *
 *  对齐 GPU 任务的轻量 detail 形态（QueueDetail 的 overview+log 两块）：
 *  header（#id + kind + 状态徽章 + 取消/跳转）+ 概览（meta 两列 grid + 参数
 *  全字段，param key 有映射用人话标签、没映射退回原 key）+ 日志（tail 回放，
 *  running 时吃 job_log_appended SSE 实时追加）。
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

export default function QueueJobDetailPage() {
  const { t } = useTranslation()
  const { jid: jidStr } = useParams()
  const jid = Number(jidStr)
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [logText, setLogText] = useState('')
  const [busy, setBusy] = useState(false)
  const [projectTitle, setProjectTitle] = useState<string | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

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
    let cancelled = false
    api.getJobLog(jid, 2000)
      .then((r) => { if (!cancelled) setLogText(r.content) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [jid])

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
    } else if (
      evt.type === 'job_log_appended' &&
      evt.job_id === jid &&
      typeof evt.text === 'string'
    ) {
      const line = evt.text
      setLogText((cur) => (cur ? `${cur}\n${line}` : line))
    }
  })

  // 日志追加时贴底（简单跟随，不做智能锁定）。
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [logText])

  // running 行「已运行」每 2s 刷新。
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

  const params = job?.params_decoded && typeof job.params_decoded === 'object'
    ? Object.entries(job.params_decoded)
    : []

  const metaItems: Array<{ label: string; value: React.ReactNode; mono?: boolean }> = job ? [
    { label: 'ID', value: <code className="font-mono">{job.id}</code> },
    { label: t('common.status'), value: (
      <span className={`badge badge-${JOB_STATUS_TONE[job.status]}`}>
        {job.status === 'running' && <span className="dot dot-running" />}
        {STATUS_LABEL[job.status]}
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
  ] : []

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      {/* Header — 对齐 QueueDetail */}
      <header className="px-6 py-4 border-b border-subtle flex items-center gap-2.5 flex-wrap shrink-0 bg-canvas">
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
        {jump && (
          <button
            onClick={() => navigate(jump)}
            className="btn btn-secondary btn-sm"
            data-testid="job-detail-jump"
          >
            {t('queue.jobs.jump')} →
          </button>
        )}
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto p-6 flex flex-col gap-4">
        {error && (
          <div className="px-3.5 py-2.5 rounded-md bg-err-soft border border-err text-err text-xs font-mono">
            {error}
          </div>
        )}

        {job && (
          <>
            {/* 概览：meta 两列 grid（对齐 QueueDetail OverviewTab） */}
            <section className="card border border-subtle bg-surface p-5">
              <h3 className="m-0 mb-3 text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                {t('queueDetail.tabOverview')}
              </h3>
              <div
                className="grid gap-x-8 gap-y-2 text-sm"
                style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}
              >
                {metaItems.map(({ label, value, mono }) => (
                  <div key={label} className="flex items-baseline gap-3 min-w-0">
                    <span className="text-fg-tertiary text-xs w-[88px] shrink-0">{label}</span>
                    <span className={`text-fg-primary min-w-0 overflow-hidden text-ellipsis ${mono ? 'font-mono' : ''}`}>
                      {value}
                    </span>
                  </div>
                ))}
              </div>
              {job.error_msg && (
                <div className="mt-3 px-3 py-2 rounded-md bg-err-soft border border-err text-err text-xs font-mono">
                  {job.error_msg}
                </div>
              )}
            </section>

            {/* 参数：全字段（用户在原生页面配置的内容），映射人话标签，未映射退回原 key */}
            {params.length > 0 && (
              <section className="card border border-subtle bg-surface p-5" data-testid="job-detail-params">
                <h3 className="m-0 mb-3 text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                  {t('queue.jobs.paramsSection')}
                </h3>
                <div
                  className="grid gap-x-8 gap-y-2 text-sm"
                  style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}
                >
                  {params.map(([k, v]) => (
                    <div key={k} className="flex items-baseline gap-3 min-w-0">
                      <span className="text-fg-tertiary text-xs w-[88px] shrink-0" title={k}>
                        {paramLabel(k, t)}
                      </span>
                      <span className="text-fg-primary font-mono text-xs min-w-0 overflow-hidden text-ellipsis whitespace-nowrap" title={fmtParamValue(v, t)}>
                        {fmtParamValue(v, t)}
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* 日志：tail 回放 + running 实时追加 */}
            <section className="card border border-subtle bg-surface p-5 flex flex-col min-h-0">
              <h3 className="m-0 mb-3 text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                {t('queueDetail.tabLogs')}
              </h3>
              <pre
                ref={logRef}
                className="m-0 p-3 rounded-md bg-canvas border border-subtle text-xs font-mono max-h-[420px] overflow-auto whitespace-pre-wrap"
                data-testid="job-detail-log"
              >
                {logText || t('queueDetail.noLogs')}
              </pre>
            </section>
          </>
        )}
      </div>
    </div>
  )
}
