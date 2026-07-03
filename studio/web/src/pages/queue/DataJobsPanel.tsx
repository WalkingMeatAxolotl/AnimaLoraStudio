/** 0.17 P-G — 数据作业只读区（队列页「数据作业」tab）。
 *
 *  project_jobs（download/preprocess/tag/reg_build/eval_*）的统一只读呈现：
 *  进行中 + 历史（后端分页）+ kind 过滤。行点击行内展开 summary（params/错误）
 *  + 日志 tail（running 时吃 job_log_appended SSE 实时追加）。行操作按既有范式
 *  icon 化：取消（pending/running，confirm）+ 跳转原生步骤页。
 *
 *  只读 = 不参与 reorder / pause；真正合并进 tasks 留 0.18（设计 D4）。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  api, type Job, type JobKind, type JobStatus, type JobsHistoryPage,
} from '../../api/client'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'

const KINDS: JobKind[] = [
  'download', 'preprocess', 'tag', 'reg_build',
  'eval_samples', 'eval_clip', 'eval_dino', 'eval_tag', 'eval_ccip',
]

const STATUS_TONE: Record<JobStatus, string> = {
  pending: 'neutral', running: 'accent', done: 'ok', failed: 'err', canceled: 'neutral',
}

const HISTORY_PAGE_SIZE = 20

function fmtAgo(ts: number): string {
  const sec = Math.max(0, Date.now() / 1000 - ts)
  if (sec < 60) return '刚刚'
  if (sec < 3600) return `${Math.floor(sec / 60)}m 前`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h 前`
  return `${Math.floor(sec / 86400)}d 前`
}

function fmtDuration(start: number | null, end: number | null): string {
  if (!start) return '—'
  const e = end ?? Date.now() / 1000
  const sec = Math.max(0, e - start)
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = Math.floor(sec / 60); const s = Math.floor(sec % 60)
  if (m < 60) return `${m}m ${s}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

/** kind → 原生步骤页深链（download 是 project 级，其余 version 级；eval_* 落训练页）。 */
function jumpPath(job: Job): string | null {
  const pid = job.project_id
  const vid = job.version_id
  if (!pid) return null
  if (job.kind === 'download') return `/projects/${pid}/download`
  if (!vid) return null
  switch (job.kind) {
    case 'preprocess': return `/projects/${pid}/v/${vid}/preprocess`
    case 'tag': return `/projects/${pid}/v/${vid}/tag`
    case 'reg_build': return `/projects/${pid}/v/${vid}/reg`
    default: return `/projects/${pid}/v/${vid}/train`
  }
}

/** params_decoded 压平成一行行 key: value（嵌套值 JSON 化截断），summary 展示用。 */
function paramLines(job: Job): Array<[string, string]> {
  const p = job.params_decoded
  if (!p || typeof p !== 'object') return []
  return Object.entries(p).map(([k, v]) => {
    const s = typeof v === 'string' ? v : JSON.stringify(v)
    return [k, s.length > 120 ? `${s.slice(0, 120)}…` : s] as [string, string]
  })
}

export default function DataJobsPanel() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [live, setLive] = useState<Job[]>([])
  const [history, setHistory] = useState<JobsHistoryPage>({
    items: [], total: 0, page: 1, page_size: HISTORY_PAGE_SIZE,
  })
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [kindFilter, setKindFilter] = useState<JobKind | null>(null)
  const [historyPage, setHistoryPage] = useState(1)
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [logText, setLogText] = useState('')
  const [projectTitles, setProjectTitles] = useState<Record<number, string>>({})
  const reloadTimer = useRef<number | null>(null)
  const expandedRef = useRef<number | null>(null)
  expandedRef.current = expandedId

  const reload = useCallback(async () => {
    try {
      const [l, h] = await Promise.all([
        api.listJobsLive(kindFilter ?? undefined),
        api.listJobsHistory({
          page: historyPage, pageSize: HISTORY_PAGE_SIZE, kind: kindFilter ?? undefined,
        }),
      ])
      setLive(l); setHistory(h); setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoaded(true)
    }
  }, [kindFilter, historyPage])
  const reloadRef = useRef(reload); reloadRef.current = reload

  useEffect(() => { void reload() }, [reload])

  // 项目名映射（行上显示项目标题；拿不到就回落 #pid）。
  useEffect(() => {
    api.listProjects()
      .then((items) => setProjectTitles(
        Object.fromEntries(items.map((p) => [p.id, p.title])),
      ))
      .catch(() => {})
  }, [])

  // 展开行 → 拉日志 tail；running 行的增量走下面的 SSE 追加。
  useEffect(() => {
    if (expandedId == null) return
    let cancelled = false
    setLogText('')
    api.getJobLog(expandedId, 300)
      .then((r) => { if (!cancelled) setLogText(r.content) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [expandedId])

  useEventStream((evt) => {
    if (evt.type === 'job_state_changed') {
      if (reloadTimer.current) return
      reloadTimer.current = window.setTimeout(() => {
        reloadTimer.current = null
        void reloadRef.current()
      }, 100)
    } else if (
      evt.type === 'job_log_appended' &&
      typeof evt.text === 'string' &&
      evt.job_id === expandedRef.current
    ) {
      const line = evt.text
      setLogText((cur) => (cur ? `${cur}\n${line}` : line))
    }
  })

  const cancelJob = async (job: Job) => {
    const ok = await confirm(
      t('queue.jobs.cancelConfirm', { id: job.id }),
      { okText: t('queue.jobs.cancelOk') },
    )
    if (!ok) return
    try {
      await api.cancelJob(job.id)
      toast(t('queueDetail.cancelSent'), 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const totalPages = Math.max(1, Math.ceil(history.total / HISTORY_PAGE_SIZE))
  const isEmpty = loaded && live.length === 0 && history.total === 0 && !kindFilter

  const KIND_LABEL = useMemo(() => Object.fromEntries(
    KINDS.map((k) => [k, t(`queue.jobs.kind.${k}`)]),
  ) as Record<JobKind, string>, [t])

  const renderRow = (job: Job) => {
    const isLive = job.status === 'running' || job.status === 'pending'
    const jump = jumpPath(job)
    const expanded = expandedId === job.id
    const projectLabel = projectTitles[job.project_id] ?? `#${job.project_id}`
    const STATUS_LABEL: Record<JobStatus, string> = {
      pending: t('status.queued'), running: t('status.running'), done: t('status.done'),
      failed: t('status.failed'), canceled: t('status.canceled'),
    }
    return (
      <div
        key={job.id}
        className={`card overflow-hidden p-0 ${job.status === 'running' ? 'border border-accent bg-accent-soft' : 'border border-subtle bg-surface'}`}
      >
        <button
          onClick={() => setExpandedId(expanded ? null : job.id)}
          title={t('queue.jobs.expandTooltip')}
          className="block w-full text-left p-0 cursor-pointer card-hover"
          data-testid={`job-row-${job.id}`}
        >
          <div
            className="px-[22px] py-4 grid gap-3 items-center"
            style={{ gridTemplateColumns: '48px minmax(0,1fr) 96px 150px 120px' }}
          >
            <span className={`font-mono text-sm ${job.status === 'running' ? 'text-accent font-semibold' : 'text-fg-tertiary'}`}>
              #{job.id}
            </span>
            <div style={{ minWidth: 0 }}>
              <div className="font-semibold text-fg-primary text-sm overflow-hidden text-ellipsis whitespace-nowrap">
                {KIND_LABEL[job.kind] ?? job.kind}
              </div>
              <div className="text-xs text-fg-tertiary mt-0.5 overflow-hidden text-ellipsis whitespace-nowrap">
                {projectLabel}{job.version_id ? ` · v${job.version_id}` : ''}
              </div>
            </div>
            <span className={`badge badge-${STATUS_TONE[job.status]} text-xs text-center`}>
              {job.status === 'running' && <span className="dot dot-running" />}
              {STATUS_LABEL[job.status]}
            </span>
            <span className="font-mono text-xs text-fg-tertiary text-right">
              {job.status === 'running' ? (
                fmtDuration(job.started_at, null)
              ) : job.finished_at ? (
                <>
                  {fmtDuration(job.started_at, job.finished_at)}
                  <br />
                  {fmtAgo(job.finished_at)}
                </>
              ) : '—'}
            </span>
            {/* action 列：取消（live）+ 跳转，icon 化 hover 显文字（既有范式） */}
            <div className="flex items-center justify-end gap-1.5">
              {isLive && (
                <button
                  onClick={(e) => { e.stopPropagation(); void cancelJob(job) }}
                  className="btn btn-ghost btn-sm px-2 text-err"
                  title={t('queue.jobs.cancel')}
                  aria-label={t('queue.jobs.cancel')}
                  data-testid={`job-cancel-btn-${job.id}`}
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              )}
              {jump && (
                <button
                  onClick={(e) => { e.stopPropagation(); navigate(jump) }}
                  className="btn btn-ghost btn-sm px-2"
                  title={t('queue.jobs.jump')}
                  aria-label={t('queue.jobs.jump')}
                  data-testid={`job-jump-btn-${job.id}`}
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M7 17L17 7M7 7h10v10" />
                  </svg>
                </button>
              )}
            </div>
          </div>
        </button>

        {/* 行内展开：params summary + 错误 + 日志 tail */}
        {expanded && (
          <div
            className="px-[22px] pb-4 flex flex-col gap-2 border-t border-subtle pt-3"
            data-testid={`job-expand-${job.id}`}
          >
            {paramLines(job).length > 0 && (
              <div className="text-xs text-fg-secondary font-mono flex flex-col gap-0.5">
                {paramLines(job).map(([k, v]) => (
                  <div key={k} className="overflow-hidden text-ellipsis whitespace-nowrap">
                    <span className="text-fg-tertiary">{k}: </span>{v}
                  </div>
                ))}
              </div>
            )}
            {job.error_msg && (
              <div className="text-xs text-err font-mono">{job.error_msg}</div>
            )}
            <pre className="m-0 p-2.5 rounded-md bg-canvas border border-subtle text-xs font-mono max-h-64 overflow-auto whitespace-pre-wrap">
              {logText || t('queueDetail.noLogs')}
            </pre>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4" data-testid="data-jobs-panel">
      {/* kind 过滤（只读区自己的轻过滤行） */}
      <div className="flex items-center gap-2">
        <select
          className="input"
          style={{ width: 180 }}
          value={kindFilter ?? 'all'}
          onChange={(e) => {
            const v = e.target.value
            setKindFilter(v === 'all' ? null : (v as JobKind))
            setHistoryPage(1)
          }}
          aria-label={t('queue.typeFilterLabel')}
          data-testid="jobs-kind-filter"
        >
          <option value="all">{t('queue.filterAll')}</option>
          {KINDS.map((k) => (
            <option key={k} value={k}>{t(`queue.jobs.kind.${k}`)}</option>
          ))}
        </select>
        <span className="flex-1" />
        <button onClick={() => void reload()} className="btn btn-ghost btn-sm">
          {t('common.refresh')}
        </button>
      </div>

      {error && (
        <div className="px-3.5 py-2.5 rounded-md bg-err-soft border border-err text-err text-xs font-mono">
          {error}
        </div>
      )}

      {!loaded ? (
        <div className="rounded-lg border border-subtle bg-surface py-8 text-center text-sm text-fg-tertiary">
          {t('common.loading')}
        </div>
      ) : isEmpty ? (
        <div className="rounded-lg border border-subtle bg-surface py-12 text-center">
          <div className="text-md font-semibold text-fg-secondary mb-1.5">
            {t('queue.jobs.empty')}
          </div>
          <div className="text-sm text-fg-tertiary">{t('queue.jobs.emptyHint')}</div>
        </div>
      ) : (
        <>
          {live.length > 0 && (
            <section className="flex flex-col gap-2">
              <h3 className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                {t('queue.sectionActive')} ({live.length})
              </h3>
              {live.map(renderRow)}
            </section>
          )}

          <section className="flex flex-col gap-2">
            <h3 className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
              {t('queue.sectionHistory')} ({history.total})
            </h3>
            {history.items.length === 0 ? (
              <div className="rounded-lg border border-subtle bg-surface py-8 text-center text-sm text-fg-tertiary">
                {t('queue.noMatch')}
              </div>
            ) : (
              history.items.map(renderRow)
            )}
            {history.total > HISTORY_PAGE_SIZE && (
              <div className="flex items-center justify-between text-[11px] text-fg-tertiary pt-1">
                <span>{t('queue.pageIndicator', { page: history.page, pages: totalPages })}</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setHistoryPage((p) => Math.max(1, p - 1))}
                    disabled={history.page <= 1}
                    className="btn btn-ghost"
                    style={{ padding: '2px 10px', fontSize: 11 }}
                    data-testid="jobs-history-prev"
                  >
                    {t('queue.prevPage')}
                  </button>
                  <button
                    onClick={() => setHistoryPage((p) => Math.min(totalPages, p + 1))}
                    disabled={history.page >= totalPages}
                    className="btn btn-ghost"
                    style={{ padding: '2px 10px', fontSize: 11 }}
                    data-testid="jobs-history-next"
                  >
                    {t('queue.nextPage')}
                  </button>
                </div>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  )
}
