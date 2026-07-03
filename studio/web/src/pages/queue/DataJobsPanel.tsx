/** 0.17 P-G — 数据作业只读区（队列页「数据作业」视图）。
 *
 *  project_jobs（download/preprocess/tag/reg_build/eval_*）的统一只读呈现：
 *  进行中 + 历史（后端分页）。kind 过滤由 Queue 页 header 漏斗下发（与任务
 *  视图同位交互）。行点击 → /queue/jobs/{id} 详情页（summary/参数/日志），
 *  与 GPU 任务行为同构。行操作 icon 化：取消（confirm）+ 跳转原生步骤页。
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
import {
  JOB_KINDS, JOB_STATUS_TONE, fmtJobAgo, fmtJobDuration, jobJumpPath,
} from './jobUtils'

const HISTORY_PAGE_SIZE = 20

export default function DataJobsPanel({
  kind, refreshToken,
}: {
  /** kind 过滤（Queue 页 header 漏斗下发；null = 全部）。 */
  kind: JobKind | null
  /** 递增触发重拉（Queue 页 header 刷新按钮）。 */
  refreshToken: number
}) {
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
  const [historyPage, setHistoryPage] = useState(1)
  const [projectTitles, setProjectTitles] = useState<Record<number, string>>({})
  const reloadTimer = useRef<number | null>(null)

  // kind 过滤变化回第 1 页。
  useEffect(() => { setHistoryPage(1) }, [kind])

  const reload = useCallback(async () => {
    try {
      const [l, h] = await Promise.all([
        api.listJobsLive(kind ?? undefined),
        api.listJobsHistory({
          page: historyPage, pageSize: HISTORY_PAGE_SIZE, kind: kind ?? undefined,
        }),
      ])
      setLive(l); setHistory(h); setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoaded(true)
    }
  }, [kind, historyPage])
  const reloadRef = useRef(reload); reloadRef.current = reload

  useEffect(() => { void reload() }, [reload, refreshToken])

  // 项目名映射（行上显示项目标题；拿不到就回落 #pid）。
  useEffect(() => {
    api.listProjects()
      .then((items) => setProjectTitles(
        Object.fromEntries(items.map((p) => [p.id, p.title])),
      ))
      .catch(() => {})
  }, [])

  useEventStream((evt) => {
    if (evt.type === 'job_state_changed') {
      if (reloadTimer.current) return
      reloadTimer.current = window.setTimeout(() => {
        reloadTimer.current = null
        void reloadRef.current()
      }, 100)
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
  const isEmpty = loaded && live.length === 0 && history.total === 0 && !kind

  const KIND_LABEL = useMemo(() => Object.fromEntries(
    JOB_KINDS.map((k) => [k, t(`queue.jobs.kind.${k}`)]),
  ) as Record<JobKind, string>, [t])

  const renderRow = (job: Job) => {
    const isLive = job.status === 'running' || job.status === 'pending'
    const jump = jobJumpPath(job)
    const projectLabel = projectTitles[job.project_id] ?? `#${job.project_id}`
    const STATUS_LABEL: Record<JobStatus, string> = {
      pending: t('status.queued'), running: t('status.running'), done: t('status.done'),
      failed: t('status.failed'), canceled: t('status.canceled'),
    }
    return (
      <button
        key={job.id}
        onClick={() => navigate(`/queue/jobs/${job.id}`)}
        title={t('queue.taskDetailTooltip')}
        className={`card card-hover block overflow-hidden text-left p-0 cursor-pointer ${job.status === 'running' ? 'border border-accent bg-accent-soft' : 'border border-subtle bg-surface'}`}
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
          <span className={`badge badge-${JOB_STATUS_TONE[job.status]} text-xs text-center`}>
            {job.status === 'running' && <span className="dot dot-running" />}
            {STATUS_LABEL[job.status]}
          </span>
          <span className="font-mono text-xs text-fg-tertiary text-right">
            {job.status === 'running' ? (
              fmtJobDuration(job.started_at, null)
            ) : job.finished_at ? (
              <>
                {fmtJobDuration(job.started_at, job.finished_at)}
                <br />
                {fmtJobAgo(job.finished_at)}
              </>
            ) : '—'}
          </span>
          {/* action 列：取消（live）+ 跳转原生页，icon 化 hover 显文字（既有范式） */}
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
    )
  }

  return (
    <div className="flex flex-col gap-4" data-testid="data-jobs-panel">
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
