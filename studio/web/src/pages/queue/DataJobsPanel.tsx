/** 数据任务视图（队列页「数据任务」，= light + io 档）。
 *
 *  R-5 台账合并终态：与 GPU 视图**同源** —— 读 /api/queue（resource_class=data）
 *  的 tasks 行，单一 ID 空间；行点击 → /queue/{id} 统一详情页；取消走
 *  /api/queue/{id}/cancel。kind 过滤 / 项目名搜索由 Queue 页 header 漏斗下发。
 *
 *  视图形态（分区/行样式/操作 icon/分页底栏）自 P-G 定稿后不变（用户感知锚点）。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  api, type QueueHistoryPage, type Task, type TaskType,
} from '../../api/client'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import {
  DATA_VIEW_KINDS, JOB_STATUS_TONE, fmtJobAgo, fmtJobDuration, jobJumpPath,
} from './jobUtils'

export default function DataJobsPanel({
  kind, q, historyPage, pageSize, onHistoryTotal, refreshToken,
}: {
  /** kind 过滤（Queue 页 header 漏斗下发；null = 全部数据档）。 */
  kind: TaskType | null
  /** 项目 title/slug 搜索（已防抖；undefined = 不过滤）。 */
  q?: string
  /** 历史分页（分页底栏在 Queue 页，与 GPU 视图同款；total 经 onHistoryTotal 回报）。 */
  historyPage: number
  pageSize: number
  onHistoryTotal: (total: number) => void
  /** 递增触发重拉（Queue 页 header 刷新按钮）。 */
  refreshToken: number
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [live, setLive] = useState<Task[]>([])
  const [history, setHistory] = useState<QueueHistoryPage>({
    items: [], total: 0, page: 1, page_size: 20,
  })
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [projectTitles, setProjectTitles] = useState<Record<number, string>>({})
  const reloadTimer = useRef<number | null>(null)

  const reload = useCallback(async () => {
    try {
      const [l, h] = await Promise.all([
        api.listQueueLive(q, kind ?? undefined, 'data'),
        api.listQueueHistory({
          page: historyPage, pageSize, q,
          type: kind ?? undefined, resourceClass: 'data',
        }),
      ])
      setLive(l); setHistory(h); onHistoryTotal(h.total); setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoaded(true)
    }
  }, [kind, q, historyPage, pageSize, onHistoryTotal])
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
    // R-5 过渡期作业仍发 job_* 事件；task_* 也听，双保险（防抖合并）。
    if (evt.type === 'job_state_changed' || evt.type === 'task_state_changed') {
      if (reloadTimer.current) return
      reloadTimer.current = window.setTimeout(() => {
        reloadTimer.current = null
        void reloadRef.current()
      }, 100)
    }
  })

  const cancelJob = async (task: Task) => {
    const ok = await confirm(
      t('queue.jobs.cancelConfirm', { id: task.id }),
      { okText: t('queue.jobs.cancelOk') },
    )
    if (!ok) return
    try {
      await api.cancelTask(task.id)
      toast(t('queueDetail.cancelSent'), 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const isEmpty = loaded && live.length === 0 && history.total === 0 && !kind && !q

  const KIND_LABEL = useMemo(() => Object.fromEntries(
    DATA_VIEW_KINDS.map((k) => [k, t(`queue.jobs.kind.${k}`)]),
  ) as Record<string, string>, [t])

  const renderRow = (task: Task) => {
    const kindOf = task.task_type ?? 'train'
    const isLive = task.status === 'running' || task.status === 'pending'
    const jump = jobJumpPath(task)
    const projectLabel = task.project_id != null
      ? (projectTitles[task.project_id] ?? `#${task.project_id}`)
      : '—'
    const STATUS_LABEL: Record<string, string> = {
      pending: t('status.queued'), running: t('status.running'), done: t('status.done'),
      failed: t('status.failed'), canceled: t('status.canceled'),
      paused: t('status.paused'), scheduled: t('status.scheduled'),
    }
    return (
      <button
        key={task.id}
        onClick={() => navigate(`/queue/${task.id}`)}
        title={t('queue.taskDetailTooltip')}
        className={`card card-hover block overflow-hidden text-left p-0 cursor-pointer ${task.status === 'running' ? 'border border-accent bg-accent-soft' : 'border border-subtle bg-surface'}`}
        data-testid={`job-row-${task.id}`}
      >
        <div
          className="px-[22px] py-4 grid gap-3 items-center"
          style={{ gridTemplateColumns: '48px minmax(0,1fr) 96px 150px 120px' }}
        >
          <span className={`font-mono text-sm ${task.status === 'running' ? 'text-accent font-semibold' : 'text-fg-tertiary'}`}>
            #{task.id}
          </span>
          <div style={{ minWidth: 0 }}>
            <div className="font-semibold text-fg-primary text-sm overflow-hidden text-ellipsis whitespace-nowrap">
              {KIND_LABEL[kindOf] ?? kindOf}
            </div>
            <div className="text-xs text-fg-tertiary mt-0.5 overflow-hidden text-ellipsis whitespace-nowrap">
              {projectLabel}{task.version_id ? ` · v${task.version_id}` : ''}
            </div>
          </div>
          <span className={`badge badge-${JOB_STATUS_TONE[task.status] ?? 'neutral'} text-xs text-center`}>
            {task.status === 'running' && <span className="dot dot-running" />}
            {STATUS_LABEL[task.status] ?? task.status}
          </span>
          <span className="font-mono text-xs text-fg-tertiary text-right">
            {task.status === 'running' ? (
              fmtJobDuration(task.started_at, null)
            ) : task.finished_at ? (
              <>
                {fmtJobDuration(task.started_at, task.finished_at)}
                <br />
                {fmtJobAgo(task.finished_at)}
              </>
            ) : '—'}
          </span>
          {/* action 列：取消（live）+ 跳转原生页，icon 化 hover 显文字（既有范式） */}
          <div className="flex items-center justify-end gap-1.5">
            {isLive && (
              <button
                onClick={(e) => { e.stopPropagation(); void cancelJob(task) }}
                className="btn btn-ghost btn-sm px-2 text-err"
                title={t('queue.jobs.cancel')}
                aria-label={t('queue.jobs.cancel')}
                data-testid={`job-cancel-btn-${task.id}`}
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
                data-testid={`job-jump-btn-${task.id}`}
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
          </section>
        </>
      )}
    </div>
  )
}
