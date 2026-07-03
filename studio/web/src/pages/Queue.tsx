import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  api, type QueueHistoryPage, type QueueHoldState, type Task,
  type TaskStatus, type TaskType,
} from '../api/client'
import { DATA_VIEW_KINDS } from './queue/jobUtils'
import { HoldQueueModal, type HoldDecision } from '../components/HoldQueueModal'
import { PauseConfirmModal } from '../components/PauseConfirmModal'
import { PauseProgressModal } from '../components/PauseProgressModal'
import StepShell from '../components/StepShell'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { useEvaluatingTasks, type EvalProgress } from '../lib/useEvalProgress'
import { useLocalStorageState } from '../lib/useLocalStorageState'
import DataJobsPanel from './queue/DataJobsPanel'

// GPU 视图（exclusive 档）的行类型。R-5：eval_samples（评估出图，底模级）
// 随台账合并归位本视图（用户感知锚点 §4-2）。
type TaskKind = 'train' | 'reg_ai' | 'generate' | 'eval_samples'

const STATUS_TONE: Record<TaskStatus, string> = {
  pending:   'neutral',
  running:   'accent',
  done:      'ok',
  failed:    'err',
  canceled:  'neutral',
  paused:    'warn',
  scheduled: 'neutral',
}

/** 读后端权威 task_type；老行 backfill 为 'train'，`?? 'train'` 仅作类型兜底。
 *  导出供 Queue.test.tsx 直接测（不再受 config_name 影响是本函数的契约）。 */
export function taskKind(task: Task): TaskKind {
  return (task.task_type ?? 'train') as TaskKind
}

// 0.17 P-E — 历史分页可选每页条数。
const HISTORY_PAGE_SIZES = [20, 50, 100]
// 0.17 P-F — 队列默认只看训练（generate/reg_ai 短任务多、会淹没列表）。
const DEFAULT_TYPE_FILTER: TaskKind = 'train'

function fmtAgo(ts: number): string {
  const sec = Math.max(0, Date.now() / 1000 - ts)
  if (sec < 60) return '刚刚'
  if (sec < 3600) return `${Math.floor(sec / 60)}m 前`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h 前`
  return `${Math.floor(sec / 86400)}d 前`
}

/** scheduled task 的计划时间：绝对（本地时区）+「约 X 后」相对提示。 */
function fmtScheduledAbs(ts: number): string {
  return new Date(ts * 1000).toLocaleString('zh-CN', {
    hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  })
}

function fmtUntil(ts: number): string {
  const sec = ts - Date.now() / 1000
  if (sec <= 0) return '即将开始'
  if (sec < 60) return '1m 内'
  if (sec < 3600) return `${Math.ceil(sec / 60)}m 后`
  if (sec < 86400) {
    const h = Math.floor(sec / 3600); const m = Math.round((sec % 3600) / 60)
    return m ? `${h}h ${m}m 后` : `${h}h 后`
  }
  return `${Math.ceil(sec / 86400)}d 后`
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

function fmtDurationShort(ms: number): string {
  if (ms < 60e3) return `${Math.round(ms / 1e3)}s`
  if (ms < 3600e3) return `${Math.round(ms / 60e3)}m`
  return `${(ms / 3600e3).toFixed(1)}h`
}

/** running task 的「已运行 Xm」标签；非 running 返 null。 */
function estimateEta(task: Task): string | null {
  if (task.status !== 'running' || !task.started_at) return null
  const elapsed = (Date.now() / 1000 - task.started_at) * 1000
  return `已运行 ${fmtDurationShort(elapsed)}`
}

/** 历史分页固定底栏（GPU / 数据两个视图共用同款，P-G 反馈统一）。
 *  只要 total 超过最小每页数就常显；testid 复用（同一时刻只渲染一个视图）。 */
function PaginationBar({
  page, total, pageSize, onPage, onPageSize,
}: {
  page: number
  total: number
  pageSize: number
  onPage: (updater: (p: number) => number) => void
  onPageSize: (n: number) => void
}) {
  const { t } = useTranslation()
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  return (
    <div className="shrink-0 -mx-6 -mb-6 px-6 py-1 border-t border-subtle flex items-center justify-between flex-wrap gap-2 bg-canvas text-[11px]">
      <div className="flex items-center gap-2 text-fg-tertiary">
        <span>{t('queue.pageIndicator', { page, pages: totalPages })}</span>
        <select
          value={pageSize}
          onChange={(e) => onPageSize(Number(e.target.value))}
          className="input"
          style={{ width: 'auto', padding: '1px 6px', fontSize: 11 }}
          data-testid="history-page-size"
        >
          {HISTORY_PAGE_SIZES.map((n) => (
            <option key={n} value={n}>{t('queue.perPage', { n })}</option>
          ))}
        </select>
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onPage((p) => Math.max(1, p - 1))}
          disabled={page <= 1}
          className="btn btn-ghost"
          style={{ padding: '2px 10px', fontSize: 11 }}
          data-testid="history-prev"
        >
          {t('queue.prevPage')}
        </button>
        <button
          onClick={() => onPage((p) => Math.min(totalPages, p + 1))}
          disabled={page >= totalPages}
          className="btn btn-ghost"
          style={{ padding: '2px 10px', fontSize: 11 }}
          data-testid="history-next"
        >
          {t('queue.nextPage')}
        </button>
      </div>
    </div>
  )
}

/** 队列行卡片。0.17 P-A 从 QueuePage 内联 map 抽出，供三个分区复用同一行渲染。
 *  monitor 只对 running 且 id===runningTaskId 的那行有意义；evalInfo 只对 terminal
 *  且仍在评估的行有值。 */
function QueueTaskRow({
  task, runningTaskId, monitor, evalInfo, isWaitingForRelease, prevAhead,
  onOpen, onResume, onCancelPaused, onStartNow, onCancelScheduled,
}: {
  task: Task
  runningTaskId: number | null
  monitor: { step?: number | null; total_steps?: number | null } | null
  evalInfo?: EvalProgress
  isWaitingForRelease: boolean
  prevAhead: number
  onOpen: (id: number) => void
  onResume: (task: Task) => void | Promise<void>
  onCancelPaused: (task: Task) => void | Promise<void>
  onStartNow: (task: Task) => void | Promise<void>
  onCancelScheduled: (task: Task) => void | Promise<void>
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const STATUS_LABEL: Record<TaskStatus, string> = {
    pending:  t('status.queued'), running: t('status.running'),
    done:     t('status.done'),   failed:  t('status.failed'),
    canceled: t('status.canceled'), paused: t('status.paused'),
    scheduled: t('status.scheduled'),
  }
  const KIND_LABEL: Record<TaskKind, string> = {
    train: t('queue.typeTrain'), reg_ai: t('queue.typeReg'),
    generate: t('queue.typeGenerate'), eval_samples: t('queue.jobs.kind.eval_samples'),
  }
  const isRunning = task.status === 'running'
  const isPaused = task.status === 'paused'
  const isScheduled = task.status === 'scheduled'
  const isTerminal = ['done', 'failed', 'canceled'].includes(task.status)
  const hasProject = !!(task.project_id && task.version_id)
  const kind = taskKind(task)
  const eta = estimateEta(task)
  const tone = STATUS_TONE[task.status]

  // 0.17 P-H 跳转列：按类型跳原生页看结果/配置。train→训练配置页、reg_ai→正则集、
  // generate→那次出图（?task= 深链回看）。
  const jump: { path: string; label: string } | null =
    kind === 'generate'
      ? { path: `/tools/generate?task=${task.id}`, label: t('queue.jumpGenerate') }
      : kind === 'reg_ai' && hasProject
        ? { path: `/projects/${task.project_id}/v/${task.version_id}/reg`, label: t('queue.jumpReg') }
        : (kind === 'train' || kind === 'eval_samples') && hasProject
          // eval_samples 的结果面板挂在版本训练页（EvalMetricsPanel），同 train 深链
          ? { path: `/projects/${task.project_id}/v/${task.version_id}/train`, label: t('queue.jumpTrain') }
          : null

  return (
    <button
      onClick={() => onOpen(task.id)}
      title={t('queue.taskDetailTooltip')}
      className={`card card-hover block overflow-hidden text-left p-0 cursor-pointer ${isRunning ? 'border border-accent bg-accent-soft' : 'border border-subtle bg-surface'}`}
    >
      <div
        className="px-[22px] py-4 grid gap-3 items-center"
        style={{ gridTemplateColumns: '48px minmax(0,1fr) 88px 96px 150px 128px 176px' }}
      >
        <span className={`font-mono text-sm ${isRunning ? 'text-accent font-semibold' : 'text-fg-tertiary font-normal'}`}>
          #{task.id}
        </span>

        <div style={{ minWidth: 0 }}>
          <div className="font-semibold text-fg-primary text-sm overflow-hidden text-ellipsis whitespace-nowrap">
            {task.name}
          </div>
          <div className="font-mono text-xs text-fg-tertiary mt-0.5 overflow-hidden text-ellipsis whitespace-nowrap">
            {task.config_name}
          </div>
        </div>

        {/* 0.17 类型列（原在 title 下方，改独立 column） */}
        <span
          className="text-xs text-fg-secondary overflow-hidden text-ellipsis whitespace-nowrap"
          title={KIND_LABEL[kind]}
        >
          {KIND_LABEL[kind]}
        </span>

        <span className={`badge badge-${tone} text-xs text-center`}>
          {isRunning && <span className="dot dot-running" />}
          {STATUS_LABEL[task.status]}
        </span>

        <div className="text-sm text-fg-secondary" style={{ minWidth: 0 }}>
          {isRunning ? (
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-fg-tertiary text-xs">
                {(() => {
                  if (
                    task.id === runningTaskId &&
                    monitor?.step != null &&
                    monitor.total_steps != null &&
                    monitor.total_steps > 0
                  ) {
                    return `step ${monitor.step.toLocaleString()} / ${monitor.total_steps.toLocaleString()}`
                  }
                  return fmtDuration(task.started_at, null)
                })()}
              </span>
              <div className="h-1 bg-overlay rounded-sm overflow-hidden">
                {(() => {
                  const haveSteps =
                    task.id === runningTaskId &&
                    monitor?.step != null &&
                    monitor.total_steps != null &&
                    monitor.total_steps > 0
                  if (haveSteps) {
                    const pct = Math.max(
                      0,
                      Math.min(100, (monitor!.step! / monitor!.total_steps!) * 100),
                    )
                    return <div className="h-full bg-accent rounded-sm" style={{ width: `${pct}%` }} />
                  }
                  return <div className="h-full bg-accent/40 rounded-sm animate-pulse" style={{ width: '20%' }} />
                })()}
              </div>
            </div>
          ) : task.error_msg ? (
            <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap block text-xs">
              {task.error_msg}
            </span>
          ) : isPaused ? (
            <span className="text-xs text-warn">
              {t('queue.pausedAtStep', {
                step: task.paused_step ?? 0,
                time: task.paused_at ? fmtAgo(task.paused_at) : '',
              })}
            </span>
          ) : isTerminal ? (
            evalInfo?.active ? (
              <span className="text-accent text-xs flex items-center gap-1" title={t('eval.evaluatingHint')}>
                <span className="dot dot-running" />
                {t('eval.evaluatingProgress', {
                  done: evalInfo.done,
                  total: evalInfo.total,
                })}
              </span>
            ) : (
              <span className="font-mono text-fg-tertiary text-xs">
                {t('queue.duration', { time: fmtDuration(task.started_at, task.finished_at) })}
              </span>
            )
          ) : (
            <span className="text-fg-tertiary text-xs">—</span>
          )}
        </div>

        <span className="font-mono text-sm text-fg-tertiary text-right">
          {isRunning ? (
            <>
              {eta && <span className="text-accent">{eta}</span>}
              {eta && <br />}
              <span className="text-xs">{fmtAgo(task.started_at!)} 开始</span>
            </>
          ) : isPaused ? (
            <span className="flex flex-col items-end gap-1">
              <span className="flex gap-1.5">
                <button
                  onClick={(e) => { e.stopPropagation(); void onResume(task) }}
                  className="btn btn-secondary btn-xs"
                  title={t('queue.resumeHint')}
                  data-testid={`resume-btn-${task.id}`}
                >
                  {t('queue.resume')}
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); void onCancelPaused(task) }}
                  className="btn btn-ghost btn-xs text-err"
                  title={t('queue.cancelPausedHint')}
                >
                  {t('queue.cancelPaused')}
                </button>
              </span>
              {isWaitingForRelease && (
                <span className="text-xs text-fg-tertiary">
                  {t('queue.waitingForRelease')}
                </span>
              )}
            </span>
          ) : isScheduled ? (
            /* 0.17 P-B — 计划任务：计划时间 + 倒计时（操作在右侧 action 列） */
            <span className="flex flex-col items-end gap-0.5">
              {task.scheduled_at ? (
                <>
                  <span>{fmtScheduledAbs(task.scheduled_at)}</span>
                  <span className="text-xs text-fg-tertiary font-normal">
                    {fmtUntil(task.scheduled_at)}
                  </span>
                </>
              ) : (
                <span>—</span>
              )}
            </span>
          ) : task.finished_at ? (
            <span className="flex flex-col items-end gap-1">
              {/* ADR 0006 Addendum 2 — failed/canceled 且恢复点在盘 → 继续训练。 */}
              {task.is_resumable && (
                <button
                  onClick={(e) => { e.stopPropagation(); void onResume(task) }}
                  className="btn btn-secondary btn-xs"
                  title={t('queue.resumeTerminalHint')}
                  data-testid={`resume-btn-${task.id}`}
                >
                  {t('queue.resumeTerminal')}
                </button>
              )}
              <span>
                <span>{fmtAgo(task.finished_at)}</span>
                <br />
                <span className="text-xs text-fg-tertiary">{t('status.done')}</span>
              </span>
            </span>
          ) : (
            <span className="flex flex-col items-end gap-0.5">
              <span>{t('queue.ahead', { n: prevAhead })}</span>
              {isWaitingForRelease && (
                <span className="text-xs text-warn">
                  {t('queue.waitingForRelease')}
                </span>
              )}
            </span>
          )}
        </span>

        {/* 0.17 action 列：跳转 + scheduled 的立即开始/取消计划，全 icon 化
            （hover title 显示文字）。 */}
        <div className="flex items-center justify-end gap-1.5">
          {isScheduled && (
            <>
              <button
                onClick={(e) => { e.stopPropagation(); void onStartNow(task) }}
                className="btn btn-ghost btn-sm px-2"
                title={t('queue.startNow')}
                aria-label={t('queue.startNow')}
                data-testid={`startnow-btn-${task.id}`}
              >
                <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); void onCancelScheduled(task) }}
                className="btn btn-ghost btn-sm px-2 text-err"
                title={t('queue.cancelScheduled')}
                aria-label={t('queue.cancelScheduled')}
                data-testid={`cancel-scheduled-btn-${task.id}`}
              >
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </>
          )}
          {jump && (
            <button
              onClick={(e) => { e.stopPropagation(); navigate(jump.path) }}
              className="btn btn-ghost btn-sm px-2"
              title={jump.label}
              aria-label={jump.label}
              data-testid={`jump-btn-${task.id}`}
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

export default function QueuePage() {
  const { t } = useTranslation()
  // 0.17 P-A/P-E — 队列拆两条数据源：live（进行中+等待，不分页）、history（已结束，
  // 后端分页）。搜索 + 历史子过滤走后端，保证分页 total 与结果一致。
  const [live, setLive] = useState<Task[]>([])
  const [history, setHistory] = useState<QueueHistoryPage>({
    items: [], total: 0, page: 1, page_size: HISTORY_PAGE_SIZES[0],
  })
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  // 搜索（防抖后进后端）+ 历史分页 / 终态子过滤。0.17 item4：过滤条件持久化到
  // localStorage，切走队列页再回来不丢（page 不持久，回来回第 1 页）。
  const [search, setSearch] = useLocalStorageState('studio:queue:search', '')
  const [searchDebounced, setSearchDebounced] = useState(search)
  const [historyStatus, setHistoryStatus] =
    useLocalStorageState<TaskStatus | null>('studio:queue:historyStatus', null)
  // 0.17 P-F 类型过滤（train/reg_ai/generate），跨 live + history 走后端。默认只看
  // 训练（generate/reg_ai 短任务多、会淹没列表；用户可切「全部」看全类型）。
  const [typeFilter, setTypeFilter] =
    useLocalStorageState<TaskKind | null>('studio:queue:typeFilter', DEFAULT_TYPE_FILTER)
  const [historyPage, setHistoryPage] = useState(1)
  const [historyPageSize, setHistoryPageSize] =
    useLocalStorageState('studio:queue:pageSize', HISTORY_PAGE_SIZES[0])
  // 过滤行折叠（与项目页一致：默认收起，header 漏斗按钮开关，收起且有筛选时带小圆点）。
  const [filtersOpen, setFiltersOpen] = useState(false)
  // 0.17 P-G — 数据作业视图开关（header toggle，对齐项目页「已归档」模式）+
  // kind 过滤（漏斗下发给 DataJobsPanel）+ 刷新令牌。
  const [queueTab, setQueueTab] =
    useLocalStorageState<'tasks' | 'jobs'>('studio:queue:tab', 'tasks')
  const [jobsKind, setJobsKind] =
    useLocalStorageState<TaskType | null>('studio:queue:jobsKind', null)
  const [jobsSearch, setJobsSearch] = useLocalStorageState('studio:queue:jobsSearch', '')
  const [jobsSearchDebounced, setJobsSearchDebounced] = useState(jobsSearch)
  const [jobsRefreshToken, setJobsRefreshToken] = useState(0)
  // 数据任务历史分页（与 GPU 视图同款固定底栏；total 由 Panel 拉取后回报）。
  const [jobsHistoryPage, setJobsHistoryPage] = useState(1)
  const [jobsPageSize, setJobsPageSize] =
    useLocalStorageState('studio:queue:jobsPageSize', HISTORY_PAGE_SIZES[0])
  const [jobsHistoryTotal, setJobsHistoryTotal] = useState(0)

  // 数据任务搜索防抖 300ms（同任务视图搜索）；过滤条件变化回第 1 页。
  useEffect(() => {
    const id = window.setTimeout(() => setJobsSearchDebounced(jobsSearch), 300)
    return () => window.clearTimeout(id)
  }, [jobsSearch])
  useEffect(() => { setJobsHistoryPage(1) }, [jobsKind, jobsSearchDebounced])
  const reloadTimer = useRef<number | null>(null)
  const { toast } = useToast()
  const { confirm } = useDialog()
  const navigate = useNavigate()

  // ADR 0006：队列挂起状态，banner + holdModal 用。
  const [holdState, setHoldState] = useState<QueueHoldState | null>(null)
  const [holdModalOpen, setHoldModalOpen] = useState(false)
  const [pausingTaskId, setPausingTaskId] = useState<number | null>(null)
  const [pauseConfirmTaskId, setPauseConfirmTaskId] = useState<number | null>(null)

  const reloadHold = useCallback(async () => {
    try {
      const s = await api.getQueueHold()
      setHoldState(s)
    } catch {
      setHoldState(null)
    }
  }, [])

  // R-5：GPU 视图 = exclusive 档（「全部」= 档位全集，含 eval_samples）。
  const reloadLive = useCallback(async () => {
    try {
      setLive(await api.listQueueLive(
        searchDebounced || undefined, typeFilter ?? undefined, 'exclusive',
      ))
      setError(null)
    } catch (e) { setError(String(e)) }
  }, [searchDebounced, typeFilter])

  const reloadHistory = useCallback(async () => {
    try {
      const r = await api.listQueueHistory({
        page: historyPage, pageSize: historyPageSize,
        q: searchDebounced || undefined, status: historyStatus ?? undefined,
        type: typeFilter ?? undefined, resourceClass: 'exclusive',
      })
      setHistory(r); setError(null)
    } catch (e) { setError(String(e)) }
  }, [historyPage, historyPageSize, searchDebounced, historyStatus, typeFilter])

  const reload = useCallback(async () => {
    await Promise.all([reloadLive(), reloadHistory()])
  }, [reloadLive, reloadHistory])

  // SSE 处理器可能被 useEventStream 捕获一次；用 ref 取最新 reload 避免 stale 闭包
  // （reloadLive/reloadHistory 随过滤条件变 identity）。
  const reloadLiveRef = useRef(reloadLive); reloadLiveRef.current = reloadLive
  const reloadHistoryRef = useRef(reloadHistory); reloadHistoryRef.current = reloadHistory

  // 搜索防抖 300ms → 进后端；改搜索词回到第 1 页。
  useEffect(() => {
    const id = window.setTimeout(() => {
      setSearchDebounced(search); setHistoryPage(1)
    }, 300)
    return () => window.clearTimeout(id)
  }, [search])

  useEffect(() => { void reloadLive() }, [reloadLive])
  useEffect(() => {
    void (async () => { await reloadHistory(); setLoaded(true) })()
  }, [reloadHistory])

  useEventStream(
    (evt) => {
      if (
        evt.type === 'task_state_changed' ||
        // R-5：数据作业（含 GPU 视图里的 eval_samples）仍发 job_* 事件
        evt.type === 'job_state_changed' ||
        evt.type === 'train_loop_started' ||
        evt.type === 'auto_epoch_backup_written' ||
        evt.type === 'queue_hold_changed'
      ) {
        if (evt.type === 'queue_hold_changed') {
          void reloadHold()
        }
        if (reloadTimer.current) return
        reloadTimer.current = window.setTimeout(() => {
          reloadTimer.current = null
          // task 状态变化可能把 task 移进 history（terminal）或移出 live，两边都刷。
          void reloadLiveRef.current(); void reloadHistoryRef.current()
        }, 100)
      }
    },
    { onOpen: () => { void reloadLiveRef.current(); void reloadHistoryRef.current(); void reloadHold() } },
  )

  useEffect(() => { void reloadHold() }, [reloadHold])

  // running 任务的 id，给 monitor 进度条 / 顶部按钮用。
  const runningTask = useMemo(
    () => live.find((t) => t.status === 'running') ?? null,
    [live],
  )
  const runningTaskId = runningTask?.id ?? null
  const hasRunning = runningTask !== null

  // 2s 时钟 tick：仅触发 re-render 让「已运行 40m」「3h 后」之类相对时间更新；
  // 不发 API。scheduled 的倒计时也靠它。
  const hasScheduled = useMemo(() => live.some((t) => t.status === 'scheduled'), [live])
  useEffect(() => {
    if (!hasRunning && !hasScheduled) return
    const tick = window.setInterval(() => setLive((ls) => [...ls]), 2000)
    return () => window.clearInterval(tick)
  }, [hasRunning, hasScheduled])

  // 评估中可见性：live 里的 done（罕见）+ history 当前页的 done 都纳入探测。
  const evalSource = useMemo(() => [...live, ...history.items], [live, history])
  const evalMap = useEvaluatingTasks(evalSource)
  const { state: monitor } = useMonitorProgress(runningTaskId)

  // live 按 id 倒序拆两段：进行中（running/paused）、等待（pending）。
  const liveSorted = useMemo(() => [...live].sort((a, b) => b.id - a.id), [live])
  const activeItems = useMemo(
    () => liveSorted.filter((t) => t.status === 'running' || t.status === 'paused'),
    [liveSorted],
  )
  const pendingItems = useMemo(
    () => liveSorted.filter((t) => t.status === 'pending'),
    [liveSorted],
  )
  // 0.17 P-B — 计划任务段：按计划时间升序（最先开始的在前），无时间的殿后。
  const scheduledItems = useMemo(
    () => liveSorted
      .filter((t) => t.status === 'scheduled')
      .sort((a, b) => (a.scheduled_at ?? Infinity) - (b.scheduled_at ?? Infinity)),
    [liveSorted],
  )

  const prevCount = useCallback((taskId: number): number => {
    let count = 0
    for (const t of liveSorted) {
      if (t.id === taskId) break
      if (t.status === 'running' || t.status === 'pending') count++
    }
    return count
  }, [liveSorted])


  const requestPause = (task: Task) => {
    setPauseConfirmTaskId(task.id)
  }
  const confirmPause = async () => {
    const taskId = pauseConfirmTaskId
    if (taskId === null) return
    setPauseConfirmTaskId(null)
    setPausingTaskId(taskId)
    try {
      await api.pauseTask(taskId)
      toast(t('queue.pauseSent'), 'success')
    } catch (e) {
      toast(t('queue.pauseFailed', { reason: String(e) }), 'error')
      setPausingTaskId(null)
    }
  }
  // hold-and-pause 快速路径（HoldQueueModal 内已 confirmed）
  const pauseTask = async (task: Task) => {
    setPausingTaskId(task.id)
    try {
      await api.pauseTask(task.id)
      toast(t('queue.pauseSent'), 'success')
    } catch (e) {
      toast(t('queue.pauseFailed', { reason: String(e) }), 'error')
      setPausingTaskId(null)
    }
  }

  const resumeTask = async (task: Task) => {
    try {
      await api.resumeTask(task.id)
      toast(t('queue.resumeSent', { id: task.id }), 'success')
      await reload()
    } catch (e) {
      const msg = String(e)
      if (msg.toLowerCase().includes('missing')) {
        toast(t('queue.resumeFailedMissing'), 'error')
      } else {
        toast(t('queue.resumeFailed', { reason: msg }), 'error')
      }
    }
  }

  // 0.17 P-B — scheduled task 手动提前：立即转 pending 参与排队。
  const startNow = async (task: Task) => {
    const ok = await confirm(
      t('queue.startNowConfirm', { id: task.id }),
      { okText: t('queue.startNow') },
    )
    if (!ok) return
    try {
      await api.startTaskNow(task.id)
      toast(t('queue.startNowSent', { id: task.id }), 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const cancelScheduled = async (task: Task) => {
    const ok = await confirm(
      t('queue.cancelScheduledConfirm', { id: task.id }),
      { tone: 'warn', okText: t('queue.cancelScheduled') },
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

  const cancelPaused = async (task: Task) => {
    const ok = await confirm(
      `${t('queue.cancelPaused')} #${task.id}？${t('queue.cancelPausedHint')}`,
      { tone: 'warn', okText: t('queue.cancelPaused') },
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

  // ADR §4.4 hold 队列：弹 confirmation modal，根据 modal 内决策调 hold + 可选 pause
  const onHoldConfirm = async (decision: HoldDecision) => {
    setHoldModalOpen(false)
    try {
      await api.holdQueue()
      toast(t('queue.holdSet'), 'success')
    } catch (e) {
      toast(String(e), 'error')
      return
    }
    if (decision.kind === 'hold-and-pause') {
      await pauseTask({ id: decision.taskId } as Task)
    }
    await reloadHold()
    await reload()
  }

  const releaseQueue = async () => {
    try {
      await api.releaseQueue()
      toast(t('queue.holdReleased'), 'success')
      await reloadHold()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const cancelRunning = async () => {
    if (!runningTask) return
    const ok = await confirm(
      `取消当前任务 #${runningTask.id}？任务会在安全点停止，且无法恢复（重启训练会从 0 开始）。`,
      { tone: 'warn', okText: t('queue.cancelCurrent') },
    )
    if (!ok) return
    setBusy(true)
    try {
      await api.cancelTask(runningTask.id)
      toast(t('queueDetail.cancelSent'), 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const isEmpty =
    live.length === 0 && history.total === 0
    && !searchDebounced && !historyStatus && typeFilter === DEFAULT_TYPE_FILTER

  const renderRow = (task: Task) => (
    <QueueTaskRow
      key={task.id}
      task={task}
      runningTaskId={runningTaskId}
      monitor={monitor}
      evalInfo={evalMap.get(task.id)}
      isWaitingForRelease={task.status === 'pending' && holdState?.held === true}
      prevAhead={prevCount(task.id)}
      onOpen={(id) => navigate(`/queue/${id}`)}
      onResume={resumeTask}
      onCancelPaused={cancelPaused}
      onStartNow={startNow}
      onCancelScheduled={cancelScheduled}
    />
  )

  // 收起过滤行时在漏斗按钮上点小圆点：typeFilter 默认 train 不算「筛选中」。
  const filtering =
    search.trim() !== '' || historyStatus !== null || typeFilter !== DEFAULT_TYPE_FILTER

  return (
    <StepShell
      idx={-1}
      /* 0.17 P-G — title/描述随视图切换：明确这是两条独立队列，不是同一列表的
         类型 filter。 */
      title={queueTab === 'jobs' ? t('queue.titleJobs') : t('queue.title')}
      subtitle={queueTab === 'jobs' ? t('queue.descriptionJobs') : t('queue.description')}
      actions={
        <>
          {queueTab === 'jobs' && <>
            {/* 数据作业视图：漏斗（kind 过滤）+ 刷新，与任务视图同位交互。 */}
            <button
              className={`btn btn-sm ${filtersOpen ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setFiltersOpen((o) => !o)}
              aria-expanded={filtersOpen}
              aria-label={t('queue.filters')}
              title={t('queue.filters')}
              data-testid="queue-filter-toggle"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 4h18l-7 8v6l-4 2v-8L3 4z" />
              </svg>
              {!filtersOpen && (jobsKind !== null || jobsSearch.trim() !== '') && (
                <span className="dot dot-running" aria-label={t('queue.filtersActive')} />
              )}
            </button>
            <button
              onClick={() => setJobsRefreshToken((n) => n + 1)}
              className="btn btn-ghost btn-sm"
            >
              {t('common.refresh')}
            </button>
          </>}
          {queueTab === 'tasks' && <>
          {/* 过滤漏斗：折叠态不占行，开关过滤行；有筛选生效且收起时带小圆点（与项目页一致）。 */}
          <button
            className={`btn btn-sm ${filtersOpen ? 'btn-secondary' : 'btn-ghost'}`}
            onClick={() => setFiltersOpen((o) => !o)}
            aria-expanded={filtersOpen}
            aria-label={t('queue.filters')}
            title={t('queue.filters')}
            data-testid="queue-filter-toggle"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 4h18l-7 8v6l-4 2v-8L3 4z" />
            </svg>
            {!filtersOpen && filtering && (
              <span className="dot dot-running" aria-label={t('queue.filtersActive')} />
            )}
          </button>
          {runningTask?.is_pausable && (
            <button
              onClick={() => requestPause(runningTask)}
              disabled={busy || pausingTaskId !== null || pauseConfirmTaskId !== null}
              className="btn btn-secondary btn-sm"
              title={t('queue.pauseHint')}
              data-testid="queue-pause-btn"
            >
              {t('queue.pause')}
            </button>
          )}
          {hasRunning && (
            <button
              onClick={() => void cancelRunning()}
              disabled={busy}
              className="btn btn-secondary btn-sm text-warn border-warn"
              title={t('queue.cancelHint')}
            >
              {t('queue.cancelCurrent')}
            </button>
          )}
          {holdState && !holdState.held && (
            <button
              onClick={() => setHoldModalOpen(true)}
              disabled={busy}
              className="btn btn-ghost btn-sm"
              data-testid="queue-hold-btn"
            >
              {t('queue.holdQueue')}
            </button>
          )}
          {holdState && holdState.held && (
            <button
              onClick={() => void releaseQueue()}
              disabled={busy}
              className="btn btn-secondary btn-sm"
              data-testid="queue-release-btn"
            >
              {t('queue.releaseQueue')}
            </button>
          )}
          {/* 队列 JSON 导入/导出已下线（预设池时代遗留：现代任务 config 是
              version 私有、导出恒空导入恒跳过）；后端 route 待单独清理 PR。 */}
          <button onClick={() => void reload()} className="btn btn-ghost btn-sm">{t('common.refresh')}</button>
          </>}
          {/* 0.17 P-G — 视图切换（放最右）：前置切换 icon（行为）+ 目标视图名
              （宾语），读作「切到 X」——名词+后缀箭头会歧义成「当前+动作」。 */}
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setQueueTab(queueTab === 'jobs' ? 'tasks' : 'jobs')}
            aria-pressed={queueTab === 'jobs'}
            data-testid="queue-jobs-toggle"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 1l4 4-4 4" />
              <path d="M3 11V9a4 4 0 0 1 4-4h14" />
              <path d="M7 23l-4-4 4-4" />
              <path d="M21 13v2a4 4 0 0 1-4 4H3" />
            </svg>
            <span>{queueTab === 'jobs' ? t('queue.tabTasks') : t('queue.tabJobs')}</span>
          </button>
        </>
      }
      belowHeader={filtersOpen && (queueTab === 'jobs' ? (
        // 0.17 P-G — 数据作业过滤行：kind 单选（与任务视图的过滤行同位）。
        <div
          className="px-6 py-2 border-b border-subtle flex items-center gap-3"
          data-testid="queue-jobs-filterbar"
        >
          <input
            className="input"
            style={{ width: '60%' }}
            value={jobsSearch}
            onChange={(e) => setJobsSearch(e.target.value)}
            placeholder={t('queue.jobs.searchPlaceholder')}
            aria-label={t('common.search')}
            data-testid="jobs-search"
          />
          <span className="flex-1" />
          <select
            className="input"
            style={{ width: '15%', minWidth: 150 }}
            value={jobsKind ?? 'all'}
            onChange={(e) => {
              const v = e.target.value
              setJobsKind(v === 'all' ? null : (v as TaskType))
            }}
            aria-label={t('queue.typeFilterLabel')}
            data-testid="jobs-kind-filter"
          >
            <option value="all">{t('queue.filterAll')}</option>
            {DATA_VIEW_KINDS.map((k) => (
              <option key={k} value={k}>{t(`queue.jobs.kind.${k}`)}</option>
            ))}
          </select>
        </div>
      ) : (
        // 0.17 P-C/P-F 过滤行 —— 与项目页 FilterBar 一致：header 下全宽条。搜索 60%
        // 下沉后端搜 name/config；类型 select 跨 live+history 按 task_type 过滤；状态
        // select 是历史段终态子过滤。
        <div
          className="px-6 py-2 border-b border-subtle flex items-center gap-3"
          data-testid="queue-filterbar"
        >
          <input
            className="input"
            style={{ width: '60%' }}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t('queue.searchPlaceholder')}
            aria-label={t('common.search')}
            data-testid="queue-search"
          />
          <span className="flex-1" />
          <select
            className="input"
            style={{ width: '10%', minWidth: 120 }}
            value={typeFilter ?? 'all'}
            onChange={(e) => {
              const v = e.target.value
              setTypeFilter(v === 'all' ? null : (v as TaskKind))
              setHistoryPage(1)
            }}
            aria-label={t('queue.typeFilterLabel')}
            data-testid="queue-type-filter"
          >
            <option value="all">{t('queue.filterAll')}</option>
            <option value="train">{t('queue.typeTrain')}</option>
            <option value="reg_ai">{t('queue.typeReg')}</option>
            <option value="generate">{t('queue.typeGenerate')}</option>
            <option value="eval_samples">{t('queue.jobs.kind.eval_samples')}</option>
          </select>
          <select
            className="input"
            style={{ width: '10%', minWidth: 120 }}
            value={historyStatus ?? 'all'}
            onChange={(e) => {
              const v = e.target.value
              setHistoryStatus(v === 'all' ? null : (v as TaskStatus))
              setHistoryPage(1)
            }}
            aria-label={t('common.status')}
            data-testid="history-status-filter"
          >
            <option value="all">{t('queue.filterAll')}</option>
            <option value="done">{t('status.done')}</option>
            <option value="failed">{t('status.failed')}</option>
            <option value="canceled">{t('status.canceled')}</option>
          </select>
        </div>
      ))}
    >
      <div className="flex flex-col gap-2.5 flex-1 min-h-0 overflow-y-auto">
        {/* ADR §4.1 队列挂起 banner — 仅 held=true 时显示，sticky 顶部。
            hold 覆盖全队列（含数据作业派发），两个视图都显示。 */}
        {holdState?.held && (
          <div
            className="sticky top-0 z-10 px-3.5 py-2.5 rounded-md bg-warn-soft border border-warn text-warn text-xs flex items-center justify-between"
            data-testid="queue-hold-banner"
          >
            <span>{t('queue.heldBanner')}</span>
            <button
              onClick={() => void releaseQueue()}
              className="btn btn-ghost btn-xs text-warn"
            >
              {t('queue.releaseQueue')}
            </button>
          </div>
        )}
        {error && (
          <div className="px-3.5 py-2.5 rounded-md bg-err-soft border border-err text-err text-xs font-mono">
            {error}
          </div>
        )}

        {queueTab === 'jobs' ? (
          /* 0.17 P-G — 数据作业只读区（project_jobs）。kind 过滤/刷新由 header 下发。 */
          <DataJobsPanel
            kind={jobsKind}
            q={jobsSearchDebounced || undefined}
            historyPage={jobsHistoryPage}
            pageSize={jobsPageSize}
            onHistoryTotal={setJobsHistoryTotal}
            refreshToken={jobsRefreshToken}
          />
        ) : !loaded ? (
          <div className="rounded-lg border border-subtle bg-surface overflow-hidden">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className={`py-[18px] px-[22px] grid gap-3 items-center opacity-40 ${i < 2 ? 'border-b border-subtle' : 'border-b-0'}`}
                style={{ gridTemplateColumns: '48px minmax(0,1fr) 88px 96px 150px 128px 176px' }}
              >
                <div className="h-3.5 rounded bg-overlay" />
                <div className="flex flex-col gap-1">
                  <div className="h-[13px] rounded bg-overlay w-3/5" />
                  <div className="h-2.5 rounded bg-overlay w-2/5" />
                </div>
                <div className="h-2.5 rounded bg-overlay" />
                <div className="h-5 rounded bg-overlay" />
                <div className="h-2.5 rounded bg-overlay" />
                <div className="h-2.5 rounded bg-overlay" />
                <div className="h-6 rounded bg-overlay" />
              </div>
            ))}
          </div>
        ) : isEmpty ? (
          <div className="rounded-lg border border-subtle bg-surface py-12 text-center">
            <div className="text-md font-semibold text-fg-secondary mb-1.5">
              {t('queue.empty')}
            </div>
            <div className="text-sm text-fg-tertiary">
              {t('queue.emptyHint')}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {/* 进行中（running + paused） */}
            {activeItems.length > 0 && (
              <section className="flex flex-col gap-2">
                <h3 className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                  {t('queue.sectionActive')} ({activeItems.length})
                </h3>
                {activeItems.map(renderRow)}
              </section>
            )}

            {/* 等待入队（pending） */}
            {pendingItems.length > 0 && (
              <section className="flex flex-col gap-2">
                <h3 className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                  {t('queue.sectionWaiting')} ({pendingItems.length})
                </h3>
                {pendingItems.map(renderRow)}
              </section>
            )}

            {/* 计划任务（scheduled，0.17 P-B）——到点自动转入等待入队 */}
            {scheduledItems.length > 0 && (
              <section className="flex flex-col gap-2" data-testid="queue-scheduled-section">
                <h3 className="text-xs font-semibold text-fg-tertiary uppercase tracking-wide">
                  {t('queue.sectionScheduled')} ({scheduledItems.length})
                </h3>
                {scheduledItems.map(renderRow)}
              </section>
            )}

            {/* 历史（terminal，后端分页） */}
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
          </div>
        )}
      </div>

      {/* 0.17 item6：分页下沉成 fixed 底栏（-mx-6/-mb-6 抵消内容区 padding 做全宽贴底）。
          item2：只要历史超过最小每页数就常显（切到 50/100 只剩一页时不消失，能切回
          20）；样式压缩省空间。GPU / 数据两个视图共用同款底栏（P-G 反馈）。 */}
      {queueTab === 'tasks' && loaded && !isEmpty && history.total > HISTORY_PAGE_SIZES[0] && (
        <PaginationBar
          page={history.page}
          total={history.total}
          pageSize={historyPageSize}
          onPage={setHistoryPage}
          onPageSize={(n) => { setHistoryPageSize(n); setHistoryPage(1) }}
        />
      )}
      {queueTab === 'jobs' && jobsHistoryTotal > HISTORY_PAGE_SIZES[0] && (
        <PaginationBar
          page={jobsHistoryPage}
          total={jobsHistoryTotal}
          pageSize={jobsPageSize}
          onPage={setJobsHistoryPage}
          onPageSize={(n) => { setJobsPageSize(n); setJobsHistoryPage(1) }}
        />
      )}

      {/* ADR Addendum 1 §UI：暂停 confirm modal — 告知用户语义后才调 api。 */}
      {pauseConfirmTaskId !== null && (
        <PauseConfirmModal
          onCancel={() => setPauseConfirmTaskId(null)}
          onConfirm={() => void confirmPause()}
        />
      )}

      {/* ADR §4.3 暂停过程 modal — pausingTaskId 非 null 时全程锁屏。 */}
      {pausingTaskId !== null && (
        <PauseProgressModal
          taskId={pausingTaskId}
          taskName={[...live, ...history.items].find((t) => t.id === pausingTaskId)?.name}
          onClose={() => setPausingTaskId(null)}
        />
      )}

      {/* ADR §4.4 挂起 confirmation modal */}
      {holdModalOpen && (
        <HoldQueueModal
          runningTask={runningTask}
          onCancel={() => setHoldModalOpen(false)}
          onConfirm={onHoldConfirm}
        />
      )}
    </StepShell>
  )
}
