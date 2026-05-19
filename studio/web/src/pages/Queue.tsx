import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import { api, type QueueHoldState, type Task, type TaskStatus } from '../api/client'
import { HoldQueueModal, type HoldDecision } from '../components/HoldQueueModal'
import { PauseProgressModal } from '../components/PauseProgressModal'
import StepShell from '../components/StepShell'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'

async function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

async function pickJsonFile(jsonErrorMsg: string): Promise<unknown | null> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input')
    input.type = 'file'; input.accept = '.json,application/json'
    input.onchange = async () => {
      const f = input.files?.[0]
      if (!f) { resolve(null); return }
      try { resolve(JSON.parse(await f.text())) }
      catch { reject(new Error(jsonErrorMsg)) }
    }
    input.click()
  })
}

type TaskKind = 'train' | 'tag' | 'reg' | 'download' | 'curate' | 'unknown'

const STATUS_TONE: Record<TaskStatus, string> = {
  pending:   'neutral',
  running:   'accent',
  done:      'ok',
  failed:    'err',
  canceled:  'neutral',
  paused:    'warn',
}

function inferKind(task: Task): TaskKind {
  const n = task.config_name.toLowerCase()
  if (n.includes('train') || n.includes('lora')) return 'train'
  if (n.includes('tag') || n.includes('caption') || n.includes('wd14')) return 'tag'
  if (n.includes('reg') || n.includes('regular')) return 'reg'
  if (n.includes('download') || n.includes('booru')) return 'download'
  if (n.includes('curate') || n.includes('filter')) return 'curate'
  return 'unknown'
}

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

function fmtDurationShort(ms: number): string {
  if (ms < 60e3) return `${Math.round(ms / 1e3)}s`
  if (ms < 3600e3) return `${Math.round(ms / 60e3)}m`
  return `${(ms / 3600e3).toFixed(1)}h`
}

export default function QueuePage() {
  const { t } = useTranslation()
  const [tasks, setTasks] = useState<Task[]>([])
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const reloadTimer = useRef<number | null>(null)
  const { toast } = useToast()
  const { confirm } = useDialog()
  const navigate = useNavigate()

  const STATUS_LABEL: Record<TaskStatus, string> = {
    pending:   t('status.queued'),
    running:   t('status.running'),
    done:      t('status.done'),
    failed:    t('status.failed'),
    canceled:  t('status.canceled'),
    paused:    t('status.paused'),
  }

  // ADR 0006：队列挂起状态，banner + holdModal 用。
  const [holdState, setHoldState] = useState<QueueHoldState | null>(null)
  const [holdModalOpen, setHoldModalOpen] = useState(false)
  const [pausingTaskId, setPausingTaskId] = useState<number | null>(null)

  const reloadHold = useCallback(async () => {
    try {
      const s = await api.getQueueHold()
      setHoldState(s)
    } catch {
      // 网络错 / 启动期 supervisor 未就绪 → 静默；下一轮 SSE 触发重试。
      setHoldState(null)
    }
  }, [])

  const KIND_LABEL: Record<TaskKind, string> = {
    train: t('nav.train'), tag: t('nav.tag'), reg: t('nav.reg'),
    download: t('nav.download'), curate: t('nav.curate'), unknown: t('monitor.taskLabel'),
  }
  const reload = useCallback(async () => {
    try { setTasks(await api.listQueue()); setError(null) }
    catch (e) { setError(String(e)) }
    finally { setLoaded(true) }
  }, [])

  useEventStream(
    (evt) => {
      // ADR 0006 PR-4 — train_loop_started 不改 task.status 但要让 UI 看到
      // is_pausable=true（解锁暂停按钮）；queue_hold_changed 要刷 banner。
      if (
        evt.type === 'task_state_changed' ||
        evt.type === 'train_loop_started' ||
        evt.type === 'queue_hold_changed'
      ) {
        if (evt.type === 'queue_hold_changed') {
          void reloadHold()
        }
        if (reloadTimer.current) return
        reloadTimer.current = window.setTimeout(() => {
          reloadTimer.current = null; void reload()
        }, 100)
      }
    },
    { onOpen: () => { void reload(); void reloadHold() } },
  )

  useEffect(() => { void reload(); void reloadHold() }, [reload, reloadHold])
  // 2s 时钟 tick：仅触发 re-render 让「23m ago」「elapsed 40m」之类的相对时间
  // 字段更新；不发任何 API。spread tasks 触发组件 re-render，下游 derived 状态
  // 跟着更新。
  useEffect(() => {
    const hasRunning = tasks.some((t) => t.status === 'running')
    if (!hasRunning) return
    const tick = window.setInterval(() => setTasks((ts) => [...ts]), 2000)
    return () => window.clearInterval(tick)
  }, [tasks])

  // 当前 running 任务的 id，给 monitor 进度条 / 状态卡片用。
  const runningTask = useMemo(
    () => tasks.find((t) => t.status === 'running') ?? null,
    [tasks],
  )
  const runningTaskId = runningTask?.id ?? null
  // monitor 进度走 useMonitorProgress hook (PR #37 增量协议)：runningTaskId
  // 切换时 hook 自动清状态 + 重拉 /api/state 冷启动；不需要本组件再写清理逻辑。
  const { state: monitor } = useMonitorProgress(runningTaskId)

  const sorted = useMemo(() => [...tasks].sort((a, b) => b.id - a.id), [tasks])

  const prevCount = useCallback((taskId: number): number => {
    let count = 0
    for (const t of sorted) {
      if (t.id === taskId) break
      if (t.status === 'running' || t.status === 'pending') count++
    }
    return count
  }, [sorted])

  const estimateEta = useCallback((task: Task): string | null => {
    if (task.status !== 'running' || !task.started_at) return null
    const elapsed = (Date.now() / 1000 - task.started_at) * 1000
    return `已运行 ${fmtDurationShort(elapsed)}`
  }, [])

  // 用 runningTask 派生比 tasks.some 再扫一遍便宜（runningTask 已经 memo 过）
  const hasRunning = runningTask !== null

  // ADR 0006 — "真暂停" 已在 PR-2/3 上线（feature_flag off 时仍走原 cancel 语义）。
  // 暂停按钮只在 task.is_pausable=true 时出现（train_loop 进入后），UI 锁屏 modal
  // 全程引导（PauseProgressModal）。
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

  return (
    <StepShell
      idx={-1}
      title={t('queue.title')}
      subtitle={t('queue.description')}
      actions={
        <>
          {/* ADR 0006: 顶部 pause 按钮 — 仅 is_pausable=true 时显示（§8.1）。
              isPausable 来自 server enrich 的 task 字段（supervisor slot.train_loop_started 派生）。 */}
          {runningTask?.is_pausable && (
            <button
              onClick={() => void pauseTask(runningTask)}
              disabled={busy || pausingTaskId !== null}
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
          <button
            disabled={busy || tasks.length === 0}
            onClick={async () => {
              try { await downloadJson(`queue_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`, await api.exportQueue()) }
              catch (e) { setError(String(e)) }
            }}
            className="btn btn-ghost btn-sm"
          >{t('common.export')}</button>
          <button
            disabled={busy}
            onClick={async () => {
              let payload: unknown
              try { payload = await pickJsonFile(t('queue.jsonError')) }
              catch (e) { toast(String(e), 'error'); return }
              if (!payload) return
              setBusy(true)
              try {
                const r = await api.importQueue(payload)
                const renamedCount = Object.keys(r.renamed).length
                toast(t('queue.imported', { n: r.imported_count, renamed: renamedCount ? `（${renamedCount} 个改名）` : '' }), 'success')
                await reload()
              } catch (e) { setError(String(e)) }
              finally { setBusy(false) }
            }}
            className="btn btn-ghost btn-sm"
          >{t('common.import')}</button>
          <button onClick={() => void reload()} className="btn btn-ghost btn-sm">{t('common.refresh')}</button>
        </>
      }
    >
      <div className="flex flex-col gap-2.5">
        {/* ADR §4.1 队列挂起 banner — 仅 held=true 时显示，sticky 顶部。 */}
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

        {!loaded ? (
          <div className="rounded-lg border border-subtle bg-surface overflow-hidden">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className={`py-[18px] px-[22px] grid gap-4 items-center opacity-40 ${i < 2 ? 'border-b border-subtle' : 'border-b-0'}`}
                style={{ gridTemplateColumns: '60px 1fr 110px 1fr 160px' }}
              >
                <div className="h-3.5 rounded bg-overlay" />
                <div className="flex flex-col gap-1">
                  <div className="h-[13px] rounded bg-overlay w-3/5" />
                  <div className="h-2.5 rounded bg-overlay w-2/5" />
                </div>
                <div className="h-5 rounded bg-overlay" />
                <div className="h-2.5 rounded bg-overlay" />
                <div className="h-2.5 rounded bg-overlay" />
              </div>
            ))}
          </div>
        ) : tasks.length === 0 ? (
          <div className="rounded-lg border border-subtle bg-surface py-12 text-center">
            <div className="text-md font-semibold text-fg-secondary mb-1.5">
              {t('queue.empty')}
            </div>
            <div className="text-sm text-fg-tertiary">
              {t('queue.emptyHint')}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {sorted.map((task) => {
              const isRunning = task.status === 'running'
              const isPaused = task.status === 'paused'
              const isTerminal = ['done', 'failed', 'canceled'].includes(task.status)
              const hasProject = !!(task.project_id && task.version_id)
              const isWaitingForRelease = task.status === 'pending' && holdState?.held === true
              const kind = inferKind(task)
              const eta = estimateEta(task)
              const tone = STATUS_TONE[task.status]

              return (
                <button
                  key={task.id}
                  onClick={() => navigate(`/queue/${task.id}`)}
                  className={`card card-hover block overflow-hidden text-left p-0 ${isRunning ? 'cursor-pointer border border-accent bg-accent-soft' : 'cursor-default border border-subtle bg-surface'}`}
                >
                  <div
                    className="px-[22px] py-4 grid gap-4 items-center"
                    style={{ gridTemplateColumns: '60px 1fr 110px 1fr 160px' }}
                  >
                    <span className={`font-mono text-sm ${isRunning ? 'text-accent font-semibold' : 'text-fg-tertiary font-normal'}`}>
                      #{task.id}
                    </span>

                    <div style={{ minWidth: 0 }}>
                      <div className="font-semibold text-fg-primary text-sm overflow-hidden text-ellipsis whitespace-nowrap">
                        {task.name}
                      </div>
                      <div className="font-mono text-xs text-fg-tertiary mt-0.5 flex items-center gap-1.5">
                        <span>{KIND_LABEL[kind]}</span>
                        <span>{task.config_name}</span>
                        {hasProject && (
                          <Link
                            to={`/projects/${task.project_id}/v/${task.version_id}/train`}
                            onClick={(e) => e.stopPropagation()}
                            className="text-accent text-xs no-underline hover:underline shrink-0"
                          >
                            {t('queue.project')}
                          </Link>
                        )}
                      </div>
                    </div>

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
                        <span className="font-mono text-fg-tertiary text-xs">
                          {t('queue.duration', { time: fmtDuration(task.started_at, task.finished_at) })}
                        </span>
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
                              onClick={(e) => { e.stopPropagation(); void resumeTask(task) }}
                              className="btn btn-secondary btn-xs"
                              title={t('queue.resumeHint')}
                              data-testid={`resume-btn-${task.id}`}
                            >
                              {t('queue.resume')}
                            </button>
                            <button
                              onClick={(e) => { e.stopPropagation(); void cancelPaused(task) }}
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
                      ) : task.finished_at ? (
                        <>
                          <span>{fmtAgo(task.finished_at)}</span>
                          <br />
                          <span className="text-xs text-fg-tertiary">{t('status.done')}</span>
                        </>
                      ) : (
                        <span className="flex flex-col items-end gap-0.5">
                          <span>{t('queue.ahead', { n: prevCount(task.id) })}</span>
                          {isWaitingForRelease && (
                            <span className="text-xs text-warn">
                              {t('queue.waitingForRelease')}
                            </span>
                          )}
                        </span>
                      )}
                    </span>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* ADR §4.3 暂停过程 modal — pausingTaskId 非 null 时全程锁屏。
          modal 自己监听 pause_state / task_state_changed 切换 phase。 */}
      {pausingTaskId !== null && (
        <PauseProgressModal
          taskId={pausingTaskId}
          taskName={tasks.find((t) => t.id === pausingTaskId)?.name}
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
