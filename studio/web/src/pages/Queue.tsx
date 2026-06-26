import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import { api, type QueueHoldState, type Task, type TaskStatus } from '../api/client'
import { HoldQueueModal, type HoldDecision } from '../components/HoldQueueModal'
import { PauseConfirmModal } from '../components/PauseConfirmModal'
import { PauseProgressModal } from '../components/PauseProgressModal'
import StepShell from '../components/StepShell'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { useEvaluatingTasks } from '../lib/useEvalProgress'

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
  const [exporting, setExporting] = useState(false)
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
  // ADR 0006 Addendum 1 §UI：确认 modal 先于 PauseProgressModal，告知"暂停可能影响
  // 实验性参数 / 丢失当前轮进度"。pauseConfirmTaskId 非 null 时显示 PauseConfirmModal；
  // 用户点确认 → 调 api + setPausingTaskId 进 PauseProgressModal；用户取消 → 清空。
  const [pauseConfirmTaskId, setPauseConfirmTaskId] = useState<number | null>(null)

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
      // ADR 0006 PR-4 — train_loop_started / auto_epoch_backup_written 都不改
      // task.status 但要让 UI 看到 is_pausable=true（解锁暂停按钮）：is_task_pausable
      // 要 train_loop_started + 首个 epoch backup 落盘二者都满足，真正翻 true 的
      // 是后者，所以两个事件都得 reload；queue_hold_changed 要刷 banner。
      if (
        evt.type === 'task_state_changed' ||
        evt.type === 'train_loop_started' ||
        evt.type === 'auto_epoch_backup_written' ||
        evt.type === 'queue_hold_changed'
      ) {
        if (evt.type === 'queue_hold_changed') {
          void reloadHold()
        }
        if (reloadTimer.current) return
        reloadTimer.current = window.setTimeout(() => {
          reloadTimer.current = null; void reload()
        }, 100)
      } else if (evt.type === 'queue_export_ready' || evt.type === 'queue_export_failed') {
        // <a> 直链发完后端 publish 这对事件,这里清 app-side "导出中..." 状态 +
        // 失败弹 toast。和 train.zip / outputs.zip 一套范式。
        setExporting(false)
        if (evt.type === 'queue_export_failed') {
          const err = typeof evt.error === 'string' ? evt.error : '?'
          setError(t('queue.exportFailed', { error: err }))
        }
      }
    },
    { onOpen: () => { void reload(); void reloadHold() } },
  )

  // 兜底：SSE 事件丢失时 60s 强制清 exporting 状态。
  useEffect(() => {
    if (!exporting) return
    const tid = window.setTimeout(() => setExporting(false), 60_000)
    return () => window.clearTimeout(tid)
  }, [exporting])

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
  // 训练结束后评估可见性：done task 仍在后台跑评估时显示「评估中 done/total」
  const evalMap = useEvaluatingTasks(tasks)
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

  // ADR 0006 + Addendum 1 — "真暂停" 已在 PR-2/3 上线，Addendum 1 翻盘后 Pause = Cancel
  // + 立即释放 GPU，恢复点是最近 epoch 末 auto_epoch_state.pt。
  // 暂停按钮只在 task.is_pausable=true 时出现（train_loop 进入后 + 首个 epoch backup
  // 完成）；点按钮 → 先弹 PauseConfirmModal 告知用户语义 → 确认才调 api → 进
  // PauseProgressModal 锁屏全程引导。
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
  // hold-and-pause 走的快速路径（HoldQueueModal 内已 confirmed，跳过 PauseConfirmModal）
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
          <button
            disabled={busy || exporting || tasks.length === 0}
            onClick={() => {
              if (exporting) return
              setExporting(true)
              // <a download> 直链 —— 浏览器原生接管下载。文件名以后端响应头
              // Content-Disposition.filename 为准（带服务端时间戳）,download
              // 属性是兜底。app-side "导出中..." 由 queue_export_ready/_failed SSE 清。
              const a = document.createElement('a')
              a.href = api.queueExportUrl()
              a.download = `queue_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`
              document.body.appendChild(a)
              a.click()
              document.body.removeChild(a)
            }}
            className="btn btn-ghost btn-sm"
          >{exporting ? t('queue.exporting') : t('common.export')}</button>
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
      <div className="flex flex-col gap-2.5 flex-1 min-h-0 overflow-y-auto">
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
                            to={`/projects/${task.project_id}?version=${task.version_id}`}
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
                        evalMap.get(task.id)?.active ? (
                          <span className="text-accent text-xs flex items-center gap-1" title={t('eval.evaluatingHint')}>
                            <span className="dot dot-running" />
                            {t('eval.evaluatingProgress', {
                              done: evalMap.get(task.id)!.done,
                              total: evalMap.get(task.id)!.total,
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
                        <span className="flex flex-col items-end gap-1">
                          {/* ADR 0006 Addendum 2 — failed/canceled 且恢复点在盘 → 继续训练。
                              paused 走上面的分支，到这里 is_resumable=true 只剩 terminal。 */}
                          {task.is_resumable && (
                            <button
                              onClick={(e) => { e.stopPropagation(); void resumeTask(task) }}
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

      {/* ADR Addendum 1 §UI：暂停 confirm modal — 告知用户语义后才调 api。 */}
      {pauseConfirmTaskId !== null && (
        <PauseConfirmModal
          onCancel={() => setPauseConfirmTaskId(null)}
          onConfirm={() => void confirmPause()}
        />
      )}

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
