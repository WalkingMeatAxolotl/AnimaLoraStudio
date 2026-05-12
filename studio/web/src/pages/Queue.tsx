import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, type Task, type TaskStatus } from '../api/client'
import StepShell from '../components/StepShell'
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

async function pickJsonFile(): Promise<unknown | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'; input.accept = '.json,application/json'
    input.onchange = async () => {
      const f = input.files?.[0]
      if (!f) { resolve(null); return }
      try { resolve(JSON.parse(await f.text())) }
      catch { alert('JSON 解析失败'); resolve(null) }
    }
    input.click()
  })
}

type TaskKind = 'train' | 'tag' | 'reg' | 'download' | 'curate' | 'unknown'

const STATUS_LABEL: Record<TaskStatus, string> = {
  pending:   '排队中',
  running:   '运行中',
  done:      '已完成',
  failed:    '失败',
  canceled:  '已取消',
}

const STATUS_TONE: Record<TaskStatus, string> = {
  pending:   'neutral',
  running:   'accent',
  done:      'ok',
  failed:    'err',
  canceled:  'neutral',
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

const KIND_LABEL: Record<TaskKind, string> = {
  train: '训练', tag: '打标', reg: '正则', download: '下载', curate: '筛选', unknown: '任务',
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
  const [tasks, setTasks] = useState<Task[]>([])
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const reloadTimer = useRef<number | null>(null)
  const { toast } = useToast()
  const navigate = useNavigate()
  const reload = useCallback(async () => {
    try { setTasks(await api.listQueue()); setError(null) }
    catch (e) { setError(String(e)) }
    finally { setLoaded(true) }
  }, [])

  useEventStream(
    (evt) => {
      if (evt.type === 'task_state_changed') {
        if (reloadTimer.current) return
        reloadTimer.current = window.setTimeout(() => {
          reloadTimer.current = null; void reload()
        }, 100)
      }
    },
    { onOpen: () => void reload() },
  )

  useEffect(() => { void reload() }, [reload])
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
  const runningTaskId = useMemo(
    () => tasks.find((t) => t.status === 'running')?.id ?? null,
    [tasks],
  )
  // monitor 进度走 useMonitorProgress hook (PR #37 增量协议)：runningTaskId
  // 切换时 hook 自动清状态 + 重拉 /api/state 冷启动；不需要本组件再写清理逻辑。
  const { state: monitor } = useMonitorProgress(runningTaskId)

  const clearDone = async () => {
    const done = tasks.filter((t) => t.status === 'done')
    if (done.length === 0) { toast('没有已完成的任务', 'success'); return }
    if (!confirm(`删除 ${done.length} 个已完成任务？`)) return
    setBusy(true)
    try {
      for (const t of done) await api.deleteTask(t.id)
      toast(`已清理 ${done.length} 个任务`, 'success')
      await reload()
    } catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

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

  const hasRunning = useMemo(() => tasks.some(t => t.status === 'running'), [tasks])

  return (
    <StepShell
      idx={-1}
      title="队列"
      subtitle="同一时刻仅运行一个任务 · 完成后自动启动下一个"
      actions={
        <>
          <button onClick={clearDone} disabled={busy} className="btn btn-ghost btn-sm">清理已完成</button>
          {hasRunning && (
            <button className="btn btn-secondary btn-sm text-warn border-warn">
              暂停队列
            </button>
          )}
          <button
            disabled={busy || tasks.length === 0}
            onClick={async () => {
              try { await downloadJson(`queue_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`, await api.exportQueue()) }
              catch (e) { setError(String(e)) }
            }}
            className="btn btn-ghost btn-sm"
          >导出</button>
          <button
            disabled={busy}
            onClick={async () => {
              const payload = await pickJsonFile()
              if (!payload) return
              setBusy(true)
              try {
                const r = await api.importQueue(payload)
                toast(`已导入 ${r.imported_count} 个任务${Object.keys(r.renamed).length ? `（${Object.keys(r.renamed).length} 个改名）` : ''}`, 'success')
                await reload()
              } catch (e) { setError(String(e)) }
              finally { setBusy(false) }
            }}
            className="btn btn-ghost btn-sm"
          >导入</button>
          <button onClick={() => void reload()} className="btn btn-ghost btn-sm">刷新</button>
        </>
      }
    >
      <div className="flex flex-col gap-2.5">
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
              队列为空
            </div>
            <div className="text-sm text-fg-tertiary">
              从项目训练页入队任务即可
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {sorted.map((t) => {
              const isRunning = t.status === 'running'
              const isTerminal = ['done', 'failed', 'canceled'].includes(t.status)
              const hasProject = !!(t.project_id && t.version_id)
              const kind = inferKind(t)
              const eta = estimateEta(t)
              const tone = STATUS_TONE[t.status]

              return (
                <button
                  key={t.id}
                  onClick={() => navigate(`/queue/${t.id}`)}
                  className={`card card-hover block overflow-hidden text-left p-0 ${isRunning ? 'cursor-pointer border border-accent bg-accent-soft' : 'cursor-default border border-subtle bg-surface'}`}
                >
                  <div
                    className="px-[22px] py-4 grid gap-4 items-center"
                    style={{ gridTemplateColumns: '60px 1fr 110px 1fr 160px' }}
                  >
                    {/* #ID */}
                    <span className={`font-mono text-sm ${isRunning ? 'text-accent font-semibold' : 'text-fg-tertiary font-normal'}`}>
                      #{t.id}
                    </span>

                    {/* 名称 + 种类 */}
                    <div style={{ minWidth: 0 }}>
                      <div className="font-semibold text-fg-primary text-sm overflow-hidden text-ellipsis whitespace-nowrap">
                        {t.name}
                      </div>
                      <div className="font-mono text-xs text-fg-tertiary mt-0.5 flex items-center gap-1.5">
                        <span>{KIND_LABEL[kind]}</span>
                        <span>{t.config_name}</span>
                        {hasProject && (
                          <Link
                            to={`/projects/${t.project_id}/v/${t.version_id}/train`}
                            onClick={(e) => e.stopPropagation()}
                            className="text-accent text-xs no-underline hover:underline shrink-0"
                          >
                            项目
                          </Link>
                        )}
                      </div>
                    </div>

                    {/* 状态 */}
                    <span className={`badge badge-${tone} text-xs text-center`}>
                      {isRunning && <span className="dot dot-running" />}
                      {STATUS_LABEL[t.status]}
                    </span>

                    {/* 进度 / 报错 */}
                    <div className="text-sm text-fg-secondary" style={{ minWidth: 0 }}>
                      {isRunning ? (
                        <div className="flex flex-col gap-0.5">
                          <span className="font-mono text-fg-tertiary text-xs">
                            {(() => {
                              // monitor 在这一行任务上有 step/total → 显示 step；
                              // 否则 fallback 时长（采样阶段或非训练任务）。
                              if (
                                t.id === runningTaskId &&
                                monitor?.step != null &&
                                monitor.total_steps != null &&
                                monitor.total_steps > 0
                              ) {
                                return `step ${monitor.step.toLocaleString()} / ${monitor.total_steps.toLocaleString()}`
                              }
                              return fmtDuration(t.started_at, null)
                            })()}
                          </span>
                          <div className="h-1 bg-overlay rounded-sm overflow-hidden">
                            {(() => {
                              const haveSteps =
                                t.id === runningTaskId &&
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
                              // 没拿到 step（采样 baseline / 非训练 / 还没起来）→ 显示
                              // 不定进度的细动画条，不假装百分比。
                              return <div className="h-full bg-accent/40 rounded-sm animate-pulse" style={{ width: '20%' }} />
                            })()}
                          </div>
                        </div>
                      ) : t.error_msg ? (
                        <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap block text-xs">
                          {t.error_msg}
                        </span>
                      ) : isTerminal ? (
                        <span className="font-mono text-fg-tertiary text-xs">
                          用时 {fmtDuration(t.started_at, t.finished_at)}
                        </span>
                      ) : (
                        <span className="text-fg-tertiary text-xs">—</span>
                      )}
                    </div>

                    {/* ETA / 时间 */}
                    <span className="font-mono text-sm text-fg-tertiary text-right">
                      {isRunning ? (
                        <>
                          {eta && <span className="text-accent">{eta}</span>}
                          {eta && <br />}
                          <span className="text-xs">{fmtAgo(t.started_at!)} 开始</span>
                        </>
                      ) : t.finished_at ? (
                        <>
                          <span>{fmtAgo(t.finished_at)}</span>
                          <br />
                          <span className="text-xs text-fg-tertiary">完成</span>
                        </>
                      ) : (
                        <span>前面 {prevCount(t.id)} 个</span>
                      )}
                    </span>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>
    </StepShell>
  )
}
