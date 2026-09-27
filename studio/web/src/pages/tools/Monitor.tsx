import { useEffect, useMemo, useState } from 'react'
import { api, type HealthResponse, type Task } from '../../api/client'
import MonitorDashboard from '../../components/MonitorDashboard'
import { useEventStream } from '../../lib/useEventStream'

export default function MonitorPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])
  const [selectedTaskDetail, setSelectedTaskDetail] = useState<Task | null>(null)
  // `?task=N` 深链：直接把监控页锁定到指定 task（书签 / 外部链接用）。
  const initialTaskId = useMemo<number | null>(() => {
    if (typeof window === 'undefined') return null
    const raw = new URLSearchParams(window.location.search).get('task')
    const n = raw === null ? NaN : Number(raw)
    return Number.isFinite(n) && n > 0 ? n : null
  }, [])
  const [taskId, setTaskId] = useState<number | null>(initialTaskId)

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e)))
    api.listQueue().then(setTasks).catch(() => setTasks([]))
  }, [])

  useEffect(() => {
    if (taskId === null || tasks.some((task) => task.id === taskId)) return
    let active = true
    void api.getTask(taskId)
      .then((task) => { if (active) setSelectedTaskDetail(task) })
      .catch(() => {})
    return () => { active = false }
  }, [taskId, tasks])

  useEventStream((evt) => {
    if (evt.type !== 'task_state_changed' || evt.task_id !== taskId || taskId === null) return
    void api.getTask(taskId).then((updated) => {
      setSelectedTaskDetail(updated)
      setTasks((current) => current.map((task) => task.id === updated.id ? updated : task))
    }).catch(() => {})
  })

  const defaultTaskId = useMemo<number | null>(() => {
    const running = tasks.find((t) => t.status === 'running')
    if (running) return running.id
    const ended = [...tasks]
      .filter((t) => t.finished_at)
      .sort((a, b) => (b.finished_at ?? 0) - (a.finished_at ?? 0))[0]
    return ended?.id ?? null
  }, [tasks])

  useEffect(() => {
    if (taskId === null && defaultTaskId !== null) setTaskId(defaultTaskId)
  }, [defaultTaskId, taskId])

  const ok = !error && health?.status === 'ok'
  const selectedTask = tasks.find((t) => t.id === taskId)
    ?? (selectedTaskDetail?.id === taskId ? selectedTaskDetail : undefined)

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      {/* 顶部状态栏 */}
      <section className="card card-compact card-pad-sm mb-field flex shrink-0 flex-wrap items-center gap-field text-xs">
        {/* 健康指示 */}
        <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${ok ? 'bg-ok' : 'bg-err'}`} />
        <span className={`font-semibold font-mono ${ok ? 'text-ok' : 'text-err'}`}>
          {error ? 'offline' : health?.status ?? '...'}
        </span>
        {health && (
          <span className="font-mono text-fg-secondary">
            v{health.version}
          </span>
        )}

        <span className="text-fg-tertiary">|</span>

        {/* 任务选择 */}
        <span className="text-fg-secondary">任务</span>
        <select
          value={taskId ?? ''}
          onChange={(e) => setTaskId(e.target.value === '' ? null : Number(e.target.value))}
          className="form-control form-control-sm form-control-sunken w-auto text-xs"
        >
          <option value="">（最新 running，没有则显示空）</option>
          {tasks.map((t) => (
            <option key={t.id} value={t.id}>
              #{t.id} · {t.name} · {t.status}
            </option>
          ))}
        </select>

        {selectedTask && (
          <>
            <span className="text-fg-tertiary">|</span>
            <span className={statusBadge(selectedTask.status)}>
              {statusLabel(selectedTask.status)}
            </span>
          </>
        )}

        <span className="flex-1" />
      </section>

      {/* 监控主体 */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {taskId !== null ? (
          <MonitorDashboard taskId={taskId} task={selectedTask} />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-related text-sm text-fg-secondary">
            <span className="text-xl">📊</span>
            <span>暂无训练任务</span>
            <span className="text-xs">启动训练后将自动显示监控数据</span>
          </div>
        )}
      </div>
    </div>
  )
}

function statusBadge(status: string): string {
  switch (status) {
    case 'running': return 'badge badge-accent'
    case 'pending': return 'badge badge-neutral'
    case 'scheduled': return 'badge badge-neutral'
    case 'done': return 'badge badge-ok'
    case 'failed': return 'badge badge-err'
    case 'canceled': return 'badge badge-neutral'
    default: return 'badge badge-neutral'
  }
}

function statusLabel(status: string): string {
  switch (status) {
    case 'running': return '运行中'
    case 'pending': return '排队中'
    case 'scheduled': return '等待入队'
    case 'done': return '已完成'
    case 'failed': return '失败'
    case 'canceled': return '已取消'
    default: return status
  }
}
