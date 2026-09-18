import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useSearchParams } from 'react-router-dom'

import { api, type Task, type TaskStatus } from '../../api/client'
import Alert from '../../components/Alert'
import Badge, { type BadgeTone } from '../../components/Badge'
import Button, { buttonClassName } from '../../components/Button'
import EmptyState from '../../components/EmptyState'
import { Select } from '../../components/FormControl'
import MonitorDashboard from '../../components/MonitorDashboard'
import StepShell from '../../components/StepShell'
import type { LogSourceStatus } from '../../components/TaskLogDrawer'
import { useEventStream } from '../../lib/useEventStream'
import { useTaskLog } from '../../lib/useTaskLog'

const HISTORY_PAGE_SIZE = 100

function parseTaskId(raw: string | null): number | null {
  if (raw === null) return null
  const value = Number(raw)
  return Number.isInteger(value) && value > 0 ? value : null
}

function isTrainTask(task: Task): boolean {
  // task_type was added after the first queue schema; missing legacy values are train.
  return task.task_type == null || task.task_type === 'train'
}

function hasMonitorEvidence(task: Task): boolean {
  return task.status === 'running' || Boolean(task.monitor_state_path)
}

function taskTimestamp(task: Task): number {
  return task.started_at ?? task.finished_at ?? task.created_at
}

function sortMonitorTasks(tasks: Task[]): Task[] {
  const statusRank: Record<TaskStatus, number> = {
    running: 0,
    paused: 1,
    pending: 2,
    scheduled: 3,
    done: 4,
    failed: 5,
    canceled: 6,
  }
  return [...tasks].sort((a, b) => {
    const rank = statusRank[a.status] - statusRank[b.status]
    return rank === 0 ? taskTimestamp(b) - taskTimestamp(a) : rank
  })
}

function statusTone(status: TaskStatus): BadgeTone {
  switch (status) {
    case 'running': return 'accent'
    case 'done': return 'success'
    case 'failed': return 'danger'
    case 'paused': return 'warning'
    default: return 'neutral'
  }
}

export default function MonitorPage() {
  const { t } = useTranslation()
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedTaskId = parseTaskId(searchParams.get('task'))

  const [taskSegments, setTaskSegments] = useState<{ live: Task[]; history: Task[] }>({
    live: [], history: [],
  })
  const [deepLinkedTask, setDeepLinkedTask] = useState<Task | null>(null)
  const [taskId, setTaskId] = useState<number | null>(requestedTaskId)
  const [listLoading, setListLoading] = useState(true)
  const [listError, setListError] = useState<string | null>(null)
  const [deepLinkLoading, setDeepLinkLoading] = useState(false)
  const [deepLinkError, setDeepLinkError] = useState<'missing' | 'wrongType' | null>(null)

  const tasks = useMemo(() => {
    const byId = new Map<number, Task>()
    for (const task of [...taskSegments.live, ...taskSegments.history]) {
      if (isTrainTask(task) && hasMonitorEvidence(task)) byId.set(task.id, task)
    }
    return sortMonitorTasks([...byId.values()])
  }, [taskSegments])

  const loadTasks = useCallback(async () => {
    setListLoading(true)
    setListError(null)
    const [liveResult, historyResult] = await Promise.allSettled([
      api.listQueueLive(undefined, 'train'),
      api.listQueueHistory({ page: 1, pageSize: HISTORY_PAGE_SIZE, type: 'train' }),
    ])
    setTaskSegments((previous) => ({
      live: liveResult.status === 'fulfilled' ? liveResult.value : previous.live,
      history: historyResult.status === 'fulfilled' ? historyResult.value.items : previous.history,
    }))
    const failures = [liveResult, historyResult]
      .filter((result): result is PromiseRejectedResult => result.status === 'rejected')
    setListError(failures.length > 0 ? failures.map((result) => String(result.reason)).join('\n') : null)
    setListLoading(false)
  }, [])

  useEffect(() => { void loadTasks() }, [loadTasks])

  // Keep task identity/status current when Queue publishes a transition. The monitor
  // payload itself is handled by useMonitorProgress; this refresh is only page context.
  useEventStream(
    useCallback((evt) => {
      if (evt.type === 'task_state_changed') void loadTasks()
    }, [loadTasks]),
  )

  const listedTask = useMemo(
    () => tasks.find((task) => task.id === requestedTaskId) ?? null,
    [requestedTaskId, tasks],
  )

  // A bookmarked train task may be older than the bounded history page. Resolve it
  // directly instead of replacing it with the current running task.
  useEffect(() => {
    if (requestedTaskId == null) {
      setDeepLinkedTask(null)
      setDeepLinkError(null)
      return
    }
    if (listedTask) {
      setDeepLinkedTask(null)
      setDeepLinkError(null)
      setTaskId(requestedTaskId)
      return
    }
    if (listLoading) return

    let active = true
    setDeepLinkLoading(true)
    setDeepLinkError(null)
    void api.getTask(requestedTaskId)
      .then((task) => {
        if (!active) return
        if (!isTrainTask(task)) {
          setDeepLinkedTask(null)
          setDeepLinkError('wrongType')
          setTaskId(null)
          return
        }
        setDeepLinkedTask(task)
        setTaskId(task.id)
      })
      .catch(() => {
        if (!active) return
        setDeepLinkedTask(null)
        setDeepLinkError('missing')
        setTaskId(null)
      })
      .finally(() => {
        if (active) setDeepLinkLoading(false)
      })
    return () => { active = false }
  }, [listLoading, listedTask, requestedTaskId])

  const visibleTasks = useMemo(() => {
    if (!deepLinkedTask || tasks.some((task) => task.id === deepLinkedTask.id)) return tasks
    return sortMonitorTasks([deepLinkedTask, ...tasks])
  }, [deepLinkedTask, tasks])

  // No explicit deep link: prefer the active train, then the newest task that has
  // monitor evidence. Once selected, later list refreshes do not steal the selection.
  useEffect(() => {
    if (requestedTaskId != null || taskId != null || listLoading) return
    const next = visibleTasks.find((task) => task.status === 'running') ?? visibleTasks[0]
    if (!next) return
    setTaskId(next.id)
    const params = new URLSearchParams(searchParams)
    params.set('task', String(next.id))
    setSearchParams(params, { replace: true })
  }, [listLoading, requestedTaskId, searchParams, setSearchParams, taskId, visibleTasks])

  const selectTask = (nextId: number) => {
    setDeepLinkedTask(null)
    setDeepLinkError(null)
    setTaskId(nextId)
    const params = new URLSearchParams(searchParams)
    params.set('task', String(nextId))
    setSearchParams(params, { replace: true })
  }

  const selectedTask = visibleTasks.find((task) => task.id === taskId) ?? null
  const log = useTaskLog(selectedTask?.id ?? null)
  const logSource = selectedTask
    ? {
        key: `monitor-task-${selectedTask.id}`,
        label: t('monitor.logLabel', { id: selectedTask.id }),
        status: selectedTask.status as LogSourceStatus,
        lines: log.lines,
        startedAt: selectedTask.started_at,
        finishedAt: selectedTask.finished_at,
        downloadUrl: log.downloadUrl,
        hasMoreBefore: log.hasMoreBefore,
        loadingAll: log.loadingAll,
        onLoadAll: log.loadAll,
      }
    : null

  const selectedStatus = selectedTask?.status
  const contextBar = (
    <div className="border-b border-subtle bg-canvas px-page py-related">
      <div className="flex min-w-0 flex-wrap items-center gap-related">
        <label htmlFor="monitor-task-select" className="text-sm font-medium text-fg-secondary">
          {t('monitor.taskLabel')}
        </label>
        <Select
          id="monitor-task-select"
          controlSize="sm"
          surface="surface"
          value={selectedTask?.id ?? ''}
          onChange={(event) => selectTask(Number(event.target.value))}
          disabled={visibleTasks.length === 0 || deepLinkLoading}
          aria-label={t('monitor.taskSelectAria')}
          className="min-w-0 max-w-full sm:min-w-[20rem]"
        >
          {visibleTasks.length === 0 && <option value="">{t('monitor.noTaskOption')}</option>}
          {visibleTasks.map((task) => (
            <option key={task.id} value={task.id}>
              #{task.id} · {task.name} · {t(`monitor.taskStatus.${task.status}`)}
            </option>
          ))}
        </Select>
        {selectedStatus && (
          <Badge tone={statusTone(selectedStatus)} active={selectedStatus === 'running'}>
            {t(`monitor.taskStatus.${selectedStatus}`)}
          </Badge>
        )}
        {listLoading && tasks.length > 0 && (
          <span className="text-xs text-fg-tertiary" role="status">{t('common.loading')}</span>
        )}
      </div>
    </div>
  )

  return (
    <StepShell
      title={t('monitor.title')}
      subtitle={t('monitor.subtitle')}
      belowHeader={contextBar}
      actions={selectedTask && (
        <Link
          to={`/queue/${selectedTask.id}`}
          className={buttonClassName({ variant: 'secondary', size: 'sm', className: 'no-underline' })}
        >
          {t('monitor.viewTaskDetails')}
        </Link>
      )}
      logSources={[logSource]}
    >
      <div className="flex min-h-0 flex-1 flex-col gap-related">
        {listError && (
          <Alert
            tone={tasks.length > 0 ? 'warning' : 'danger'}
            size="sm"
            title={t('monitor.taskListErrorTitle')}
            action={(
              <Button size="sm" onClick={() => void loadTasks()}>
                {t('common.retry')}
              </Button>
            )}
          >
            <span title={listError}>
              {t(tasks.length > 0 ? 'monitor.taskListStale' : 'monitor.taskListError')}
            </span>
          </Alert>
        )}

        {deepLinkError && (
          <Alert tone="danger" size="sm" title={t('monitor.deepLinkErrorTitle')}>
            {t(deepLinkError === 'wrongType' ? 'monitor.deepLinkWrongType' : 'monitor.deepLinkMissing', {
              id: requestedTaskId,
            })}
          </Alert>
        )}

        {deepLinkLoading && !selectedTask && (
          <div className="grid flex-1 place-items-center text-sm text-fg-tertiary" role="status">
            {t('monitor.loadingTask')}
          </div>
        )}

        {!deepLinkLoading && !selectedTask && listLoading && (
          <div className="grid flex-1 place-items-center text-sm text-fg-tertiary" role="status">
            {t('monitor.loadingTasks')}
          </div>
        )}

        {!deepLinkLoading && !selectedTask && !listLoading && !deepLinkError && !listError && (
          <EmptyState
            className="m-auto max-w-xl"
            title={t('monitor.noTasksTitle')}
            description={t('monitor.noTasksDescription')}
            action={(
              <Link
                to="/queue"
                className={buttonClassName({ variant: 'secondary', size: 'sm', className: 'no-underline' })}
              >
                {t('monitor.openQueue')}
              </Link>
            )}
          />
        )}

        {selectedTask && (
          <div className="min-h-0 flex-1 overflow-hidden">
            <MonitorDashboard taskId={selectedTask.id} taskStatus={selectedTask.status} />
          </div>
        )}
      </div>
    </StepShell>
  )
}

export {
  hasMonitorEvidence as _hasMonitorEvidenceForTest,
  isTrainTask as _isTrainTaskForTest,
  sortMonitorTasks as _sortMonitorTasksForTest,
}
