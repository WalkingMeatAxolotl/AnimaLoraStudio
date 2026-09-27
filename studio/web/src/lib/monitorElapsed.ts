import type { Task } from '../api/client'

export type MonitorTaskTiming = Pick<
  Task,
  'status' | 'started_at' | 'finished_at' | 'paused_at'
>

/**
 * Monitor elapsed time follows the authoritative task lifecycle.
 *
 * `monitorStartTime` remains a fallback for legacy state snapshots whose task
 * row has no started_at. Terminal and paused tasks never fall back to `now`:
 * doing so would make historical elapsed time keep increasing on every render.
 */
export function getMonitorElapsedSeconds(
  monitorStartTime: number | undefined,
  task: MonitorTaskTiming | undefined,
  nowSeconds = Date.now() / 1000,
): number | null {
  const startedAt = task?.started_at ?? monitorStartTime
  if (typeof startedAt !== 'number' || !Number.isFinite(startedAt) || startedAt <= 0) return null

  let endedAt: number | null
  switch (task?.status) {
    case 'done':
    case 'failed':
    case 'canceled':
      endedAt = task.finished_at
      break
    case 'paused':
      endedAt = task.paused_at ?? task.finished_at
      break
    case 'pending':
    case 'scheduled':
      return null
    case 'running':
    case undefined:
      endedAt = nowSeconds
      break
    default:
      return null
  }

  if (typeof endedAt !== 'number' || !Number.isFinite(endedAt) || endedAt < startedAt) return null
  return endedAt - startedAt
}
