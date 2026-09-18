import { describe, expect, it } from 'vitest'
import { getMonitorElapsedSeconds } from './monitorElapsed'

const NOW = 2_000

describe('getMonitorElapsedSeconds', () => {
  it('uses the authoritative task start while a task is running', () => {
    expect(getMonitorElapsedSeconds(1_200, {
      status: 'running',
      started_at: 1_000,
      finished_at: null,
      paused_at: null,
    }, NOW)).toBe(1_000)
  })

  it.each([
    ['done', 1_600],
    ['failed', 1_550],
    ['canceled', 1_500],
  ] as const)('freezes a %s task at finished_at', (status, finishedAt) => {
    expect(getMonitorElapsedSeconds(1_200, {
      status,
      started_at: 1_000,
      finished_at: finishedAt,
      paused_at: null,
    }, NOW)).toBe(finishedAt - 1_000)
  })

  it('freezes a paused task at paused_at', () => {
    expect(getMonitorElapsedSeconds(1_200, {
      status: 'paused',
      started_at: 1_000,
      finished_at: null,
      paused_at: 1_450,
    }, NOW)).toBe(450)
  })

  it('does not invent a live duration for a terminal row missing its end time', () => {
    expect(getMonitorElapsedSeconds(1_200, {
      status: 'failed',
      started_at: 1_000,
      finished_at: null,
      paused_at: null,
    }, NOW)).toBeNull()
  })

  it('falls back to monitor start time for a legacy running task', () => {
    expect(getMonitorElapsedSeconds(1_200, {
      status: 'running',
      started_at: null,
      finished_at: null,
      paused_at: null,
    }, NOW)).toBe(800)
  })

  it('shows no elapsed time before a task starts', () => {
    expect(getMonitorElapsedSeconds(undefined, {
      status: 'pending',
      started_at: null,
      finished_at: null,
      paused_at: null,
    }, NOW)).toBeNull()
  })
})
