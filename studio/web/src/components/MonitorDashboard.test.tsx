import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { MonitorProgress } from '../lib/useMonitorProgress'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import MonitorDashboard from './MonitorDashboard'

vi.mock('../lib/useMonitorProgress', () => ({ useMonitorProgress: vi.fn() }))
vi.mock('./SeriesChart', () => ({
  SeriesChart: ({ data }: { data: unknown[] }) => <div data-testid="series-chart" data-points={data.length} />,
}))

function progress(overrides: Partial<MonitorProgress> = {}): MonitorProgress {
  return {
    state: null,
    status: 'loading',
    streamStatus: 'connecting',
    lastUpdatedAt: null,
    refreshing: false,
    error: null,
    connected: false,
    refetch: vi.fn(async () => {}),
    ...overrides,
  }
}

describe('MonitorDashboard evidence states', () => {
  beforeEach(() => {
    vi.mocked(useMonitorProgress).mockReturnValue(progress())
  })

  it('shows an actionable initial snapshot error', () => {
    const refetch = vi.fn(async () => {})
    vi.mocked(useMonitorProgress).mockReturnValue(progress({
      status: 'error',
      error: 'Error: offline',
      refetch,
    }))

    render(<MonitorDashboard taskId={7} taskStatus="running" />)

    expect(screen.getByText('无法读取监控数据')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '重试' }))
    expect(refetch).toHaveBeenCalledTimes(1)
  })

  it('labels terminal evidence as historical and exposes semantic progress', () => {
    vi.mocked(useMonitorProgress).mockReturnValue(progress({
      state: {
        step: 50,
        total_steps: 100,
        losses: [{ step: 50, loss: 0.4 }],
        lr_history: [],
        samples: [],
      },
      status: 'ready',
      streamStatus: 'live',
      connected: true,
      lastUpdatedAt: new Date('2026-09-18T10:00:00Z').getTime(),
    }))

    render(<MonitorDashboard taskId={7} taskStatus="done" />)

    expect(screen.getByText('历史快照')).toBeInTheDocument()
    const bar = screen.getByRole('progressbar', { name: '训练进度' })
    expect(bar).toHaveAttribute('aria-valuenow', '50')
    expect(bar).toHaveAttribute('aria-valuemax', '100')
  })

  it('keeps a stale snapshot visible while reconnecting', () => {
    const refetch = vi.fn(async () => {})
    vi.mocked(useMonitorProgress).mockReturnValue(progress({
      state: {
        step: 12,
        total_steps: 100,
        losses: [{ step: 12, loss: 0.6 }],
        lr_history: [],
        samples: [],
      },
      status: 'ready',
      streamStatus: 'reconnecting',
      lastUpdatedAt: Date.now(),
      refetch,
    }))

    render(<MonitorDashboard taskId={7} taskStatus="running" />)

    expect(screen.getByText('正在恢复实时连接')).toBeInTheDocument()
    expect(screen.getByText('重连中')).toBeInTheDocument()
    expect(screen.getByText('12')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '重新读取' }))
    expect(refetch).toHaveBeenCalledTimes(1)
  })
})
