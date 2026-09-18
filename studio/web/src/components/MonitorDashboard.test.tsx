import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import i18n from '../i18n'

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
  let scrollIntoView: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.mocked(useMonitorProgress).mockReturnValue(progress())
    scrollIntoView = vi.fn()
    Object.defineProperty(Element.prototype, 'scrollIntoView', {
      value: scrollIntoView,
      configurable: true,
    })
  })

  afterEach(async () => {
    vi.unstubAllGlobals()
    await act(async () => { await i18n.changeLanguage('zh') })
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
        start_time: Date.now() / 1000 - 3600,
        speed: 2,
      },
      status: 'ready',
      streamStatus: 'live',
      connected: true,
      lastUpdatedAt: new Date('2026-09-18T10:00:00Z').getTime(),
    }))

    render(<MonitorDashboard taskId={7} taskStatus="done" />)

    expect(screen.getByText('历史快照')).toBeInTheDocument()
    expect(screen.getAllByText('预计剩余')).toHaveLength(1)
    expect(screen.getByText(/已用时间 1时 00分/)).toBeInTheDocument()
    expect(screen.getByText(/2\.00 it\/s · 每秒迭代次数/)).toBeInTheDocument()
    const bar = screen.getByRole('progressbar', { name: '训练进度' })
    expect(bar).toHaveAttribute('aria-valuenow', '50')
    expect(bar).toHaveAttribute('aria-valuemax', '100')
    expect(screen.getByRole('slider', { name: '损失趋势 · 平滑' })).toBeEnabled()
    expect(screen.getByRole('slider', { name: '学习率趋势 · 平滑' })).toBeDisabled()
    expect(screen.getByRole('slider', { name: '优化器 d 趋势 · 平滑' })).toBeDisabled()
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

  it('supports roving sample selection and opens the preview from a real button', () => {
    vi.mocked(useMonitorProgress).mockReturnValue(progress({
      state: {
        step: 30,
        total_steps: 100,
        losses: [],
        lr_history: [],
        samples: [
          { path: 'step_10.png', step: 10 },
          { path: 'step_20.png', step: 20 },
          { path: 'step_30.png', step: 30 },
        ],
      },
      status: 'ready',
      streamStatus: 'live',
    }))

    render(<MonitorDashboard taskId={7} taskStatus="running" />)

    const options = screen.getAllByRole('option')
    expect(options).toHaveLength(3)
    expect(options[2]).toHaveAttribute('aria-selected', 'true')
    expect(options[2]).toHaveAttribute('tabindex', '0')
    expect(options[0]).toHaveAttribute('tabindex', '-1')

    options[2].focus()
    fireEvent.keyDown(options[2], { key: 'ArrowLeft' })
    expect(options[1]).toHaveFocus()
    expect(options[1]).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(options[1], { key: 'Home' })
    expect(options[0]).toHaveFocus()
    fireEvent.keyDown(options[0], { key: 'End' })
    expect(options[2]).toHaveFocus()

    const previewButton = screen.getByRole('button', { name: /放大查看训练样图 3 \/ 3/ })
    previewButton.focus()
    fireEvent.click(previewButton)
    expect(screen.getByRole('dialog', { name: '图片预览' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '关闭' }))
    expect(previewButton).toHaveFocus()
  })

  it('uses non-animated sample scrolling when reduced motion is requested', () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: true }))
    vi.mocked(useMonitorProgress).mockReturnValue(progress({
      state: {
        step: 10,
        total_steps: 100,
        losses: [],
        lr_history: [],
        samples: [{ path: 'step_10.png', step: 10 }],
      },
      status: 'ready',
      streamStatus: 'live',
    }))

    render(<MonitorDashboard taskId={7} taskStatus="running" />)

    expect(scrollIntoView).toHaveBeenCalledWith(expect.objectContaining({ behavior: 'auto' }))
  })

  it('localizes the flat expert metrics and chart controls in English', async () => {
    await i18n.changeLanguage('en')
    vi.mocked(useMonitorProgress).mockReturnValue(progress({
      state: {
        step: 5,
        total_steps: 20,
        losses: [{ step: 5, loss: 0.5 }],
        lr_history: [{ step: 5, lr: 0.0001 }],
        optimizer_metrics_history: [{ step: 5, d: 1.2 }],
        samples: [],
      },
      status: 'ready',
      streamStatus: 'live',
    }))

    render(<MonitorDashboard taskId={7} taskStatus="running" />)

    expect(screen.getByText('Current step')).toBeInTheDocument()
    expect(screen.getByText('Recent loss')).toBeInTheDocument()
    expect(screen.getByText('Overall average loss')).toBeInTheDocument()
    expect(screen.getByText('Learning rate')).toBeInTheDocument()
    expect(screen.getByText('VRAM')).toBeInTheDocument()
    expect(screen.getByText('Training samples')).toBeInTheDocument()
    expect(screen.getByText('Loss trend')).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'Loss trend · Smoothing' })).toBeInTheDocument()
  })
})
