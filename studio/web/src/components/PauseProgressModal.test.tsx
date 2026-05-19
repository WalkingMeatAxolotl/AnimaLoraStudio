/**
 * PauseProgressModal 单元测试（ADR 0006 PR-4 §4.3）。
 *
 * 覆盖渲染 + 状态机基础；SSE / fake-timer 驱动 timeout 走另一条路径。
 */
import { fireEvent, render, screen, act } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

let onEventCb: ((evt: { type: string; task_id?: number; status?: string; step?: number }) => void) | null = null

vi.mock('../lib/useEventStream', () => ({
  useEventStream: (cb: (evt: { type: string }) => void) => {
    onEventCb = cb
  },
}))

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (k: string, opts?: Record<string, string | number>) => {
      if (opts && typeof opts === 'object') return `${k}<${JSON.stringify(opts)}>`
      return k
    },
  }),
}))

const toastMock = vi.fn()
vi.mock('./Toast', () => ({
  useToast: () => ({ toast: toastMock }),
}))

const cancelTaskMock = vi.fn().mockResolvedValue({ task_id: 1, canceled: true })
vi.mock('../api/client', () => ({
  api: {
    cancelTask: (id: number) => cancelTaskMock(id),
  },
}))

import { PauseProgressModal } from './PauseProgressModal'

describe('PauseProgressModal', () => {
  beforeEach(() => {
    onEventCb = null
    toastMock.mockClear()
    cancelTaskMock.mockClear()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('mounts in saving phase', () => {
    render(<PauseProgressModal taskId={42} taskName="my_lora" onClose={vi.fn()} />)
    expect(screen.getByTestId('pause-saving')).toBeInTheDocument()
  })

  it('transitions to saved on pause_state event for matching task', () => {
    const onClose = vi.fn()
    render(<PauseProgressModal taskId={42} taskName="my_lora" onClose={onClose} />)
    act(() => {
      onEventCb?.({ type: 'pause_state', task_id: 42, step: 1000 })
    })
    expect(screen.getByTestId('pause-saved')).toBeInTheDocument()
    expect(screen.queryByTestId('pause-saving')).toBeNull()
  })

  it('ignores events for other tasks', () => {
    render(<PauseProgressModal taskId={42} onClose={vi.fn()} />)
    act(() => {
      onEventCb?.({ type: 'pause_state', task_id: 99, step: 500 })
    })
    expect(screen.queryByTestId('pause-saved')).toBeNull()
    expect(screen.getByTestId('pause-saving')).toBeInTheDocument()
  })

  it('saving → timeout after 30s', () => {
    vi.useFakeTimers()
    render(<PauseProgressModal taskId={42} onClose={vi.fn()} />)
    act(() => {
      vi.advanceTimersByTime(31_000)
    })
    expect(screen.getByTestId('pause-timeout')).toBeInTheDocument()
  })

  it('wait-more from timeout → back to saving', () => {
    vi.useFakeTimers()
    render(<PauseProgressModal taskId={42} onClose={vi.fn()} />)
    act(() => {
      vi.advanceTimersByTime(31_000)
    })
    expect(screen.getByTestId('pause-timeout')).toBeInTheDocument()
    fireEvent.click(screen.getByText('queue.pauseProgress.waitMore'))
    expect(screen.getByTestId('pause-saving')).toBeInTheDocument()
  })

  it('clicking ok in saved phase calls onClose', () => {
    const onClose = vi.fn()
    render(<PauseProgressModal taskId={42} onClose={onClose} />)
    act(() => {
      onEventCb?.({ type: 'pause_state', task_id: 42, step: 5 })
    })
    fireEvent.click(screen.getByText('queue.pauseProgress.ok'))
    expect(onClose).toHaveBeenCalled()
  })

  it('terminates: calls cancelTask + closes', async () => {
    vi.useFakeTimers()
    const onClose = vi.fn()
    render(<PauseProgressModal taskId={42} onClose={onClose} />)
    act(() => {
      vi.advanceTimersByTime(31_000)
    })
    fireEvent.click(screen.getByText('queue.pauseProgress.terminate'))
    // 给 promise 一次 microtask 通过
    await act(async () => {
      await Promise.resolve()
    })
    expect(cancelTaskMock).toHaveBeenCalledWith(42)
  })
})
