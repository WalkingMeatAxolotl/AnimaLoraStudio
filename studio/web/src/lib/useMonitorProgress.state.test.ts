import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { api, type MonitorState } from '../api/client'

const streamHarness = vi.hoisted(() => ({
  onEvent: null as null | ((event: Record<string, unknown>) => void),
  onOpen: null as null | (() => void),
  onError: null as null | (() => void),
}))

vi.mock('./useEventStream', () => ({
  isEventStreamOpen: () => false,
  useEventStream: (
    onEvent: (event: Record<string, unknown>) => void,
    options?: { onOpen?: () => void; onError?: () => void },
  ) => {
    streamHarness.onEvent = onEvent
    streamHarness.onOpen = options?.onOpen ?? null
    streamHarness.onError = options?.onError ?? null
  },
}))

import { useMonitorProgress } from './useMonitorProgress'

const SNAPSHOT: MonitorState = {
  task_id: 7,
  step: 12,
  total_steps: 100,
  losses: [{ step: 12, loss: 0.42 }],
  lr_history: [],
  samples: [],
}

describe('useMonitorProgress evidence status', () => {
  beforeEach(() => {
    streamHarness.onEvent = null
    streamHarness.onOpen = null
    streamHarness.onError = null
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('does not report an HTTP snapshot as a live SSE connection', async () => {
    const getState = vi.spyOn(api, 'getMonitorState').mockResolvedValue(SNAPSHOT)
    const { result } = renderHook(() => useMonitorProgress(7))

    await waitFor(() => expect(result.current.status).toBe('ready'))
    expect(result.current.state?.step).toBe(12)
    expect(result.current.streamStatus).toBe('connecting')
    expect(result.current.connected).toBe(false)
    expect(result.current.lastUpdatedAt).toEqual(expect.any(Number))

    await act(async () => { streamHarness.onOpen?.() })
    await waitFor(() => expect(getState).toHaveBeenCalledTimes(2))
    expect(result.current.streamStatus).toBe('live')
    expect(result.current.connected).toBe(true)
  })

  it('retains a loaded snapshot and exposes a later refresh failure', async () => {
    const getState = vi.spyOn(api, 'getMonitorState')
      .mockResolvedValueOnce(SNAPSHOT)
      .mockRejectedValueOnce(new Error('refresh failed'))
    const { result } = renderHook(() => useMonitorProgress(7))

    await waitFor(() => expect(result.current.status).toBe('ready'))
    await act(async () => { await result.current.refetch() })

    expect(getState).toHaveBeenCalledTimes(2)
    expect(result.current.status).toBe('ready')
    expect(result.current.state?.step).toBe(12)
    expect(result.current.error).toContain('refresh failed')
  })

  it('distinguishes a missing monitor snapshot from a transport failure', async () => {
    const missing = Object.assign(new Error('not found'), { status: 404 })
    vi.spyOn(api, 'getMonitorState').mockRejectedValue(missing)
    const { result } = renderHook(() => useMonitorProgress(7))

    await waitFor(() => expect(result.current.status).toBe('unavailable'))
    expect(result.current.state).toBeNull()
    expect(result.current.error).toContain('not found')
  })

  it('marks the live stream reconnecting without discarding monitor evidence', async () => {
    vi.spyOn(api, 'getMonitorState').mockResolvedValue(SNAPSHOT)
    const { result } = renderHook(() => useMonitorProgress(7))

    await waitFor(() => expect(result.current.status).toBe('ready'))
    act(() => { streamHarness.onOpen?.() })
    await waitFor(() => expect(result.current.streamStatus).toBe('live'))
    act(() => { streamHarness.onError?.() })

    expect(result.current.streamStatus).toBe('reconnecting')
    expect(result.current.connected).toBe(false)
    expect(result.current.state?.step).toBe(12)
  })
})
