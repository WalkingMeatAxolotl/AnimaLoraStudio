import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { isEventStreamOpen, useEventStream } from './useEventStream'

class FakeEventSource {
  static instances: FakeEventSource[] = []
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 2

  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.CONNECTING

  constructor(public url: string) {
    FakeEventSource.instances.push(this)
  }

  close(): void {
    this.readyState = FakeEventSource.CLOSED
  }
}

describe('useEventStream shared connection status', () => {
  beforeEach(() => {
    FakeEventSource.instances = []
    vi.stubGlobal('EventSource', FakeEventSource)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('reports open and reconnecting transitions to a subscriber', () => {
    const onOpen = vi.fn()
    const onError = vi.fn()
    const { unmount } = renderHook(() => useEventStream(vi.fn(), { onOpen, onError }))
    const source = FakeEventSource.instances[0]

    expect(source.url).toBe('/api/events')
    expect(isEventStreamOpen()).toBe(false)

    source.readyState = FakeEventSource.OPEN
    act(() => { source.onopen?.() })
    expect(onOpen).toHaveBeenCalledTimes(1)
    expect(isEventStreamOpen()).toBe(true)

    source.readyState = FakeEventSource.CONNECTING
    act(() => { source.onerror?.() })
    expect(onError).toHaveBeenCalledTimes(1)
    expect(isEventStreamOpen()).toBe(false)

    unmount()
    expect(source.readyState).toBe(FakeEventSource.CLOSED)
  })
})
