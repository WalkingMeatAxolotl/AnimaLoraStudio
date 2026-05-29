/**
 * PR-3 C2 — installGlobalErrorHandlers + reportClientError 单测。
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { reportClientError, setLastApiTraceId } from './report'
import { _resetInstalledForTests, installGlobalErrorHandlers } from './setup'

describe('reportClientError', () => {
  let fetchSpy: ReturnType<typeof vi.fn>

  beforeEach(() => {
    setLastApiTraceId(undefined)
    fetchSpy = vi.fn(() => Promise.resolve(new Response(null, { status: 204 })))
    vi.stubGlobal('fetch', fetchSpy)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('POSTs to /api/client-errors with normalized body', () => {
    reportClientError({
      kind: 'manual',
      message: 'test message',
      stack: 'Error: x\n  at foo:1',
    })
    expect(fetchSpy).toHaveBeenCalledOnce()
    const [url, init] = fetchSpy.mock.calls[0]
    expect(url).toBe('/api/client-errors')
    expect(init?.method).toBe('POST')
    expect(init?.keepalive).toBe(true)
    const body = JSON.parse(String(init?.body ?? '{}'))
    expect(body.kind).toBe('manual')
    expect(body.message).toBe('test message')
    expect(body.stack).toContain('Error: x')
    expect(body.url).toBe(globalThis.location.href)
    expect(body.user_agent).toBeDefined()
    expect(body.client_ts).toBeDefined()
  })

  it('attaches trace_id_last_4xx when set', () => {
    setLastApiTraceId('abc123def456789012345678')
    reportClientError({ kind: 'manual', message: 'm' })
    const body = JSON.parse(String(fetchSpy.mock.calls[0][1]?.body ?? '{}'))
    expect(body.trace_id_last_4xx).toBe('abc123def456789012345678')
  })

  it('silently swallows fetch rejection (no throw)', async () => {
    fetchSpy.mockImplementation(() => Promise.reject(new Error('network down')))
    // 不应 throw
    expect(() => reportClientError({ kind: 'manual', message: 'm' })).not.toThrow()
  })

  it('silently swallows when fetch itself throws synchronously', () => {
    fetchSpy.mockImplementation(() => {
      throw new Error('fetch unavailable')
    })
    expect(() => reportClientError({ kind: 'manual', message: 'm' })).not.toThrow()
  })
})

describe('installGlobalErrorHandlers', () => {
  let fetchSpy: ReturnType<typeof vi.fn>
  let addSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    _resetInstalledForTests()
    fetchSpy = vi.fn(() => Promise.resolve(new Response(null, { status: 204 })))
    vi.stubGlobal('fetch', fetchSpy)
    addSpy = vi.spyOn(globalThis, 'addEventListener')
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    addSpy.mockRestore()
  })

  it('registers window error + unhandledrejection listeners', () => {
    installGlobalErrorHandlers()
    const events = addSpy.mock.calls.map((c) => c[0])
    expect(events).toContain('error')
    expect(events).toContain('unhandledrejection')
  })

  it('is idempotent (no double register)', () => {
    installGlobalErrorHandlers()
    const calls1 = addSpy.mock.calls.length
    installGlobalErrorHandlers()
    expect(addSpy.mock.calls.length).toBe(calls1)
  })

  it('window error handler reports with stack/source/line', () => {
    installGlobalErrorHandlers()
    const errHandler = addSpy.mock.calls.find((c) => c[0] === 'error')?.[1]
    expect(errHandler).toBeTypeOf('function')

    const err = new Error('boom')
    const ev = {
      error: err,
      message: 'boom',
      filename: 'app.js',
      lineno: 42,
      colno: 7,
    } as unknown as ErrorEvent
    ;(errHandler as (e: ErrorEvent) => void)(ev)

    expect(fetchSpy).toHaveBeenCalled()
    const body = JSON.parse(String(fetchSpy.mock.calls[0][1]?.body ?? '{}'))
    expect(body.kind).toBe('window.error')
    expect(body.message).toBe('boom')
    expect(body.source).toBe('app.js')
    expect(body.line).toBe(42)
    expect(body.col).toBe(7)
  })

  it('unhandledrejection handler reports with Error reason', () => {
    installGlobalErrorHandlers()
    const rejHandler = addSpy.mock.calls.find((c) => c[0] === 'unhandledrejection')?.[1]

    const reason = new Error('promise died')
    const ev = { reason } as unknown as PromiseRejectionEvent
    ;(rejHandler as (e: PromiseRejectionEvent) => void)(ev)

    const body = JSON.parse(String(fetchSpy.mock.calls[0][1]?.body ?? '{}'))
    expect(body.kind).toBe('unhandledrejection')
    expect(body.message).toBe('promise died')
    expect(body.stack).toBeDefined()
  })

  it('unhandledrejection handler stringifies non-Error reason', () => {
    installGlobalErrorHandlers()
    const rejHandler = addSpy.mock.calls.find((c) => c[0] === 'unhandledrejection')?.[1]
    ;(rejHandler as (e: PromiseRejectionEvent) => void)({
      reason: { code: 42, detail: 'plain object' },
    } as unknown as PromiseRejectionEvent)
    const body = JSON.parse(String(fetchSpy.mock.calls[0][1]?.body ?? '{}'))
    expect(body.kind).toBe('unhandledrejection')
    expect(body.message).toContain('"code":42')
  })
})
