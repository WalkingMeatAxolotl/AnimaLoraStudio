/**
 * useUploadProgress 单元测试 —— 状态机 + formatter。
 *
 * 进度状态机：
 *   - start(n) → phase='uploading', total=n, loaded=0
 *   - onProgress(half) → phase='uploading', speed/eta 推断合理
 *   - onProgress(full) → phase='processing'（loaded === total）
 *   - finish() → phase='done', speed=0, eta=null
 *   - fail(e) → phase='error', error message
 *   - reset() → INITIAL
 *
 * lengthComputable=false 时 total 归 0、ETA 返回 null。
 */
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import {
  formatBytes,
  formatEta,
  formatSpeed,
  useUploadProgress,
} from './useUploadProgress'

describe('useUploadProgress state machine', () => {
  it('start sets phase=uploading and total', () => {
    const { result } = renderHook(() => useUploadProgress())
    act(() => result.current.start(1000))
    expect(result.current.state.phase).toBe('uploading')
    expect(result.current.state.total).toBe(1000)
    expect(result.current.state.loaded).toBe(0)
  })

  it('switches to processing when loaded === total', () => {
    const { result } = renderHook(() => useUploadProgress())
    act(() => result.current.start(1000))
    act(() => {
      result.current.onProgress({ loaded: 1000, total: 1000, lengthComputable: true })
    })
    expect(result.current.state.phase).toBe('processing')
    expect(result.current.state.speedBps).toBe(0)
    expect(result.current.state.etaSec).toBeNull()
  })

  it('stays in uploading mid-transfer with positive speed', () => {
    vi.useFakeTimers()
    const start = 1000
    vi.setSystemTime(start)
    const perfSpy = vi.spyOn(performance, 'now').mockReturnValue(0)
    try {
      const { result } = renderHook(() => useUploadProgress())
      perfSpy.mockReturnValue(0)
      act(() => result.current.start(1000))
      perfSpy.mockReturnValue(500) // 0.5s later
      act(() => {
        result.current.onProgress({ loaded: 250, total: 1000, lengthComputable: true })
      })
      expect(result.current.state.phase).toBe('uploading')
      expect(result.current.state.speedBps).toBeGreaterThan(0)
      // 速度 ≈ 250 bytes / 0.5s = 500 B/s；剩余 750 → ETA ≈ 1.5s
      expect(result.current.state.etaSec).not.toBeNull()
      expect(result.current.state.etaSec!).toBeGreaterThan(0)
    } finally {
      perfSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it('treats lengthComputable=false as unknown total / no ETA', () => {
    const { result } = renderHook(() => useUploadProgress())
    act(() => result.current.start(1000))
    act(() => {
      result.current.onProgress({ loaded: 500, total: 0, lengthComputable: false })
    })
    expect(result.current.state.total).toBe(0)
    expect(result.current.state.etaSec).toBeNull()
    expect(result.current.state.phase).toBe('uploading')
  })

  it('finish marks loaded=total and phase=done', () => {
    const { result } = renderHook(() => useUploadProgress())
    act(() => result.current.start(1000))
    act(() =>
      result.current.onProgress({ loaded: 600, total: 1000, lengthComputable: true }),
    )
    act(() => result.current.finish())
    expect(result.current.state.phase).toBe('done')
    expect(result.current.state.loaded).toBe(1000)
    expect(result.current.state.speedBps).toBe(0)
  })

  it('fail captures error message', () => {
    const { result } = renderHook(() => useUploadProgress())
    act(() => result.current.start(1000))
    act(() => result.current.fail(new Error('boom')))
    expect(result.current.state.phase).toBe('error')
    expect(result.current.state.error).toBe('boom')
  })

  it('reset returns to idle', () => {
    const { result } = renderHook(() => useUploadProgress())
    act(() => result.current.start(1000))
    act(() => result.current.reset())
    expect(result.current.state.phase).toBe('idle')
    expect(result.current.state.total).toBe(0)
    expect(result.current.state.loaded).toBe(0)
  })
})

describe('formatters', () => {
  it('formatBytes handles 0 / negative / NaN', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(-5)).toBe('0 B')
    expect(formatBytes(Number.NaN)).toBe('0 B')
  })

  it('formatBytes picks unit and precision', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(1024)).toBe('1.0 KB')
    expect(formatBytes(1536)).toBe('1.5 KB')
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB')
    expect(formatBytes(50 * 1024 * 1024)).toBe('50 MB')
    expect(formatBytes(2 * 1024 * 1024 * 1024)).toBe('2.0 GB')
  })

  it('formatSpeed returns — for 0', () => {
    expect(formatSpeed(0)).toBe('—')
    expect(formatSpeed(1024)).toBe('1.0 KB/s')
  })

  it('formatEta handles null / negative / inf', () => {
    expect(formatEta(null)).toBe('—')
    expect(formatEta(-1)).toBe('—')
    expect(formatEta(Number.POSITIVE_INFINITY)).toBe('—')
  })

  it('formatEta picks unit', () => {
    expect(formatEta(0.5)).toBe('<1s')
    expect(formatEta(45)).toBe('45s')
    expect(formatEta(125)).toBe('2m05s')
    expect(formatEta(3725)).toBe('1h02m')
  })
})
