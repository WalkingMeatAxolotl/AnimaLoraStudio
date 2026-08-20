import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { LogPage } from '../api/client'

// 可控的事件流：记录 handler / onOpen 让测试手动派发
const bus: { handler: ((e: Record<string, unknown>) => void) | null; onOpen: (() => void) | null } = {
  handler: null, onOpen: null,
}
vi.mock('./useEventStream', () => ({
  useEventStream: (h: (e: Record<string, unknown>) => void, opts?: { onOpen?: () => void }) => {
    bus.handler = h
    bus.onOpen = opts?.onOpen ?? null
  },
}))

import { api } from '../api/client'
import { useTaskLog } from './useTaskLog'

function page(lines: [number, string][], extra: Partial<LogPage> = {}): LogPage {
  const last = lines[lines.length - 1]
  const end = last ? last[0] + last[1].length + 1 : (extra.start_offset ?? 0)
  return {
    task_id: 7,
    lines: lines.map(([offset, text]) => ({ offset, text })),
    start_offset: lines[0]?.[0] ?? 0,
    end_offset: end,
    size: end,
    has_more_before: false,
    ...extra,
  }
}

const emit = (e: Record<string, unknown>) => act(() => { bus.handler?.(e) })

beforeEach(() => {
  bus.handler = null
  bus.onOpen = null
  vi.restoreAllMocks()
})

describe('useTaskLog', () => {
  it('冷启动拉尾部；SSE 按 end_offset 去重追加；冷启动前到达的事件先缓冲', async () => {
    const getLog = vi.spyOn(api, 'getLog')
    let resolveTail!: (p: LogPage) => void
    getLog.mockImplementationOnce(() => new Promise<LogPage>((r) => { resolveTail = r }))
    const { result } = renderHook(() => useTaskLog(7, { tail: 10 }))
    expect(result.current.status).toBe('loading')

    // 冷启动未完成时到达：缓冲
    emit({ type: 'task_log_appended', task_id: 7, text: 'c', seq: 3, end_offset: 6 })   // 冷启动已含（≤ end）
    emit({ type: 'task_log_appended', task_id: 7, text: 'd', seq: 4, end_offset: 8 })   // 新的
    await act(async () => { resolveTail(page([[0, 'a'], [2, 'b'], [4, 'c']], { has_more_before: true })) })

    await waitFor(() => expect(result.current.status).toBe('ready'))
    expect(result.current.lines).toEqual(['a', 'b', 'c', 'd'])
    expect(result.current.hasMoreBefore).toBe(true)
    // 重复 / 过期事件忽略；其它 task 的忽略
    emit({ type: 'task_log_appended', task_id: 7, text: 'd', seq: 4, end_offset: 8 })
    emit({ type: 'task_log_appended', task_id: 8, text: 'zzz', seq: 9, end_offset: 99 })
    emit({ type: 'job_log_appended', job_id: 7, text: 'e', seq: 5, end_offset: 10 })   // job 同 id 空间
    expect(result.current.lines).toEqual(['a', 'b', 'c', 'd', 'e'])
  })

  it('重连 onOpen 与状态变化时按游标 after 补拉，不重复已含行', async () => {
    const getLog = vi.spyOn(api, 'getLog')
    getLog.mockResolvedValueOnce(page([[0, 'a'], [2, 'b']]))  // tail → cursor 4
    const { result } = renderHook(() => useTaskLog(7))
    await waitFor(() => expect(result.current.status).toBe('ready'))

    getLog.mockResolvedValueOnce(page([[4, 'c'], [6, 'd']], { start_offset: 4 }))
    await act(async () => { bus.onOpen?.() })
    await waitFor(() => expect(result.current.lines).toEqual(['a', 'b', 'c', 'd']))
    expect(getLog).toHaveBeenLastCalledWith(7, { after: 4, limit: 500 })

    getLog.mockResolvedValueOnce(page([[8, 'e']], { start_offset: 8 }))
    emit({ type: 'task_state_changed', task_id: 7, status: 'done' })
    await waitFor(() => expect(result.current.lines).toEqual(['a', 'b', 'c', 'd', 'e']))
    expect(getLog).toHaveBeenLastCalledWith(7, { after: 8, limit: 500 })
  })

  it('加载更早：before=首行 offset，前插并更新 hasMoreBefore', async () => {
    const getLog = vi.spyOn(api, 'getLog')
    getLog.mockResolvedValueOnce(page([[20, 'k'], [22, 'l']], { has_more_before: true }))
    const { result } = renderHook(() => useTaskLog(7, { tail: 2 }))
    await waitFor(() => expect(result.current.status).toBe('ready'))

    getLog.mockResolvedValueOnce(page([[16, 'i'], [18, 'j']], { has_more_before: false }))
    act(() => result.current.loadEarlier())
    await waitFor(() => expect(result.current.lines).toEqual(['i', 'j', 'k', 'l']))
    expect(getLog).toHaveBeenLastCalledWith(7, { before: 20, limit: 2 })
    expect(result.current.hasMoreBefore).toBe(false)
    // 没有更早时 no-op
    const calls = getLog.mock.calls.length
    act(() => result.current.loadEarlier())
    expect(getLog.mock.calls.length).toBe(calls)
  })

  it('event_malformed 合成完整行契约的 WARNING 行；超出 maxLines 从头裁并标记可回翻', async () => {
    const getLog = vi.spyOn(api, 'getLog')
    getLog.mockResolvedValueOnce(page([[0, 'a'], [2, 'b']]))
    const { result } = renderHook(() => useTaskLog(7, { maxLines: 3 }))
    await waitFor(() => expect(result.current.status).toBe('ready'))
    emit({ type: 'event_malformed', task_id: 7, raw_preview: '__EVENT__:x:{' })
    // 行契约（后端 LOG_LINE_RE 同构）：ts + 级别 + logger 名，不再是裸 `WARNING: `
    expect(result.current.lines[2]).toMatch(
      /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} WARNING web\.logview: /,
    )
    expect(result.current.lines[2]).toContain('__EVENT__:x:{')
    emit({ type: 'task_log_appended', task_id: 7, text: 'c', seq: 1, end_offset: 6 })
    expect(result.current.lines).toHaveLength(3)
    expect(result.current.lines[0]).toBe('b')
    expect(result.current.hasMoreBefore).toBe(true)
  })

  it('id 为空 → idle 且不请求；切换 id 清空重拉', async () => {
    const getLog = vi.spyOn(api, 'getLog')
    const { result, rerender } = renderHook(({ id }: { id: number | null }) => useTaskLog(id), {
      initialProps: { id: null as number | null },
    })
    expect(result.current.status).toBe('idle')
    expect(getLog).not.toHaveBeenCalled()
    getLog.mockResolvedValueOnce(page([[0, 'x']]))
    rerender({ id: 9 })
    await waitFor(() => expect(result.current.lines).toEqual(['x']))
    expect(result.current.downloadUrl).toBe('/api/logs/9/raw')
  })
})
