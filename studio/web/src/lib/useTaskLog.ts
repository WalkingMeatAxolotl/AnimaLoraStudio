/** 单个任务 / 作业 run.log 的数据源（docs/design/logging-target-state.md §3.3/§3.4）。
 *
 * - 冷启动只拉尾部 `tail` 行（`GET /api/logs/{id}?tail=`），顶部「加载更早」按
 *   `before=<首行 offset>` 往前翻
 * - 增量走 SSE `task_log_appended` / `job_log_appended`，用 `end_offset` 去重
 *   （≤ 当前游标的事件是冷启动已含的或重复的）
 * - 断线重连（`useEventStream` onOpen）与任务状态变化时用 `after=<游标>` 补拉，
 *   断线期间丢的行不会丢
 * - `event_malformed` 合成一条 WARNING 行（裸前缀，LogView 解析着色）
 * - 客户端最多保留 `maxLines` 行：超出从头丢，丢掉的部分可再「加载更早」拿回
 *
 * 返回 `lines: string[]`（原文）；解析/着色在 LogView。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { api } from '../api/client'
import i18n from '../i18n'
import { useEventStream, type StudioEvent } from './useEventStream'

export type TaskLogStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface TaskLogState {
  lines: string[]
  status: TaskLogStatus
  error: string | null
  hasMoreBefore: boolean
  loadingEarlier: boolean
  /** 往前再拉一页；没有更早或正在拉时 no-op */
  loadEarlier: () => void
  /** 重新从尾部拉（出错重试用） */
  refresh: () => void
  /** 原始文件下载地址（id 为空时 null） */
  downloadUrl: string | null
}

interface Entry { offset: number; text: string }

const DEFAULT_TAIL = 500
const DEFAULT_MAX = 5000

function _matches(evt: StudioEvent, id: number): boolean {
  if (evt.type === 'task_log_appended' || evt.type === 'event_malformed') {
    if (evt.task_id === id) return true
  }
  if (evt.type === 'job_log_appended' || evt.type === 'event_malformed') {
    if (evt.job_id === id) return true
  }
  return false
}

export function useTaskLog(
  id: number | null,
  opts: { tail?: number; maxLines?: number } = {},
): TaskLogState {
  const tail = opts.tail ?? DEFAULT_TAIL
  const maxLines = opts.maxLines ?? DEFAULT_MAX

  const [entries, setEntries] = useState<Entry[]>([])
  const [status, setStatus] = useState<TaskLogStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [hasMoreBefore, setHasMoreBefore] = useState(false)
  const [loadingEarlier, setLoadingEarlier] = useState(false)

  // 游标与首行 offset 用 ref：SSE handler / 补拉闭包要拿最新值
  const cursorRef = useRef<number | null>(null)     // null = 冷启动未完成
  const startRef = useRef<number>(0)
  const pendingRef = useRef<StudioEvent[]>([])      // 冷启动完成前到达的事件
  const idRef = useRef<number | null>(id)
  idRef.current = id
  const fillingRef = useRef(false)

  // entries 的真相在 ref 里（SSE handler / 补拉是异步闭包），state 只是镜像；
  // 这样裁剪上限、推进首行 offset 都是普通赋值，不在 setState updater 里做副作用
  const entriesRef = useRef<Entry[]>([])
  const commit = useCallback(() => { setEntries(entriesRef.current.slice()) }, [])

  const appendEntries = useCallback((add: Entry[]) => {
    if (add.length === 0) return
    let next = entriesRef.current.concat(add)
    if (next.length > maxLines) {
      next = next.slice(next.length - maxLines)
      startRef.current = next[0].offset
      setHasMoreBefore(true)
    }
    entriesRef.current = next
    commit()
  }, [maxLines, commit])

  const applyEvent = useCallback((evt: StudioEvent) => {
    const cursor = cursorRef.current
    if (cursor === null) return
    if (evt.type === 'event_malformed') {
      const preview = typeof evt.raw_preview === 'string' ? evt.raw_preview : ''
      appendEntries([{ offset: cursor, text: `WARNING: ${i18n.t('logView.eventMalformed')} ${preview}`.trimEnd() }])
      return
    }
    const text = typeof evt.text === 'string' ? evt.text : ''
    const end = typeof evt.end_offset === 'number' ? evt.end_offset : null
    if (end !== null && end <= cursor) return  // 冷启动已含 / 重复
    appendEntries([{ offset: cursor, text }])
    if (end !== null) cursorRef.current = end
  }, [appendEntries])

  /** 从游标往后补拉（重连 / 状态变化）：把 SSE 丢掉的行补齐，再推进游标 */
  const fillAfter = useCallback(async () => {
    const tid = idRef.current
    const cursor = cursorRef.current
    if (tid == null || cursor === null || fillingRef.current) return
    fillingRef.current = true
    try {
      let after = cursor
      for (let guard = 0; guard < 20; guard++) {
        const page = await api.getLog(tid, { after, limit: tail })
        if (idRef.current !== tid) return
        if (page.lines.length === 0) break
        // 可能与期间到达的 SSE 行重叠：只收游标之后的
        const fresh = page.lines.filter((l) => l.offset >= (cursorRef.current ?? 0))
        appendEntries(fresh)
        if (page.end_offset > (cursorRef.current ?? 0)) cursorRef.current = page.end_offset
        if (page.end_offset <= after) break
        after = page.end_offset
        if (page.lines.length < tail) break
      }
    } catch {
      // 补拉失败不打扰；下一次事件 / 重连再试
    } finally {
      fillingRef.current = false
    }
  }, [appendEntries, tail])

  const loadTail = useCallback(async (tid: number) => {
    setStatus('loading')
    setError(null)
    cursorRef.current = null
    pendingRef.current = []
    try {
      const page = await api.getLog(tid, { tail })
      if (idRef.current !== tid) return
      startRef.current = page.start_offset
      cursorRef.current = page.end_offset
      setHasMoreBefore(page.has_more_before)
      entriesRef.current = page.lines.slice()
      commit()
      setStatus('ready')
      // 冷启动期间到达的事件按游标补进来
      const pending = pendingRef.current
      pendingRef.current = []
      for (const evt of pending) applyEvent(evt)
    } catch (e) {
      if (idRef.current !== tid) return
      setStatus('error')
      setError(String(e))
    }
  }, [tail, applyEvent, commit])

  useEffect(() => {
    entriesRef.current = []
    setEntries([])
    setHasMoreBefore(false)
    setLoadingEarlier(false)
    cursorRef.current = null
    pendingRef.current = []
    startRef.current = 0
    if (id == null) { setStatus('idle'); setError(null); return }
    void loadTail(id)
  }, [id, loadTail])

  useEventStream(
    useCallback((evt: StudioEvent) => {
      const tid = idRef.current
      if (tid == null) return
      if (_matches(evt, tid)) {
        if (cursorRef.current === null) pendingRef.current.push(evt)
        else applyEvent(evt)
        return
      }
      if ((evt.type === 'task_state_changed' && evt.task_id === tid)
        || (evt.type === 'job_state_changed' && evt.job_id === tid)) {
        void fillAfter()
      }
    }, [applyEvent, fillAfter]),
    { onOpen: () => { void fillAfter() } },
  )

  const loadEarlier = useCallback(() => {
    const tid = idRef.current
    if (tid == null || !hasMoreBefore || loadingEarlier) return
    setLoadingEarlier(true)
    void api.getLog(tid, { before: startRef.current, limit: tail })
      .then((page) => {
        if (idRef.current !== tid) return
        if (page.lines.length > 0) {
          startRef.current = page.start_offset
          entriesRef.current = page.lines.concat(entriesRef.current)
          commit()
        }
        setHasMoreBefore(page.has_more_before && page.lines.length > 0)
      })
      .catch(() => { /* 保留按钮，用户可再点 */ })
      .finally(() => setLoadingEarlier(false))
  }, [hasMoreBefore, loadingEarlier, tail, commit])

  const refresh = useCallback(() => {
    const tid = idRef.current
    if (tid != null) void loadTail(tid)
  }, [loadTail])

  const lines = useMemo(() => entries.map((e) => e.text), [entries])

  return {
    lines, status, error, hasMoreBefore, loadingEarlier, loadEarlier, refresh,
    downloadUrl: id == null ? null : api.logRawUrl(id),
  }
}
