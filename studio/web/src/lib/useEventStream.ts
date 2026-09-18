import { useEffect, useRef } from 'react'

export interface StudioEvent {
  type: string
  task_id?: number
  status?: string
  [key: string]: unknown
}

interface Options {
  /** 连接（含每次重连）成功时回调。EventSource 自动重连，断开期间事件会丢，
   * 这里给消费者一个口子重新 cold-fetch 补齐。第一次 onopen 也会触发；
   * 想区分「初次 vs 重连」消费者自己用 ref 计数。 */
  onOpen?: () => void
}

type Listener = (evt: StudioEvent) => void
type OpenListener = () => void

// ── 共享 EventSource ──────────────────────────────────────────────────────────
// 全 app 只开一条 /api/events 长连接，所有 useEventStream 调用方共享。
// 早期实现是每个调用方各开一条，结果 ~16 处 hook + StrictMode 双 mount 把
// 浏览器 HTTP/1.1 「单 origin 6 连接」上限打爆，普通 fetch 永远拿不到 socket
// （表现为 outputs 一直挂起、刷页面也加载不出）。
const _listeners = new Set<Listener>()
const _openListeners = new Set<OpenListener>()
let _es: EventSource | null = null

function _ensureOpen(): void {
  if (_es || typeof EventSource === 'undefined') return
  const es = new EventSource('/api/events')
  es.onopen = () => {
    for (const cb of _openListeners) {
      try { cb() } catch { /* 单个订阅者回调炸不影响其他 */ }
    }
  }
  es.onmessage = (e) => {
    let evt: StudioEvent
    try { evt = JSON.parse(e.data) as StudioEvent } catch { return }
    for (const cb of _listeners) {
      try { cb(evt) } catch { /* 同上 */ }
    }
  }
  es.onerror = () => {
    // EventSource 自动重连；这里只是个钩子，不主动关闭
  }
  _es = es
}

function _maybeClose(): void {
  if (_es && _listeners.size === 0 && _openListeners.size === 0) {
    _es.close()
    _es = null
  }
}

/**
 * 订阅 /api/events SSE 流。回调每次拿到一条事件。
 * 自动断线重连（EventSource 行为已经是这样）。
 *
 * 多个组件调用时共享同一条底层 EventSource，避免占满浏览器单 origin 连接配额。
 */
export function useEventStream(
  onEvent: (evt: StudioEvent) => void,
  options?: Options,
): void {
  // 用 ref 接闭包，让 useEffect 只在 mount 时绑一次，但 handler 内永远拿最新
  const onEventRef = useRef(onEvent)
  onEventRef.current = onEvent
  const onOpenRef = useRef(options?.onOpen)
  onOpenRef.current = options?.onOpen

  useEffect(() => {
    // jsdom / SSR / 老浏览器没有 EventSource — 不连 SSE 让组件在测试环境也能挂载
    if (typeof EventSource === 'undefined') return
    const handler: Listener = (evt) => onEventRef.current(evt)
    const openHandler: OpenListener = () => onOpenRef.current?.()
    _listeners.add(handler)
    _openListeners.add(openHandler)
    _ensureOpen()
    // 共享连接已经 open 时，新订阅者也要触发一次 onOpen 让它去 cold-fetch
    if (_es && _es.readyState === EventSource.OPEN) {
      try { onOpenRef.current?.() } catch { /* ignore */ }
    }
    return () => {
      _listeners.delete(handler)
      _openListeners.delete(openHandler)
      _maybeClose()
    }
  }, [])
}
