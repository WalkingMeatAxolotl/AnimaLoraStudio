import { useEffect } from 'react'

export interface StudioEvent {
  type: string
  task_id?: number
  status?: string
  [key: string]: unknown
}

/**
 * 订阅 /api/events SSE 流。回调每次拿到一条事件。
 * 自动断线重连（EventSource 行为已经是这样）。
 */
export function useEventStream(onEvent: (evt: StudioEvent) => void): void {
  useEffect(() => {
    // jsdom / SSR / 老浏览器没有 EventSource — 不连 SSE 让组件在测试环境也能挂载
    if (typeof EventSource === 'undefined') return
    const es = new EventSource('/api/events')
    es.onmessage = (e) => {
      try {
        const evt = JSON.parse(e.data) as StudioEvent
        onEvent(evt)
      } catch {
        // 忽略畸形帧
      }
    }
    es.onerror = () => {
      // EventSource 自动重连；这里只是个钩子，不主动关闭
    }
    return () => es.close()
    // 故意只在挂载时绑定一次：onEvent 通常是闭包，组件可以用 ref 接收最新值
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
