import { useCallback, useRef, useState } from 'react'

/** Job / Task 的最小公共形状 — 回放 guard 只看这三个字段。 */
interface ReplayableItem {
  id: number
  status: string
  version_id?: number | null
}

export const splitLog = (log: string): string[] => (log ? log.split('\n') : [])

/**
 * 「最近一次任务 + 日志回放」状态容器（Tagging / Regularization 共用）。
 *
 * 进页面 / SSE 重连（onOpen）时调 refresh() 从服务端 hydrate 最近一次
 * 任务和全量日志；SSE 增量日志照常走 setItem / setLogs append。
 *
 * refresh 三层防回退 guard：
 * 1. 本地正在跟踪 running/pending 任务时，不被另一个 id 的结果顶掉
 * 2. 同 id 时服务端日志比本地短（文件落盘滞后）不覆盖
 * 3. 服务端无任务时只清掉「属于其它 version」的残留状态
 *
 * onHydrated 在 refresh 实际改写状态时回调（清空时传 null），给调用方
 * 同步衍生状态（如 aiBusy）。fetchLatest / onHydrated 走 ref，不要求
 * 调用方 memoize。
 */
export function useLatestJobReplay<T extends ReplayableItem>(
  vid: number | null,
  fetchLatest: (vid: number) => Promise<{ item: T | null; log: string }>,
  onHydrated?: (item: T | null) => void,
) {
  const [item, setItem] = useState<T | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const itemRef = useRef<T | null>(null)
  const logsRef = useRef<string[]>([])
  const itemIdRef = useRef<number | null>(null)
  itemRef.current = item
  logsRef.current = logs
  itemIdRef.current = item?.id ?? null
  const fetchRef = useRef(fetchLatest)
  fetchRef.current = fetchLatest
  const onHydratedRef = useRef(onHydrated)
  onHydratedRef.current = onHydrated

  const refresh = useCallback(async () => {
    if (!vid) return
    try {
      const r = await fetchRef.current(vid)
      if (!r.item) {
        if (itemRef.current?.version_id !== vid) {
          setItem(null)
          setLogs([])
          onHydratedRef.current?.(null)
        }
        return
      }
      const current = itemRef.current
      if (
        current?.version_id === vid &&
        current.id !== r.item.id &&
        (current.status === 'pending' || current.status === 'running')
      ) return
      const hydratedLogs = splitLog(r.log)
      if (
        current?.version_id === vid &&
        current.id === r.item.id &&
        logsRef.current.length > hydratedLogs.length
      ) return
      setItem(r.item)
      setLogs(hydratedLogs)
      onHydratedRef.current?.(r.item)
    } catch { /* hydrate 失败不阻塞页面 */ }
  }, [vid])

  return { item, logs, setItem, setLogs, itemIdRef, refresh }
}
