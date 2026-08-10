/** 测试出图历史栏 —— DB 单源。
 *
 * 唯一来源:`GET /api/generate/timeline`(tasks 表台账,server 端拼好图 URL
 * 与 available)。旧「disk 扫盘 ∪ cache index」双源已退役 —— 双源 union 不去
 * 重正是「刷新后列表翻倍」的结构性根源。
 *
 * pending/running/scheduled 行在此过滤掉:rail 的 live 项由 Generate.tsx 的
 * listQueueLive(队列瞬态,负责显示跟随/取消)渲染,时间线只吃 terminal 行,
 * 两边按 taskId 天然不重叠。
 *
 * 刷新时机(均由 Generate.tsx 触发):mount / task done(ingest)/
 * `generate_images_updated` SSE(落盘 executor 异步完成)/ 用户手动。
 */
import { useEffect, useRef, useState } from 'react'
import { api } from '../../../api/client'
import { adaptTimelineEntry, type HistoryEntry } from './entryAdapter'

export type { HistoryEntry, HistoryXYMeta } from './entryAdapter'

const LIVE_STATUSES = new Set(['pending', 'running', 'scheduled'])

export interface UseGenerateHistoryResult {
  /** terminal 行(done / 有产出的 canceled),server 已按 id desc 排 */
  entries: HistoryEntry[]
  loading: boolean
  /** 重拉 timeline(task done / SSE / 手动刷新共用一个入口) */
  refresh: () => Promise<void>
}

export function useGenerateHistory(): UseGenerateHistoryResult {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [loading, setLoading] = useState(true)
  const loadedRef = useRef(false)

  const refresh = async () => {
    try {
      const data = await api.listGenerateTimeline()
      setEntries(
        data.entries
          .filter((e) => !LIVE_STATUSES.has(e.status))
          .map(adaptTimelineEntry),
      )
    } catch {
      // 拉取失败不挂前端 —— 保留上次列表
    }
  }

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true
    setLoading(true)
    void refresh().finally(() => setLoading(false))
  }, [])

  return { entries, loading, refresh }
}
