/** 测试出图历史栏（plan Step 4 重写）。
 *
 * 设计转向（决策 #5 / #9 / #12）：
 * - 砍 IndexedDB 整层（~-250 LOC）—— 不再做"params + thumb 双份持久化"
 * - 单一 source of truth = 磁盘 PNG（持久 entry）+ server cache（临时 entry）
 * - mount 总拉 disk-history（不分模式）；不受 save_test_images 开关影响
 * - session in-memory 持有所有 entry，关 tab 即丢；刷新页面重 mount 拉 disk
 * - HistoryEntry 是 union: DiskEntry | CacheEntry（adapter pattern，分支收敛
 *   到 entryAdapter.ts）
 * - 没有 merge upgrade / dedup / pruneStale —— 两类 entry 自然独立、不会撞 id
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  entryDelete,
  type CacheEntry,
  type DiskEntry,
  type HistoryEntry,
} from './entryAdapter'
import type { GenerateParamsSnapshot } from './paramsSnapshot'

export type { CacheEntry, DiskEntry, HistoryEntry, HistoryXYMeta } from './entryAdapter'

interface DiskHistoryServerEntry {
  id: string
  date: string
  mode: 'single' | 'xy'
  filename: string
  path: string
  image_url: string
  thumb_url: string
  created_at: number
  schema_version: number
  params: unknown
}

interface DiskHistoryResponse {
  entries: DiskHistoryServerEntry[]
}

function diskEntryFromServer(d: DiskHistoryServerEntry): DiskEntry {
  return {
    source: 'disk',
    id: d.id,
    mode: d.mode,
    date: d.date,
    filename: d.filename,
    imageUrl: d.image_url,
    thumbUrl: d.thumb_url,
    createdAt: d.created_at * 1000,  // server 给秒；entry.createdAt 用 ms
    params: d.params as GenerateParamsSnapshot,
  }
}

export interface UseGenerateHistoryResult {
  /** 所有 entry，按 createdAt desc 排 */
  entries: HistoryEntry[]
  /** disk-history 拉取中（mount 短暂期间为 true） */
  loading: boolean
  /** 添加新 entry（出图完成 / 落盘后调用） */
  add: (entry: HistoryEntry) => void
  /** 删除 entry：DiskEntry 调 DELETE endpoint；CacheEntry 仅本地 splice */
  remove: (id: string) => Promise<void>
  /** 手动重拉 disk-history（多 tab 同步 / 外部改 studio_data 后用户主动刷新） */
  refresh: () => Promise<void>
}

export function useGenerateHistory(): UseGenerateHistoryResult {
  const [diskEntries, setDiskEntries] = useState<DiskEntry[]>([])
  const [cacheEntries, setCacheEntries] = useState<CacheEntry[]>([])
  const [loading, setLoading] = useState(true)
  const loadedRef = useRef(false)

  const fetchDisk = async () => {
    setLoading(true)
    try {
      const r = await fetch('/api/generate/disk/history')
      if (!r.ok) return
      const data = (await r.json()) as DiskHistoryResponse
      setDiskEntries(data.entries.map(diskEntryFromServer))
    } catch {
      // 拉取失败不挂前端 —— 历史栏只显示 session 期间出的 CacheEntry
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true
    void fetchDisk()
  }, [])

  // entries union 按 createdAt desc 排。两类 entry 自然独立 —— 同一张图
  // 不会既是 disk 又是 cache（开关冻结后 task 走单一路径）。
  const entries = useMemo<HistoryEntry[]>(
    () => [...diskEntries, ...cacheEntries].sort((a, b) => b.createdAt - a.createdAt),
    [diskEntries, cacheEntries],
  )

  const add = (entry: HistoryEntry) => {
    if (entry.source === 'disk') {
      setDiskEntries((prev) => [entry, ...prev])
    } else {
      setCacheEntries((prev) => [entry, ...prev])
    }
  }

  const remove = async (id: string) => {
    const target = entries.find((e) => e.id === id)
    if (!target) return
    if (target.source === 'disk') {
      try {
        await entryDelete(target)
      } catch {
        // server 失败仍本地剔（用户能看到列表里少了一条；下次 refresh 时
        // 如果文件真在仍会回来 —— 是预期的"乐观删除"模式）
      }
      setDiskEntries((prev) => prev.filter((e) => e.id !== id))
    } else {
      setCacheEntries((prev) => prev.filter((e) => e.id !== id))
    }
  }

  return { entries, loading, add, remove, refresh: fetchDisk }
}
