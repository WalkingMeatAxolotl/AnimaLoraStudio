/** 测试出图历史栏。
 *
 * 两条 source 都从 server 拉（本地零持久化层）：
 * - DiskEntry：`/api/generate/disk/history` 扫 `studio_data/test/<date>/` 下
 *   落盘 PNG metadata（save_test_images=true 写的）
 * - CacheEntry：`/api/generate/cache/index` 当前 session 加密磁盘 cache
 *   (save_test_images=false 时；server 重启 / SSE 断连 30s + LRU 后丢)
 *
 * 之前的版本前端 useState 持 cacheEntries —— 切路由组件 unmount 就丢了；现在
 * cache 由 server 持，前端只 fetch 视图，切路由 mount 再拉。零持久化心智 +
 * 零脏数据（server 死了 cache index 也空了，不会指向不存在的图）。
 *
 * add() 被砍：server 端 image_done 自动入 cache，前端只 refresh 拉新视图。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type CacheGenerateHistoryEntry } from '../../../api/client'
import {
  entryDelete,
  type CacheEntry,
  type DiskEntry,
  type HistoryEntry,
  type HistoryXYMeta,
} from './entryAdapter'
import type { GenerateParamsSnapshot } from './paramsSnapshot'

export type { CacheEntry, DiskEntry, HistoryEntry, HistoryXYMeta } from './entryAdapter'

interface DiskHistoryServerXYMeta {
  x_axis: string | null
  y_axis: string | null
  x_values: string[]
  y_values: Array<string | null>
  samples: Array<{
    path: string
    xy: { xi: number; yi: number; xv: string | null; yv: string | null }
    image_url: string
  }>
}

interface DiskHistoryServerEntry {
  id: string
  date: string
  mode: 'single' | 'xy'
  filename?: string
  folder?: string
  path: string
  image_url: string
  thumb_url: string
  created_at: number
  schema_version: number
  params: unknown
  xy_meta?: DiskHistoryServerXYMeta | null
}

interface DiskHistoryResponse {
  entries: DiskHistoryServerEntry[]
}

function xyMetaFromServer(meta: DiskHistoryServerXYMeta): HistoryXYMeta {
  return {
    xAxis: meta.x_axis ?? '',
    yAxis: meta.y_axis,
    xValues: meta.x_values,
    yValues: meta.y_values,
    samples: meta.samples.map((s) => ({
      path: s.path,
      xy: { xi: s.xy.xi, yi: s.xy.yi, xv: s.xy.xv ?? '', yv: s.xy.yv },
      imageUrl: s.image_url,
    })),
  }
}

function diskEntryFromServer(d: DiskHistoryServerEntry): DiskEntry {
  return {
    source: 'disk',
    id: d.id,
    mode: d.mode,
    date: d.date,
    filename: d.filename,
    folder: d.folder,
    imageUrl: d.image_url,
    thumbUrl: d.thumb_url,
    createdAt: d.created_at * 1000,  // server 给秒；entry.createdAt 用 ms
    params: d.params as GenerateParamsSnapshot,
    xyMeta: d.xy_meta ? xyMetaFromServer(d.xy_meta) : undefined,
  }
}

/** server cache index → 前端 CacheEntry。XY 模式从 server samples 重建 xyMeta；
 *  axis 元数据从 params.xy_draft 派生（跟 entryBadge 同一套）。 */
function cacheEntryFromServer(c: CacheGenerateHistoryEntry): CacheEntry {
  const params = c.params as unknown as GenerateParamsSnapshot
  let xyMeta: HistoryXYMeta | undefined
  if (c.mode === 'xy' && c.samples && c.samples.length > 0) {
    const xDraft = params.xy_draft?.x
    const yDraft = params.xy_draft?.y
    const xValues = xDraft?.raw.split(',').map((s) => s.trim()).filter(Boolean) ?? []
    const yValues = yDraft
      ? yDraft.raw.split(',').map((s) => s.trim()).filter(Boolean)
      : [null as string | null]
    xyMeta = {
      xAxis: xDraft?.axis ?? '',
      yAxis: yDraft?.axis ?? null,
      xValues,
      yValues,
      samples: c.samples.map((s) => ({
        path: s.filename,
        xy: {
          xi: s.xy.xi, yi: s.xy.yi,
          xv: typeof s.xy.xv === 'number' ? s.xy.xv : (s.xy.xv ?? ''),
          yv: s.xy.yv,
        },
      })),
    }
  }
  return {
    source: 'cache',
    id: c.id,
    mode: c.mode,
    taskId: c.taskId,
    createdAt: c.createdAt,
    filenames: c.filenames,
    params,
    xyMeta,
  }
}

export interface UseGenerateHistoryResult {
  /** 所有 entry，按 createdAt desc 排 */
  entries: HistoryEntry[]
  /** 任一 source 在拉取中 */
  loading: boolean
  /** 删除 entry：DiskEntry 调 DELETE endpoint；CacheEntry 仅本地 splice */
  remove: (id: string) => Promise<void>
  /** 手动重拉 disk-history（多 tab 同步 / 外部改 studio_data 后用户主动刷新） */
  refresh: () => Promise<void>
  /** 重拉 cache index（image_done SSE 后 Generate.tsx 调） */
  refreshCache: () => Promise<void>
}

export function useGenerateHistory(): UseGenerateHistoryResult {
  const [diskEntries, setDiskEntries] = useState<DiskEntry[]>([])
  const [cacheEntries, setCacheEntries] = useState<CacheEntry[]>([])
  const [loading, setLoading] = useState(true)
  const loadedRef = useRef(false)

  const fetchDisk = async () => {
    try {
      const r = await fetch('/api/generate/disk/history')
      if (!r.ok) return
      const data = (await r.json()) as DiskHistoryResponse
      setDiskEntries(data.entries.map(diskEntryFromServer))
    } catch {
      // 拉取失败不挂前端 —— 历史栏只显示另一 source
    }
  }

  const fetchCache = async () => {
    try {
      const data = await api.listCacheGenerateHistory()
      setCacheEntries(data.entries.map(cacheEntryFromServer))
    } catch {
      // 同上
    }
  }

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true
    setLoading(true)
    void Promise.all([fetchDisk(), fetchCache()]).finally(() => setLoading(false))
  }, [])

  // entries union 按 createdAt desc 排。两类 entry 自然独立 —— 同一 task
  // 不会既走 disk 又走 cache（save_test_images 开关 dispatch 时冻结，task
  // 只走单一路径）。
  const entries = useMemo<HistoryEntry[]>(
    () => [...diskEntries, ...cacheEntries].sort((a, b) => b.createdAt - a.createdAt),
    [diskEntries, cacheEntries],
  )

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
      // CacheEntry：本地剔即可（server 端 LRU / shutdown 会清理）
      setCacheEntries((prev) => prev.filter((e) => e.id !== id))
    }
  }

  return { entries, loading, remove, refresh: fetchDisk, refreshCache: fetchCache }
}
