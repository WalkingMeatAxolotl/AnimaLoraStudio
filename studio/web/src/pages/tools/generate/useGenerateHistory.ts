/**
 * commit 16：测试出图历史栏（IndexedDB 持久化）。
 *
 * - 按 mode 三个独立桶：single / xy / compare
 * - 每条历史一个封面缩略图（dataUrl，~256px PNG）
 * - 跨 SPA 路由保持；浏览器 tab 关闭后清（IndexedDB 默认行为，与
 *   sessionStorage 不同 —— IndexedDB 跨 tab 持久，但用户决策是"tab 关
 *   就丢"，所以我们写 tab 级 sessionStorage 之上的 in-memory cache，
 *   IndexedDB 只用作页面刷新但不关 tab 时的恢复）
 *
 * 实际选择：IndexedDB（用户决策"无上限，几十 mb 对现代计算机太小"），
 * tab 关后留下也无伤大雅 —— 用户重开 tab 还能看到历史，符合"看图/对比"
 * 主流程。整体内存 / 磁盘可控（每条 thumb ~20KB，1000 条也才 20MB）。
 */
import { useEffect, useRef, useState } from 'react'
import { api } from '../../../api/client'

const DB_NAME = 'anima-generate-history'
const DB_VERSION = 1
const STORE = 'entries'

export type HistoryMode = 'single' | 'xy' | 'compare'

/** XY 历史回看用的 axis 元数据。回看时复用 PreviewXYGrid 渲染（带轴标签）。 */
export interface HistoryXYMeta {
  /** 'lora_ckpt' / 'lora_scale' / 'steps' / 'cfg_scale' */
  xAxis: string
  yAxis: string | null
  xValues: string[]
  yValues: Array<string | null>
  /** 每个 sample 的 xy 元数据；filename 来自 path 末段 */
  samples: Array<{
    path: string
    xy: { xi: number; yi: number; xv: string | number; yv: string | number | null }
  }>
}

export interface HistoryEntry {
  id: string
  mode: HistoryMode
  taskId: number
  createdAt: number
  thumbnailDataUrl: string  // 256px PNG/JPEG，封面（XY 取 (0,0)，对比取左图）
  /** 后端 cache 里的 filenames，按 sample order；点击时 fetch 原图，404 fallback thumb */
  filenames: string[]
  /** XY: 'XY M×N'；compare: '2×'；single: '' */
  badge?: string
  /** XY 模式才填：回看时重建 PreviewXYGrid 用 */
  xy?: HistoryXYMeta
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id' })
        store.createIndex('mode_createdAt', ['mode', 'createdAt'])
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function loadAll(): Promise<HistoryEntry[]> {
  try {
    const db = await openDb()
    return await new Promise<HistoryEntry[]>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly')
      const store = tx.objectStore(STORE)
      const req = store.getAll()
      req.onsuccess = () => {
        const items = (req.result as HistoryEntry[]).sort(
          (a, b) => b.createdAt - a.createdAt
        )
        resolve(items)
      }
      req.onerror = () => reject(req.error)
    })
  } catch {
    // IndexedDB 不可用（隐私模式 / Safari 限制）→ 返回空数组，不挂前端
    return []
  }
}

async function putEntry(entry: HistoryEntry): Promise<void> {
  try {
    const db = await openDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite')
      tx.objectStore(STORE).put(entry)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    /* 忽略：写失败不阻塞主流程 */
  }
}

async function deleteEntry(id: string): Promise<void> {
  try {
    const db = await openDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite')
      tx.objectStore(STORE).delete(id)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    /* ignore */
  }
}

async function clearMode(mode: HistoryMode): Promise<void> {
  try {
    const db = await openDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite')
      const store = tx.objectStore(STORE)
      const req = store.openCursor()
      req.onsuccess = () => {
        const cur = req.result
        if (cur) {
          if ((cur.value as HistoryEntry).mode === mode) cur.delete()
          cur.continue()
        }
      }
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    /* ignore */
  }
}

/** 把图片 URL → canvas 缩到 maxPx → PNG dataUrl。封面缩略图用。 */
export async function makeThumbnail(
  imageUrl: string, maxPx = 256
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const w = img.naturalWidth, h = img.naturalHeight
      const scale = Math.min(1, maxPx / Math.max(w, h))
      const tw = Math.max(1, Math.round(w * scale))
      const th = Math.max(1, Math.round(h * scale))
      const canvas = document.createElement('canvas')
      canvas.width = tw
      canvas.height = th
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        reject(new Error('no 2d context'))
        return
      }
      ctx.drawImage(img, 0, 0, tw, th)
      try {
        resolve(canvas.toDataURL('image/png'))
      } catch (e) {
        reject(e)
      }
    }
    img.onerror = () => reject(new Error(`failed to load ${imageUrl}`))
    img.src = imageUrl
  })
}

export interface UseGenerateHistoryResult {
  entries: HistoryEntry[]
  add: (entry: Omit<HistoryEntry, 'id' | 'createdAt'>) => Promise<void>
  clearByMode: (mode: HistoryMode) => Promise<void>
  /** 检查每条 entry 的第一张图是否还在 server cache 里；
   * 404 / fail 的 entry 删除（"原图已释放，留着只剩 thumbnail 没意义"）。
   * 返回删除的 entry 数量。 */
  pruneStale: () => Promise<number>
}

/** 全局 history 状态 hook。所有调用者共享一份内存视图（loadAll 一次）。 */
export function useGenerateHistory(): UseGenerateHistoryResult {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const loadedRef = useRef(false)

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true
    void loadAll().then(setEntries)
  }, [])

  const add = async (entry: Omit<HistoryEntry, 'id' | 'createdAt'>) => {
    const full: HistoryEntry = {
      ...entry,
      id: typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : Math.random().toString(36).slice(2),
      createdAt: Date.now(),
    }
    await putEntry(full)
    setEntries((prev) => [full, ...prev])
  }

  const clearByMode = async (mode: HistoryMode) => {
    await clearMode(mode)
    setEntries((prev) => prev.filter((e) => e.mode !== mode))
  }

  const pruneStale = async (): Promise<number> => {
    // 并发 HEAD 请求每条 entry 的第一张图；4xx/5xx 则视为失效，IndexedDB 删
    const stale: string[] = []
    await Promise.all(entries.map(async (e) => {
      const fn = e.filenames[0]
      if (!fn) return  // 没 filename 不动
      const url = api.generateSampleUrl(e.taskId, fn)
      try {
        const r = await fetch(url, { method: 'HEAD' })
        if (!r.ok) stale.push(e.id)
      } catch {
        // 网络错误（断网等）不算失效，留着下次再试
      }
    }))
    if (stale.length === 0) return 0
    await Promise.all(stale.map((id) => deleteEntry(id)))
    setEntries((prev) => prev.filter((e) => !stale.includes(e.id)))
    return stale.length
  }

  return { entries, add, clearByMode, pruneStale }
}
