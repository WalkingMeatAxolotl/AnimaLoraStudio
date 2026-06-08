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
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type DiskGenerateHistoryEntry } from '../../../api/client'
import type { GenerateParamsSnapshot } from './paramsSnapshot'

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
  /** 测试参数快照（用于历史点击回填 prefs）。老 entry 缺此字段 → 回填 noop。 */
  params?: GenerateParamsSnapshot
  /** 落盘成功后回写：第一张图（single）/ 合成图（xy）的 server path；用于和
   *  GET /api/generate/disk-history 拉到的 disk entries 做 dedup。 */
  diskPath?: string
  /** 缺省时按 generateSampleUrl(taskId, filename) 构造（cache 来源 entry）。
   *  磁盘来源 entry 必填（指向 /api/generate/disk-image/...），因为它的 taskId
   *  是 sentinel，generateSampleUrl 路径不可用。 */
  imageUrls?: string[]
}

/** entry 对应位置 idx 的图片 URL。disk entry 走 imageUrls，cache entry 走
 *  generateSampleUrl(taskId, filename)。所有历史回看渲染处都该用这个 helper。 */
export function historyImageUrl(entry: HistoryEntry, idx = 0): string {
  const explicit = entry.imageUrls?.[idx]
  if (explicit) return explicit
  const fn = entry.filenames[idx] ?? ''
  return api.generateSampleUrl(entry.taskId, fn)
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

async function updateEntry(id: string, patch: Partial<HistoryEntry>): Promise<HistoryEntry | null> {
  try {
    const db = await openDb()
    return await new Promise<HistoryEntry | null>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite')
      const store = tx.objectStore(STORE)
      const getReq = store.get(id)
      getReq.onsuccess = () => {
        const cur = getReq.result as HistoryEntry | undefined
        if (!cur) { resolve(null); return }
        const next = { ...cur, ...patch, id: cur.id }
        store.put(next)
        tx.oncomplete = () => resolve(next)
      }
      tx.onerror = () => reject(tx.error)
    })
  } catch {
    return null
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

/** disk-history server entry → HistoryEntry shape。
 *  - taskId = -1 sentinel（cache 接口不可用，所有图片走 imageUrls）
 *  - thumbnailDataUrl 直接放 disk-image URL（不预生成缩略图，浏览器自缩；
 *    落盘历史规模有限，避免 disk → fetch → canvas → dataURL 的额外开销）
 *  - id 保持服务端给的 "disk:<date>:<mode>:image_<N>"（前端按此 dedup）
 */
function diskEntryToHistory(d: DiskGenerateHistoryEntry): HistoryEntry {
  return {
    id: d.id,
    mode: d.mode,
    taskId: -1,
    createdAt: d.created_at * 1000,  // server 给的是秒；HistoryEntry.createdAt 是 ms
    thumbnailDataUrl: d.url,
    filenames: [d.filename],
    imageUrls: [d.url],
    diskPath: d.path,
    params: d.params as unknown as GenerateParamsSnapshot,
  }
}

async function loadDisk(): Promise<HistoryEntry[]> {
  try {
    const resp = await api.listDiskGenerateHistory()
    return resp.entries.map(diskEntryToHistory)
  } catch {
    return []
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
  /** 返回新 entry 的 id，方便 add 后续 patch（如落盘成功回写 diskPath） */
  add: (entry: Omit<HistoryEntry, 'id' | 'createdAt'>) => Promise<string>
  /** 局部更新 entry —— 主要给 saveTestImages 回写 diskPath 用 */
  patch: (id: string, patch: Partial<HistoryEntry>) => Promise<void>
  clearByMode: (mode: HistoryMode) => Promise<void>
  /** 检查每条 entry 的第一张图是否还在 server cache 里；
   * 404 / fail 的 entry：若已落盘（有 diskPath）保留（磁盘还能看），否则删除。
   * 返回删除的 entry 数量。 */
  pruneStale: () => Promise<number>
}

/** 合并 IDB + disk entries：diskPath 重复时 IDB 优先（带本地 thumbnail）。 */
function mergeEntries(idb: HistoryEntry[], disk: HistoryEntry[]): HistoryEntry[] {
  const usedDiskPaths = new Set(
    idb.map((e) => e.diskPath).filter((p): p is string => Boolean(p))
  )
  const remainingDisk = disk.filter((d) => !d.diskPath || !usedDiskPaths.has(d.diskPath))
  return [...idb, ...remainingDisk].sort((a, b) => b.createdAt - a.createdAt)
}

/** 全局 history 状态 hook。IDB 持久化 + 磁盘 sidecar 兜底（跨会话回看）。 */
export function useGenerateHistory(): UseGenerateHistoryResult {
  const [idbEntries, setIdbEntries] = useState<HistoryEntry[]>([])
  const [diskEntries, setDiskEntries] = useState<HistoryEntry[]>([])
  const loadedRef = useRef(false)

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true
    void loadAll().then(setIdbEntries)
    void loadDisk().then(setDiskEntries)
  }, [])

  const entries = useMemo(
    () => mergeEntries(idbEntries, diskEntries),
    [idbEntries, diskEntries],
  )

  const add = async (entry: Omit<HistoryEntry, 'id' | 'createdAt'>) => {
    const full: HistoryEntry = {
      ...entry,
      id: typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : Math.random().toString(36).slice(2),
      createdAt: Date.now(),
    }
    await putEntry(full)
    setIdbEntries((prev) => [full, ...prev])
    return full.id
  }

  const patch = async (id: string, p: Partial<HistoryEntry>) => {
    const next = await updateEntry(id, p)
    if (next) setIdbEntries((prev) => prev.map((e) => (e.id === id ? next : e)))
  }

  const clearByMode = async (mode: HistoryMode) => {
    // 只清 IDB；磁盘文件用户得手动去文件夹删（避免误删历史档案）。
    // 下次 loadDisk 时磁盘 entries 仍会出现 — 这是有意保留。
    await clearMode(mode)
    setIdbEntries((prev) => prev.filter((e) => e.mode !== mode))
  }

  const pruneStale = async (): Promise<number> => {
    // 只对 IDB entries 操作（disk entries 是只读视图，背后是磁盘文件）。
    // 已落盘（diskPath）的 IDB entry：图在磁盘上随时回看，不删（仅 HEAD 失败说明
    // 内存 cache 释放了，并不代表"图没了"）。
    const stale: string[] = []
    await Promise.all(idbEntries.map(async (e) => {
      const fn = e.filenames[0]
      if (!fn) return  // 没 filename 不动
      if (e.diskPath) return  // 已落盘 → 永远不剪
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
    setIdbEntries((prev) => prev.filter((e) => !stale.includes(e.id)))
    return stale.length
  }

  return { entries, add, patch, clearByMode, pruneStale }
}
