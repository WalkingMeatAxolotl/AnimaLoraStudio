/** 历史 entry source adapter（plan 决策 #12）。
 *
 * 所有"按 entry.source 分支"的逻辑收敛到这一个文件 —— UI / handler / hook 全
 * 部走这里的 helper，**不直接 switch entry.source**。
 *
 * 未来加第三种 source（'upload' / 'remote'）只改这个文件，消费端零改动。
 *
 * 为什么用 function helper 不用 object method（`entry.imageUrl()`）：
 * - entry 要序列化进 sessionStorage（撤销 snapshot）+ 跨 hook 传递；object method
 *   不能 serialize
 * - function helper 跟 React/TS 风格更一致
 * - 测试时 mock helper 比 mock class method 干净
 */
import { api } from '../../../api/client'
import type { GenerateParamsSnapshot } from './paramsSnapshot'

/** XY 历史回看的 axis 元数据（CacheEntry + DiskEntry 共用 —— PreviewXYGrid 重建用）。
 *
 *  - CacheEntry: samples[].path 是 cache 文件名；imageUrl 留空，PreviewXYGrid
 *    fallback 走 `api.generateSampleUrl(taskId, filename)`
 *  - DiskEntry: server 已 encode 好 imageUrl（`/api/generate/disk/image/<date>/xy/<folder>/<cell>`），
 *    PreviewXYGrid 直接吃 */
export interface HistoryXYMeta {
  xAxis: string
  yAxis: string | null
  xValues: string[]
  yValues: Array<string | null>
  samples: Array<{
    path: string
    xy: { xi: number; yi: number; xv: string | number; yv: string | number | null }
    /** disk-served 时 server 端已 URL-encode 好；cache 时 undefined → 回退 api.generateSampleUrl */
    imageUrl?: string
  }>
}

/** 持久 entry：磁盘上的 PNG / 文件夹，server disk-history 接口返回的纯派生数据。
 *
 * single：一张 PNG（filename + imageUrl 指向它）
 * xy：一个 `xy plot N/` 文件夹（folder + imageUrl 指向 composite，
 *     xyMeta.samples 是每 cell 信息 + 直读 URL）。
 */
export interface DiskEntry {
  source: 'disk'
  /** server 返回的稳定 id：'disk:<sha1-12>'（决策 #12，避免文件名带空格塞进 React key） */
  id: string
  mode: 'single' | 'xy'
  /** YYYY-MM-DD，对应文件夹 */
  date: string
  /** single：PNG 文件名（含扩展名）；xy：undefined（用 folder） */
  filename?: string
  /** xy：文件夹名 `xy plot <N>`；single：undefined */
  folder?: string
  /** 大图 URL，可直接 <img src=...> 用（server 端已 URL encode）。xy 模式指向 composite。 */
  imageUrl: string
  /** 缩略图 URL，server 在线缩 + ETag */
  thumbUrl: string
  /** PNG / composite mtime */
  createdAt: number
  /** 从 PNG anima_params 解出来的参数快照（xy 模式是 composite 的 XY snapshot） */
  params: GenerateParamsSnapshot
  /** xy 模式 per-cell 元数据（PreviewXYGrid 重建用）；single 模式 undefined */
  xyMeta?: HistoryXYMeta
}

/** 临时 entry：仅在 server 内存 cache 中存活，session 期间使用。
 *  关 tab / server 重启 / LRU 剔除即丢。 */
export interface CacheEntry {
  source: 'cache'
  id: string  // uuid
  mode: 'single' | 'xy'
  /** daemon task id，cache URL 构造用 */
  taskId: number
  createdAt: number
  /** server cache 里的文件名列表（XY 是 per-cell N 张；single 是 1 张） */
  filenames: string[]
  /** 出图当时构造的参数快照（不从 cache 读） */
  params: GenerateParamsSnapshot
  /** XY 模式的 axis + per-cell 元数据，重建 PreviewXYGrid 用 */
  xyMeta?: HistoryXYMeta
}

export type HistoryEntry = DiskEntry | CacheEntry

// ---------------------------------------------------------------------------
// 按 source 切的 helper —— **唯一**允许 switch entry.source 的地方
// ---------------------------------------------------------------------------

/** entry 对应位置 idx 的大图 URL。 */
export function entryImageUrl(e: HistoryEntry, idx = 0): string {
  switch (e.source) {
    case 'disk':
      return e.imageUrl  // server 已 encode
    case 'cache': {
      const fn = e.filenames[idx] ?? e.filenames[0] ?? ''
      return api.generateSampleUrl(e.taskId, fn)
    }
  }
}

/** entry 缩略图 URL（小图栏用）。 */
export function entryThumbUrl(e: HistoryEntry): string {
  switch (e.source) {
    case 'disk':
      return e.thumbUrl
    case 'cache':
      // CacheEntry 不做服务端缩略图 —— 直接用大图 URL + CSS 缩放
      // (session 期间出图不多，浏览器加载几张原图可接受；不要为这种短期 entry
      // 加 IDB / 服务端 thumb 复杂度)
      return entryImageUrl(e, 0)
  }
}

/** entry 携带的 params snapshot（用于历史点击回填）。 */
export function entryParams(e: HistoryEntry): GenerateParamsSnapshot {
  return e.params  // 两个 source 字段名一致
}

/** entry 对应的 generate task id（0.17 P-H `?task=` 深链按此命中条目）。
 *  cache 顶层带 taskId；disk 藏在 server enrich 进 PNG anima_params 的 task_id。 */
export function entryTaskId(e: HistoryEntry): number | undefined {
  switch (e.source) {
    case 'cache':
      return e.taskId
    case 'disk':
      return (e.params as { task_id?: number }).task_id
  }
}

/** entry 显示标签（PreviewHistoryRail / badges 文案）。 */
export function entryDisplayLabel(e: HistoryEntry): string {
  switch (e.source) {
    case 'disk':
      // xy 用 folder ("xy plot 3")；single 用 filename 去 .png 后缀
      if (e.mode === 'xy' && e.folder) return e.folder
      return (e.filename ?? '').replace(/\.png$/i, '')
    case 'cache':
      return `#${e.taskId}`
  }
}

/** XY 历史栏 entry 的 badge（"XY 5×3"）。 */
export function entryBadge(e: HistoryEntry): string | undefined {
  if (e.mode !== 'xy') return undefined
  if (e.source === 'cache' && e.xyMeta) {
    const xs = new Set(e.xyMeta.samples.map((s) => s.xy.xi))
    const ys = new Set(e.xyMeta.samples.map((s) => s.xy.yi))
    return `XY ${xs.size}×${ys.size || 1}`
  }
  if (e.source === 'disk' && e.params.xy_draft) {
    const xLen = e.params.xy_draft.x.raw.split(',').filter((s) => s.trim()).length
    const yLen = e.params.xy_draft.y?.raw.split(',').filter((s) => s.trim()).length ?? 1
    return `XY ${xLen}×${yLen}`
  }
  return 'XY'
}

/** 删除 entry：
 *  - DiskEntry single：单文件 DELETE `/api/generate/disk/<date>/single/<encoded>`
 *  - DiskEntry xy：整文件夹 DELETE `/api/generate/disk/<date>/xy/<encoded folder>`
 *  - CacheEntry：仅本地 splice（无 server 删）
 *  返回 server 端是否真删了（CacheEntry 永远 false）。 */
export async function entryDelete(e: HistoryEntry): Promise<{ removed: boolean }> {
  if (e.source !== 'disk') return { removed: false }
  let url: string
  if (e.mode === 'xy') {
    if (!e.folder) throw new Error('xy disk entry missing folder')
    url = `/api/generate/disk/${e.date}/xy/${encodeURIComponent(e.folder)}`
  } else {
    if (!e.filename) throw new Error('single disk entry missing filename')
    url = `/api/generate/disk/${e.date}/single/${encodeURIComponent(e.filename)}`
  }
  const r = await fetch(url, { method: 'DELETE' })
  if (!r.ok) throw new Error(`delete failed: ${r.status}`)
  const data = await r.json() as { ok: boolean; noop?: boolean }
  return { removed: !data.noop }
}
