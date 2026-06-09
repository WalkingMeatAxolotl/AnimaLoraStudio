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

/** XY 历史回看的 axis 元数据（仅 CacheEntry 重建 PreviewXYGrid 用；
 *  DiskEntry 是合成大图，不需要 per-cell 信息走网格视图） */
export interface HistoryXYMeta {
  xAxis: string
  yAxis: string | null
  xValues: string[]
  yValues: Array<string | null>
  samples: Array<{
    path: string
    xy: { xi: number; yi: number; xv: string | number; yv: string | number | null }
  }>
}

/** 持久 entry：磁盘上的 PNG，server disk-history 接口返回的纯派生数据。 */
export interface DiskEntry {
  source: 'disk'
  /** server 返回的稳定 id：'disk:<sha1-12>'（决策 #12，避免文件名带空格塞进 React key） */
  id: string
  mode: 'single' | 'xy'
  /** YYYY-MM-DD，对应文件夹 */
  date: string
  /** 文件名（含扩展名）；URL 已编码版在 imageUrl/thumbUrl 里 */
  filename: string
  /** 大图 URL，可直接 <img src=...> 用（server 端已 URL encode） */
  imageUrl: string
  /** 缩略图 URL，server 在线缩 + ETag */
  thumbUrl: string
  /** PNG mtime * 1000 */
  createdAt: number
  /** 从 PNG anima_params 解出来的参数快照 */
  params: GenerateParamsSnapshot
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

/** entry 显示标签（PreviewHistoryRail / badges 文案）。 */
export function entryDisplayLabel(e: HistoryEntry): string {
  switch (e.source) {
    case 'disk':
      return e.filename.replace(/\.png$/i, '')
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

/** 删除 entry：DiskEntry 走 DELETE endpoint；CacheEntry 仅本地 splice（无 server 删）。
 *  返回 server 端是否真删了文件（CacheEntry 永远 false）。 */
export async function entryDelete(e: HistoryEntry): Promise<{ removed: boolean }> {
  if (e.source !== 'disk') return { removed: false }
  // server 端 URL 是 /api/generate/disk/image/<date>/<mode>/<encoded>，
  // DELETE 走 /api/generate/disk/<date>/<mode>/<encoded>（少一层 image/）
  // 复用 imageUrl 末段的 encoded filename
  const encoded = e.imageUrl.slice(e.imageUrl.lastIndexOf('/') + 1)
  const url = `/api/generate/disk/${e.date}/${e.mode}/${encoded}`
  const r = await fetch(url, { method: 'DELETE' })
  if (!r.ok) throw new Error(`delete failed: ${r.status}`)
  const data = await r.json() as { ok: boolean; noop?: boolean }
  return { removed: !data.noop }
}
