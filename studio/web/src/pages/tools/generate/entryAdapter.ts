/** 出图时间线 entry(DB 单源,替代旧 disk/cache 双 source adapter)。
 *
 * 一条 entry = 一次图片任务(tasks 表行)。图字节可能在磁盘(save=on 落盘)
 * 或 session cache(temp),但前端不再关心 source —— server 在 timeline 端点
 * 把每张图拼好 URL;图不在(temp 会话结束 / 文件手删)→ `released`,
 * 参数仍可回填。
 *
 * helper 保持函数式(不用 object method):entry 要序列化 + 跨 hook 传递。
 */
import type { GenerateTimelineEntry } from '../../../api/client'
import type { GenerateParamsSnapshot } from './paramsSnapshot'
import { splitAxisRaw } from './xy'

/** XY 历史回看的 axis 元数据(PreviewXYGrid 重建用)。 */
export interface HistoryXYMeta {
  xAxis: string
  yAxis: string | null
  xValues: string[]
  yValues: Array<string | null>
  samples: Array<{
    path: string
    xy: { xi: number; yi: number; xv: string | number; yv: string | number | null }
    /** server 已拼好(disk 图 URL 或 temp sample URL) */
    imageUrl?: string
  }>
}

export interface HistoryImage {
  url: string
  thumbUrl?: string
  xi?: number
  yi?: number
}

export interface HistoryEntry {
  /** `t<taskId>`(React key / override 比对用) */
  id: string
  taskId: number
  status: string
  mode: 'single' | 'xy'
  /** ms(tasks.created_at 秒 ×1000) */
  createdAt: number
  /** 参数快照;老行 / 异常缺失时 undefined(点击只切图不回填) */
  params?: GenerateParamsSnapshot
  images: HistoryImage[]
  /** 首图当前可取(disk 文件在 / cache 命中) */
  available: boolean
  /** 图已不可取(temp 会话结束 / 文件手删):显示占位,参数仍在 */
  released: boolean
  storage: 'disk' | 'temp'
  /** xy 落盘文件夹名("xy plot 3"),批次标识显示用 */
  xyFolder?: string
  /** 盘上 composite 大图(导出 / 外站上传);应用内回看用 cells 渲网格 */
  compositeUrl?: string
  xyMeta?: HistoryXYMeta
}

/** server timeline 行 → HistoryEntry。xyMeta 从 params.xy_draft + images
 *  的 (xi, yi) 重建(轴 label 判读跟 entryBadge 同一套)。 */
export function adaptTimelineEntry(e: GenerateTimelineEntry): HistoryEntry {
  const params = (e.params ?? undefined) as GenerateParamsSnapshot | undefined
  const images: HistoryImage[] = e.images.map((i) => ({
    url: i.url,
    thumbUrl: i.thumb_url,
    xi: i.xi,
    yi: i.yi,
  }))
  const mode: 'single' | 'xy' = e.mode === 'xy' ? 'xy' : 'single'
  let xyMeta: HistoryXYMeta | undefined
  const cells = images.filter((i) => i.xi != null)
  if (mode === 'xy' && cells.length > 0) {
    const xDraft = params?.xy_draft?.x
    const yDraft = params?.xy_draft?.y
    const xValues = xDraft ? splitAxisRaw(xDraft.raw) : []
    const yValues: Array<string | null> = yDraft
      ? splitAxisRaw(yDraft.raw)
      : [null]
    xyMeta = {
      xAxis: xDraft?.axis ?? '',
      yAxis: yDraft?.axis ?? null,
      xValues,
      yValues,
      samples: cells
        .map((i) => ({
          path: `cell x${i.xi} y${i.yi}.png`,
          xy: {
            xi: i.xi ?? 0,
            yi: i.yi ?? 0,
            xv: xValues[i.xi ?? 0] ?? '',
            yv: yDraft ? (yValues[i.yi ?? 0] ?? null) : null,
          },
          imageUrl: i.url,
        }))
        .sort((a, b) => a.xy.yi - b.xy.yi || a.xy.xi - b.xy.xi),
    }
  }
  return {
    id: `t${e.task_id}`,
    taskId: e.task_id,
    status: e.status,
    mode,
    createdAt: Math.round((e.created_at ?? 0) * 1000),
    params,
    images,
    available: e.available,
    released: !e.available,
    storage: e.storage,
    xyFolder: e.xy_folder,
    compositeUrl: e.composite_url,
    xyMeta,
  }
}

/** entry 对应位置 idx 的大图 URL(无图 → 空串,消费方先看 released)。 */
export function entryImageUrl(e: HistoryEntry, idx = 0): string {
  return e.images[idx]?.url ?? e.images[0]?.url ?? ''
}

/** entry 缩略图 URL(小图栏用;temp 图无服务端 thumb,直接大图 + CSS 缩放)。 */
export function entryThumbUrl(e: HistoryEntry): string {
  const first = e.images[0]
  return first?.thumbUrl ?? first?.url ?? ''
}

/** entry 携带的 params snapshot(历史点击回填用;可能缺失)。 */
export function entryParams(e: HistoryEntry): GenerateParamsSnapshot | undefined {
  return e.params
}

/** entry 对应的 generate task id(`?task=` 深链按此命中)。 */
export function entryTaskId(e: HistoryEntry): number {
  return e.taskId
}

/** entry 显示标签:xy 用落盘文件夹名("xy plot 3");single 用落盘文件名
 *  (从 URL 反解);temp / 已释放回退 `#<taskId>`。 */
export function entryDisplayLabel(e: HistoryEntry): string {
  if (e.mode === 'xy' && e.xyFolder) return e.xyFolder
  const url = e.images[0]?.url ?? ''
  if (e.storage === 'disk' && url) {
    try {
      const last = decodeURIComponent(url.split('/').pop() ?? '')
      if (last) return last.replace(/\.png$/i, '')
    } catch { /* URL 异常 → 回退 */ }
  }
  return `#${e.taskId}`
}

/** XY 历史栏 entry 的 badge("XY 5×3")。 */
export function entryBadge(e: HistoryEntry): string | undefined {
  if (e.mode !== 'xy') return undefined
  if (e.params?.xy_draft) {
    const xLen = splitAxisRaw(e.params.xy_draft.x.raw).length
    const yLen = e.params.xy_draft.y ? splitAxisRaw(e.params.xy_draft.y.raw).length : 1
    return `XY ${xLen}×${yLen}`
  }
  if (e.xyMeta) {
    const xs = new Set(e.xyMeta.samples.map((s) => s.xy.xi))
    const ys = new Set(e.xyMeta.samples.map((s) => s.xy.yi))
    return `XY ${xs.size}×${ys.size || 1}`
  }
  return 'XY'
}
