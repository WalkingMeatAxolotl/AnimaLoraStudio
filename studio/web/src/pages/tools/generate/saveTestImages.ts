/** 测试出图自动落盘（Settings → testing → save_test_images 开启时）。
 *
 * 路径：
 *  - single: `studio_data/test/<YYYY-MM-DD>/single/single image N.png`
 *  - xy:     `studio_data/test/<YYYY-MM-DD>/xy/xy plot N/{xy plot.png + cell x{xi} y{yi}.png ...}`
 *
 * single 每张 sample 单独上传（一图一 POST）；
 * xy 一次 multipart 发：composite + N 张 cell + cells_manifest（按 cell
 * 物化的 single-snapshot），server 端落到同一文件夹下，atomic 整个 commit。
 *
 * 上传时附 params JSON（GenerateParamsSnapshot）：后端写进 PNG `anima_params`
 * tEXt + single 写 a1111 `parameters` 块，供历史栏回看 / 用户拷走 PNG 复用参数。
 *
 * 失败 / 后端 403（开关被关）都静默吞掉 —— 不打扰用户主流程。
 */
import { api } from '../../../api/client'
import { composeXYMatrix, type ExportInput } from './exportXY'
import { buildCellSnapshot, type GenerateParamsSnapshot } from './paramsSnapshot'

interface SingleSaveResult {
  path: string
  index: number
  filename: string
}

interface XYSaveResult {
  folder: string
  index: number
  composite: string
  cells: string[]
}

/** 落盘 single 模式所有 sample。返回每张图的 server path（与 filenames 同序，
 *  失败的位置为 null）。调用者用第 0 张的 path 作为 entry.diskPath（去重 key）。 */
export async function saveSingleSamples(
  taskId: number,
  filenames: string[],
  params: GenerateParamsSnapshot,
): Promise<Array<string | null>> {
  const paths: Array<string | null> = []
  for (const fn of filenames) {
    try {
      const res = await fetch(api.generateSampleUrl(taskId, fn))
      if (!res.ok) { paths.push(null); continue }
      const blob = await res.blob()
      const fd = new FormData()
      fd.append('mode', 'single')
      fd.append('image', blob, 'single.png')
      fd.append('params', JSON.stringify(params))
      const r = await fetch('/api/generate/save', { method: 'POST', body: fd })
      if (!r.ok) { paths.push(null); continue }
      const data = await r.json() as SingleSaveResult
      paths.push(data.path)
    } catch {
      paths.push(null)
    }
  }
  return paths
}

/** 落盘 xy 文件夹（composite + 每 cell 原图）。返回 server 端文件夹路径（失败为 null）。
 *  `xySnapshot.mode` 必须是 'xy'；本函数会派生 per-cell single-snapshot。 */
export async function saveXYMatrix(
  input: ExportInput,
  xySnapshot: GenerateParamsSnapshot,
): Promise<string | null> {
  try {
    const { samples, taskId, xAxis, yAxis, xValues, yValues } = input
    const xLoraIndex = xySnapshot.xy_draft?.x.loraIndex ?? null
    const yLoraIndex = xySnapshot.xy_draft?.y?.loraIndex ?? null

    // 1) 拉所有 cell PNG bytes + 构 per-cell single-snapshot
    type CellEntry = { xi: number; yi: number; blob: Blob; params: GenerateParamsSnapshot }
    const cellEntries: CellEntry[] = []
    for (const s of samples) {
      const fn = s.path.split(/[\\/]/).pop()
      if (!fn) continue
      try {
        const res = await fetch(api.generateSampleUrl(taskId, fn))
        if (!res.ok) continue
        const blob = await res.blob()
        const xv = xValues[s.xy.xi] ?? ''
        const yv = yAxis ? (yValues[s.xy.yi] ?? '') : null
        const cellParams = buildCellSnapshot(xySnapshot, s.xy, {
          x: { axis: xAxis, loraIndex: xLoraIndex, value: xv },
          y: yAxis ? { axis: yAxis, loraIndex: yLoraIndex, value: yv ?? '' } : null,
        })
        cellEntries.push({ xi: s.xy.xi, yi: s.xy.yi, blob, params: cellParams })
      } catch {
        // 单 cell 拉失败跳过；server 会按 manifest 校验，缺 cell 就不发送
      }
    }
    if (cellEntries.length === 0) return null

    // 2) composite 大图（沿用现成 composeXYMatrix）
    const composite = await composeXYMatrix(input)

    // 3) multipart 一次发：composite + N 张 cell + cells_manifest
    const fd = new FormData()
    fd.append('mode', 'xy')
    fd.append('image', composite, 'xy plot.png')
    fd.append('params', JSON.stringify(xySnapshot))
    const manifest = cellEntries.map(({ xi, yi, params }) => ({ xi, yi, params }))
    fd.append('cells_manifest', JSON.stringify(manifest))
    for (const { xi, yi, blob } of cellEntries) {
      fd.append('cells', blob, `cell x${xi} y${yi}.png`)
    }
    const r = await fetch('/api/generate/save', { method: 'POST', body: fd })
    if (!r.ok) return null
    const data = await r.json() as XYSaveResult
    return data.folder
  } catch {
    return null
  }
}
