/** 测试出图自动落盘（Settings → testing → save_test_images 开启时）。
 *
 * 路径：studio_data/test/<YYYY-MM-DD>/{single,xy}/image_N.png
 * - single：每张 sample 单独上传一份（image_N 逐张递增）
 * - xy：用 composeXYMatrix 把整张网格合成单图后上传一次
 * - compare：调用方负责跳过；本模块不处理
 *
 * 上传时附 params JSON（GenerateParamsSnapshot）：后端写进 PNG `anima_params`
 * tEXt + 同目录 image_N.json sidecar，供历史栏回看 / 用户拷走 PNG 复用参数。
 *
 * 失败 / 后端 403（开关被关）都静默吞掉 —— 不打扰用户主流程。
 */
import { api } from '../../../api/client'
import { composeXYMatrix, type ExportInput } from './exportXY'
import type { GenerateParamsSnapshot } from './paramsSnapshot'

interface SaveResult {
  /** 落盘路径（用于回写 IndexedDB entry.diskPath，做 disk-history 去重） */
  path: string
  index: number
}

async function postSave(
  mode: 'single' | 'xy',
  blob: Blob,
  params: GenerateParamsSnapshot,
): Promise<SaveResult | null> {
  const fd = new FormData()
  fd.append('mode', mode)
  fd.append('image', blob, `${mode}.png`)
  fd.append('params', JSON.stringify(params))
  const r = await fetch('/api/generate/save', { method: 'POST', body: fd })
  if (!r.ok) return null
  return (await r.json()) as SaveResult
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
      const r = await postSave('single', blob, params)
      paths.push(r?.path ?? null)
    } catch {
      paths.push(null)
    }
  }
  return paths
}

/** 落盘 xy 合成大图。返回落盘 path（失败为 null）。 */
export async function saveXYMatrix(
  input: ExportInput,
  params: GenerateParamsSnapshot,
): Promise<string | null> {
  try {
    const blob = await composeXYMatrix(input)
    const r = await postSave('xy', blob, params)
    return r?.path ?? null
  } catch {
    return null
  }
}
