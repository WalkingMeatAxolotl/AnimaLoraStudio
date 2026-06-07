/** 测试出图自动落盘（Settings → testing → save_test_images 开启时）。
 *
 * 路径：studio_data/test/<YYYY-MM-DD>/{single,xy}/image_N.png
 * - single：每张 sample 单独上传一份（image_N 逐张递增）
 * - xy：用 composeXYMatrix 把整张网格合成单图后上传一次
 * - compare：调用方负责跳过；本模块不处理
 *
 * 失败 / 后端 403（开关被关）都静默吞掉 —— 不打扰用户主流程。
 */
import { api } from '../../../api/client'
import { composeXYMatrix, type ExportInput } from './exportXY'

async function postSave(mode: 'single' | 'xy', blob: Blob): Promise<void> {
  const fd = new FormData()
  fd.append('mode', mode)
  fd.append('image', blob, `${mode}.png`)
  await fetch('/api/generate/save', { method: 'POST', body: fd })
}

export async function saveSingleSamples(taskId: number, filenames: string[]): Promise<void> {
  for (const fn of filenames) {
    try {
      const res = await fetch(api.generateSampleUrl(taskId, fn))
      if (!res.ok) continue
      const blob = await res.blob()
      await postSave('single', blob)
    } catch {
      // 单张失败不阻塞剩下的
    }
  }
}

export async function saveXYMatrix(input: ExportInput): Promise<void> {
  try {
    const blob = await composeXYMatrix(input)
    await postSave('xy', blob)
  } catch {
    // 合成 / 上传失败静默
  }
}
