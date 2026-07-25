/** XY 矩阵导出：所有 cell 图 + X/Y 轴标签合并成一张 PNG。
 *
 * 用 canvas 直接合成：
 *   1. 把所有 sample 的图 fetch 进 HTMLImageElement
 *   2. 第一张图决定 cell 尺寸（同 task 所有图比例同）
 *   3. canvas 尺寸 = padding + labelW + xLen × cellW × （labelH + yLen × cellH）
 *   4. 画 X / Y 轴标签 + 每个 cell 图
 *   5. toBlob → 下载
 *
 * 同源 API 路径，drawImage 不需要 crossOrigin/CORS。
 */
import { api } from '../../../api/client'
import i18n from '../../../i18n'

interface ExportSample {
  path: string
  xy: { xi: number; yi: number }
  /** 非 Generate cache 来源时由调用方给完整 URL（如 eval 的出图） */
  imageUrl?: string
}

export interface ExportInput {
  samples: ExportSample[]
  taskId: number
  /** 轴标签：**已格式化**的显示文本，直接画到 canvas 上。
   *
   *  以前这里收 XYAxisType 再自己 formatAxisValue —— 那把导出绑死在 Generate 的
   *  轴类型枚举上。改收成品文本后 eval 的 checkpoint × prompt 矩阵也能用同一个导出。 */
  xLabels: string[]
  /** null = 单轴（不画 Y 标签列） */
  yLabels: Array<string | null> | null
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`failed to load ${url}`))
    img.src = url
  })
}

/** 把 XY 矩阵合成成一张 PNG Blob —— exportXYMatrix 下载 + saveTestImages 自动落盘共用。 */
export async function composeXYMatrix(input: ExportInput): Promise<Blob> {
  const { samples, taskId, xLabels, yLabels } = input
  const yRows: Array<string | null> = yLabels ?? [null]
  const xLen = xLabels.length
  const yLen = Math.max(yRows.length, 1)
  if (xLen === 0 || samples.length === 0) {
    throw new Error(i18n.t('generate.noExportableXyData'))
  }

  // 加载所有 cell 图
  const imgsByPos = new Map<string, HTMLImageElement>()
  await Promise.all(samples.map(async (s) => {
    const fn = s.path.split(/[\\/]/).pop() ?? ''
    if (!fn) return
    try {
      const img = await loadImage(api.generateSampleUrl(taskId, fn))
      imgsByPos.set(`${s.xy.yi}_${s.xy.xi}`, img)
    } catch { /* skip 失败的 cell */ }
  }))
  if (imgsByPos.size === 0) {
    throw new Error(i18n.t('generate.allCellsLoadFailed'))
  }

  // 第一张图决定 cell 尺寸（保留原始分辨率）
  const first = imgsByPos.values().next().value as HTMLImageElement
  const cellW = first.naturalWidth
  const cellH = first.naturalHeight

  // 文字 / padding 尺寸（按 cellW 自适应：大图大字）
  const fontSize = Math.max(18, Math.round(cellW / 28))
  const labelH = Math.round(fontSize * 2.6)
  const labelW = yLabels ? Math.round(fontSize * 7) : 0
  const padding = Math.round(fontSize * 1.2)
  const totalW = padding * 2 + labelW + xLen * cellW
  const totalH = padding * 2 + labelH + yLen * cellH

  // 渲染
  const canvas = document.createElement('canvas')
  canvas.width = totalW
  canvas.height = totalH
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error(i18n.t('generate.canvas2dUnavailable'))

  ctx.fillStyle = '#15140f'  // 同 design tokens 的 bg-canvas dark
  ctx.fillRect(0, 0, totalW, totalH)
  ctx.fillStyle = '#f0eee5'
  ctx.font = `${fontSize}px JetBrains Mono, ui-monospace, Menlo, monospace`
  ctx.textBaseline = 'middle'

  // X 轴标签（顶部）
  ctx.textAlign = 'center'
  for (let xi = 0; xi < xLen; xi++) {
    const x = padding + labelW + xi * cellW + cellW / 2
    const y = padding + labelH / 2
    const txt = xLabels[xi] ?? ''
    ctx.fillText(txt, x, y, cellW - 8)
  }

  // Y 轴标签（左侧）
  if (yLabels) {
    ctx.textAlign = 'right'
    for (let yi = 0; yi < yLen; yi++) {
      const yv = yRows[yi]
      if (yv == null) continue
      const x = padding + labelW - 8
      const y = padding + labelH + yi * cellH + cellH / 2
      ctx.fillText(yv ?? '', x, y, labelW - 16)
    }
  }

  // cells
  for (let yi = 0; yi < yLen; yi++) {
    for (let xi = 0; xi < xLen; xi++) {
      const img = imgsByPos.get(`${yi}_${xi}`)
      if (!img) continue
      ctx.drawImage(
        img,
        padding + labelW + xi * cellW,
        padding + labelH + yi * cellH,
        cellW, cellH,
      )
    }
  }

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => {
      if (!b) { reject(new Error(i18n.t('generate.canvasToBlobNull'))); return }
      resolve(b)
    }, 'image/png')
  })
}

export async function exportXYMatrix(input: ExportInput): Promise<void> {
  const blob = await composeXYMatrix(input)
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `xy_matrix_${input.taskId}_${Date.now()}.png`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
