/** 像素面积分桶 —— 映射到常见 LoRA 训练分辨率（共享于 Preprocess 页 +
 *  Overview 详情 tab）。bin 边界按总像素数（w*h）划分，跟训练分辨率匹配。 */

export const PX_BINS = [
  { id: 'lt-512',    label: '< 512²',         lo: 0,           hi: 512 * 512,   sortKey: 0 },
  { id: '512-768',   label: '512² – 768²',    lo: 512 * 512,   hi: 768 * 768,   sortKey: 512 * 512 },
  { id: '768-1024',  label: '768² – 1024²',   lo: 768 * 768,   hi: 1024 * 1024, sortKey: 768 * 768 },
  { id: '1024-1536', label: '1024² – 1536²',  lo: 1024 * 1024, hi: 1536 * 1536, sortKey: 1024 * 1024 },
  { id: '1536-2048', label: '1536² – 2048²',  lo: 1536 * 1536, hi: 2048 * 2048, sortKey: 1536 * 1536 },
  { id: 'gt-2048',   label: '> 2048²',        lo: 2048 * 2048, hi: Infinity,    sortKey: 2048 * 2048 },
] as const

export type PxBinId = (typeof PX_BINS)[number]['id']

export function pxBinFor(w: number | null, h: number | null): PxBinId | null {
  if (w == null || h == null) return null
  const area = w * h
  const bin = PX_BINS.find((b) => area >= b.lo && area < b.hi)
  return bin?.id ?? PX_BINS[PX_BINS.length - 1].id
}

export interface PixelHistBin {
  id: PxBinId
  label: string
  n: number
}

/** 给定一组 (w, h)，返回非空 bin 的 histogram 数据（适配 BarHistogram）。 */
export function computePixelHist(items: ReadonlyArray<{ w: number | null; h: number | null }>): PixelHistBin[] {
  const counts = new Map<PxBinId, number>(PX_BINS.map((b) => [b.id, 0]))
  for (const it of items) {
    const id = pxBinFor(it.w, it.h)
    if (id) counts.set(id, (counts.get(id) ?? 0) + 1)
  }
  return PX_BINS
    .map((b) => ({ id: b.id, label: b.label, n: counts.get(b.id) ?? 0 }))
    .filter((b) => b.n > 0)
}
