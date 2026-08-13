/** timeline entry adapter —— server 行 → HistoryEntry 的派生逻辑。 */
import { describe, expect, it } from 'vitest'
import type { GenerateTimelineEntry } from '../../../api/client'
import {
  adaptTimelineEntry,
  entryBadge,
  entryDisplayLabel,
  entryImageUrl,
  entryTaskId,
  entryThumbUrl,
} from './entryAdapter'

function serverEntry(over: Partial<GenerateTimelineEntry> = {}): GenerateTimelineEntry {
  return {
    task_id: 42,
    status: 'done',
    created_at: 1754800000,
    mode: 'single',
    storage: 'disk',
    params: { mode: 'single' },
    images: [{
      url: '/api/generate/disk/image/2026-08-10/single/single%20image%205.png',
      thumb_url: '/api/generate/disk/thumb/2026-08-10/single/single%20image%205.png?w=128',
    }],
    available: true,
    ...over,
  }
}

describe('adaptTimelineEntry', () => {
  it('基本字段：taskId / created_at 秒→ms / released=!available', () => {
    const e = adaptTimelineEntry(serverEntry())
    expect(entryTaskId(e)).toBe(42)
    expect(e.id).toBe('t42')
    expect(e.createdAt).toBe(1754800000_000)
    expect(e.released).toBe(false)
    expect(entryImageUrl(e)).toContain('/disk/image/')
    expect(entryThumbUrl(e)).toContain('w=128')
  })

  it('available=false → released，thumb 回退空串', () => {
    const e = adaptTimelineEntry(serverEntry({ images: [], available: false }))
    expect(e.released).toBe(true)
    expect(entryThumbUrl(e)).toBe('')
    expect(entryImageUrl(e)).toBe('')
  })

  it('temp 行：sample URL + 显示标签回退 #taskId', () => {
    const e = adaptTimelineEntry(serverEntry({
      storage: 'temp',
      images: [{ url: '/api/generate/42/sample/img_p0_0.png' }],
    }))
    expect(entryThumbUrl(e)).toBe('/api/generate/42/sample/img_p0_0.png')
    expect(entryDisplayLabel(e)).toBe('#42')
  })

  it('disk single 显示标签从 URL 反解文件名', () => {
    const e = adaptTimelineEntry(serverEntry())
    expect(entryDisplayLabel(e)).toBe('single image 5')
  })

  it('xy 行：从 images (xi,yi) + params.xy_draft 重建 xyMeta，按 yi,xi 排序', () => {
    const e = adaptTimelineEntry(serverEntry({
      mode: 'xy',
      xy_folder: 'xy plot 3',
      params: {
        mode: 'xy',
        xy_draft: {
          x: { axis: 'steps', raw: '20, 25', loraIndex: null },
          y: { axis: 'cfg_scale', raw: '3, 4', loraIndex: null },
        },
      },
      images: [
        { url: '/img/c11', xi: 1, yi: 1 },
        { url: '/img/c00', xi: 0, yi: 0 },
        { url: '/img/c10', xi: 1, yi: 0 },
        { url: '/img/c01', xi: 0, yi: 1 },
      ],
    }))
    expect(e.xyMeta).toBeDefined()
    expect(e.xyMeta!.xValues).toEqual(['20', '25'])
    expect(e.xyMeta!.yValues).toEqual(['3', '4'])
    expect(e.xyMeta!.samples.map((s) => [s.xy.xi, s.xy.yi])).toEqual(
      [[0, 0], [1, 0], [0, 1], [1, 1]],
    )
    expect(e.xyMeta!.samples[1].xy.xv).toBe('25')
    expect(e.xyMeta!.samples[2].xy.yv).toBe('4')
    expect(entryDisplayLabel(e)).toBe('xy plot 3')
    expect(entryBadge(e)).toBe('XY 2×2')
  })

  it('xy 行缺 params（老行）：xyMeta 仍从 images 重建，badge 从 xyMeta 推', () => {
    const e = adaptTimelineEntry(serverEntry({
      mode: 'xy',
      params: null,
      images: [
        { url: '/img/c00', xi: 0, yi: 0 },
        { url: '/img/c10', xi: 1, yi: 0 },
      ],
    }))
    expect(e.params).toBeUndefined()
    expect(e.xyMeta!.samples).toHaveLength(2)
    expect(entryBadge(e)).toBe('XY 2×1')
  })
})
