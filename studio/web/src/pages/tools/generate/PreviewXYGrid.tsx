import { useMemo, useState } from 'react'
import { api, type MonitorState } from '../../../api/client'
import { AXIS_LABELS } from './xy'
import type { XYAxisDraft } from './xy'

type Density = 'compact' | 'standard' | 'large'

const DENSITY_THUMB: Record<Density, number> = {
  compact: 96,
  standard: 160,
  large: 240,
}

/** 计算每行最大列数 —— 基于 X values 数量；超过 5 自动滚动 */
function colsForX(xLen: number): number {
  return Math.min(Math.max(xLen, 1), 5)
}

/** XY 模式预览网格：按 monitorState.samples[].xy 排成 N×M。
 *
 * 缺图占位灰格（task 还在跑中或某 cell 失败）。点 cell 触发选中（compare
 * 模式 commit 6 用），双击放大新窗口看原图。 */
export default function PreviewXYGrid({
  samples, taskId, xDraft, yDraft, onCellClick, selectedIndices,
}: {
  samples: NonNullable<MonitorState['samples']>
  taskId: number
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  onCellClick?: (sampleIdx: number) => void
  /** 高亮已选中的 cell（compare 模式候选） */
  selectedIndices?: number[]
}) {
  const [density, setDensity] = useState<Density>('standard')

  // 把 samples 索引到 (xi, yi) 二维表
  const xValues = useMemo(
    () => xDraft.raw.split(',').map((s) => s.trim()).filter(Boolean),
    [xDraft.raw],
  )
  const yValues = useMemo(
    () => yDraft ? yDraft.raw.split(',').map((s) => s.trim()).filter(Boolean) : [null],
    [yDraft],
  )
  const xLen = xValues.length
  const yLen = yValues.length

  // (yi, xi) → sample 在 samples[] 里的索引（用于 selectedIndices 高亮 + click）
  const cellIndex = useMemo(() => {
    const m = new Map<string, number>()
    samples.forEach((s, idx) => {
      if (s.xy) m.set(`${s.xy.yi}_${s.xy.xi}`, idx)
    })
    return m
  }, [samples])

  const thumb = DENSITY_THUMB[density]
  const cols = colsForX(xLen)
  const selSet = new Set(selectedIndices ?? [])

  return (
    <div className="flex flex-col gap-3">
      {/* 工具条：列宽切换 + 计数 */}
      <div className="flex items-center justify-between">
        <span className="caption">
          {xLen}{yDraft ? ` × ${yLen}` : ''} = {xLen * yLen} 张
          {samples.length < xLen * yLen && samples.length > 0 && (
            <span className="text-fg-tertiary"> · 已出 {samples.length}</span>
          )}
        </span>
        <div className="flex items-center gap-1" role="tablist" aria-label="网格密度">
          {(['compact', 'standard', 'large'] as Density[]).map((d) => (
            <button
              key={d}
              onClick={() => setDensity(d)}
              className={`btn btn-sm text-xs ${density === d ? 'btn-primary' : 'btn-ghost text-fg-secondary'}`}
            >
              {d === 'compact' ? '紧凑' : d === 'standard' ? '标准' : '大图'}
            </button>
          ))}
        </div>
      </div>

      {/* 网格本体 */}
      <div className="overflow-x-auto">
        <table className="border-collapse" style={{ tableLayout: 'fixed' }}>
          {yDraft && (
            <thead>
              <tr>
                <th className="text-2xs text-fg-tertiary p-1">
                  {AXIS_LABELS[yDraft.axis]} \ {AXIS_LABELS[xDraft.axis]}
                </th>
                {xValues.slice(0, cols).map((xv, xi) => (
                  <th key={xi} className="text-2xs text-fg-tertiary font-mono p-1 text-center" style={{ width: thumb }}>
                    {xv}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {yValues.map((yv, yi) => (
              <tr key={yi}>
                {yDraft && (
                  <td className="text-2xs text-fg-tertiary font-mono p-1 text-right" style={{ width: 60 }}>
                    {yv ?? ''}
                  </td>
                )}
                {xValues.slice(0, cols).map((xv, xi) => {
                  const idx = cellIndex.get(`${yi}_${xi}`)
                  const sample = idx != null ? samples[idx] : null
                  const filename = sample ? sample.path.split(/[\\/]/).pop() : null
                  const isSel = idx != null && selSet.has(idx)
                  return (
                    <td key={xi} className="p-1 align-top">
                      {sample && filename ? (
                        <button
                          onClick={() => idx != null && onCellClick?.(idx)}
                          onDoubleClick={() => filename && window.open(api.generateSampleUrl(taskId, filename), '_blank')}
                          className={`block p-0 cursor-pointer overflow-hidden rounded border-2 bg-transparent ${
                            isSel ? 'border-accent' : 'border-transparent hover:border-dim'
                          }`}
                          style={{ width: thumb, height: thumb }}
                          title={!yDraft
                            ? `${AXIS_LABELS[xDraft.axis]}=${xv}`
                            : `${AXIS_LABELS[xDraft.axis]}=${xv} · ${AXIS_LABELS[yDraft.axis]}=${yv}`}
                        >
                          <img
                            src={api.generateSampleUrl(taskId, filename)}
                            className="w-full h-full object-cover"
                            alt={filename}
                            loading="lazy"
                          />
                        </button>
                      ) : (
                        <div
                          className="grid place-items-center rounded border border-subtle bg-sunken text-fg-tertiary text-2xs"
                          style={{ width: thumb, height: thumb }}
                        >
                          …
                        </div>
                      )}
                      {!yDraft && (
                        <div className="text-2xs text-fg-tertiary text-center font-mono mt-0.5">{xv}</div>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
