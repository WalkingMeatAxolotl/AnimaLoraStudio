import { useMemo, useState } from 'react'
import { api, type MonitorState } from '../../../api/client'
import { AXIS_LABELS, formatAxisValue, type XYAxisDraft } from './xy'

type Density = 'compact' | 'standard' | 'large'

/** 紧凑/标准/大图 → 单 cell 最小宽度（px）；多余空间按 1fr 均分到所有列。
 * 用 minmax(MIN, 1fr)，让 cell 同时撑满容器宽度 + 不至于太小。 */
const DENSITY_MIN: Record<Density, number> = {
  compact: 80,
  standard: 160,
  large: 280,
}

/** XY 模式预览网格：按 monitorState.samples[].xy 排成 N×M（CSS grid）。
 *
 * - 不再限制列数（之前 colsForX 截到 5 → 12 张矩阵只显 10）
 * - gap 2px（用户决策）
 * - cell aspect 1:1，object-cover 填满
 * - 列模板 `60px repeat(xLen, minmax(MIN, 1fr))` 让每行所有 cell 同宽
 *   且至少 MIN 宽，多余空间均分；超出容器宽时整个 grid 横滚
 */
export default function PreviewXYGrid({
  samples, taskId, xDraft, yDraft, onCellClick, selectedIndices,
}: {
  samples: NonNullable<MonitorState['samples']>
  taskId: number
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  onCellClick?: (sampleIdx: number) => void
  selectedIndices?: number[]
}) {
  const [density, setDensity] = useState<Density>('standard')

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

  const cellIndex = useMemo(() => {
    const m = new Map<string, number>()
    samples.forEach((s, idx) => {
      if (s.xy) m.set(`${s.xy.yi}_${s.xy.xi}`, idx)
    })
    return m
  }, [samples])

  const minW = DENSITY_MIN[density]
  const selSet = new Set(selectedIndices ?? [])

  // grid 列：左 axis label 列 + N 个图 cell 列。yDraft 不存在时省略 label 列。
  const labelColW = yDraft ? 60 : 0
  const gridCols = yDraft
    ? `${labelColW}px repeat(${xLen}, minmax(${minW}px, 1fr))`
    : `repeat(${xLen}, minmax(${minW}px, 1fr))`

  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0">
      <div className="flex items-center justify-between shrink-0">
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

      {/* grid 自带横向滚动（X 列太多撑爆容器时） */}
      <div className="flex-1 min-h-0 overflow-auto">
        <div style={{ display: 'grid', gridTemplateColumns: gridCols, gap: 2 }}>
          {/* 表头：左上角空白（仅当有 yDraft）+ X 标签 */}
          {yDraft && <div />}
          {xValues.map((xv, xi) => (
            <div
              key={`h-${xi}`}
              className="text-2xs text-fg-tertiary font-mono text-center truncate"
              style={{ padding: '4px 2px' }}
              title={xv}
            >
              {formatAxisValue(xDraft.axis, xv)}
            </div>
          ))}

          {/* 数据行 */}
          {yValues.map((yv, yi) => (
            <Row
              key={`y-${yi}`}
              yi={yi} yv={yv}
              xValues={xValues}
              xDraft={xDraft}
              yDraft={yDraft}
              cellIndex={cellIndex}
              samples={samples}
              taskId={taskId}
              selSet={selSet}
              onCellClick={onCellClick}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function Row({
  yi, yv, xValues, xDraft, yDraft, cellIndex, samples, taskId, selSet, onCellClick,
}: {
  yi: number
  yv: string | null
  xValues: string[]
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  cellIndex: Map<string, number>
  samples: NonNullable<MonitorState['samples']>
  taskId: number
  selSet: Set<number>
  onCellClick?: (sampleIdx: number) => void
}) {
  return (
    <>
      {yDraft && (
        <div
          className="text-2xs text-fg-tertiary font-mono text-right truncate self-center"
          style={{ paddingRight: 4 }}
          title={yv ?? ''}
        >
          {yv != null ? formatAxisValue(yDraft.axis, yv) : ''}
        </div>
      )}
      {xValues.map((xv, xi) => {
        const idx = cellIndex.get(`${yi}_${xi}`)
        const sample = idx != null ? samples[idx] : null
        const filename = sample ? sample.path.split(/[\\/]/).pop() : null
        const isSel = idx != null && selSet.has(idx)
        return (
          <div key={`c-${yi}-${xi}`} style={{ aspectRatio: '1' }}>
            {sample && filename ? (
              <button
                onClick={() => idx != null && onCellClick?.(idx)}
                onDoubleClick={() => filename && window.open(api.generateSampleUrl(taskId, filename), '_blank')}
                className={`block w-full h-full p-0 cursor-pointer overflow-hidden rounded-sm border-2 bg-transparent ${
                  isSel ? 'border-accent' : 'border-transparent hover:border-dim'
                }`}
                title={!yDraft
                  ? `${AXIS_LABELS[xDraft.axis]}=${formatAxisValue(xDraft.axis, xv)}`
                  : `${AXIS_LABELS[xDraft.axis]}=${formatAxisValue(xDraft.axis, xv)} · ${AXIS_LABELS[yDraft.axis]}=${formatAxisValue(yDraft.axis, yv ?? '')}`}
              >
                <img
                  src={api.generateSampleUrl(taskId, filename)}
                  className="w-full h-full object-cover"
                  alt={filename}
                  loading="lazy"
                />
              </button>
            ) : (
              <div className="grid place-items-center w-full h-full rounded-sm border border-subtle bg-sunken text-fg-tertiary text-2xs">
                …
              </div>
            )}
          </div>
        )
      })}
    </>
  )
}
