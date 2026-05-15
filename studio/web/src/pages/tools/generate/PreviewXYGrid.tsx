import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type MonitorState } from '../../../api/client'
import { exportXYMatrix } from './exportXY'
import FullscreenViewer from './FullscreenViewer'
import { AXIS_LABELS, formatAxisValue, type XYAxisDraft } from './xy'

// zoom = 单 cell 物理宽度（px）。固定列宽 → 滚轮 zoom 视觉立即生效；
// 列总宽 > 容器时横滚（已有 overflow:auto 兜底）。
// MIN = ZOOM_DEFAULT (100%)：用户决策不允许小于 100%（cell 太小看不清
// 没意义）；MAX 动态 = 容器宽（保证最大单 cell 占满一屏）；
// DEFAULT = 200px (100%)。
const ZOOM_DEFAULT = 200
const ZOOM_MIN = ZOOM_DEFAULT
const ZOOM_STEP = 24

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
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
  const [cellW, setCellW] = useState(ZOOM_DEFAULT)
  const [maxW, setMaxW] = useState(ZOOM_DEFAULT * 6) // 容器还没 mount 时的兜底值
  const [fullscreenIdx, setFullscreenIdx] = useState<number | null>(null)
  const [exporting, setExporting] = useState(false)
  const [exportMsg, setExportMsg] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const handleExport = async () => {
    if (exporting) return
    setExporting(true)
    setExportMsg(null)
    try {
      const exportSamples = samples
        .filter((s): s is typeof s & { xy: NonNullable<typeof s.xy> } => s.xy != null)
        .map((s) => ({ path: s.path, xy: { xi: s.xy.xi, yi: s.xy.yi } }))
      await exportXYMatrix({
        samples: exportSamples,
        taskId,
        xAxis: xDraft.axis,
        yAxis: yDraft?.axis ?? null,
        xValues,
        yValues,
      })
      setExportMsg('已下载')
      setTimeout(() => setExportMsg(null), 3000)
    } catch (e) {
      setExportMsg(`失败: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setExporting(false)
    }
  }
  // pan 状态。movedRef 让"拖动过"的 mouseup 不触发 cell click（capture 阶段拦截）
  const dragRef = useRef<{ startX: number; startY: number; sX: number; sY: number } | null>(null)
  const movedRef = useRef(false)

  // ZOOM_MAX 动态 = 容器宽（一张图一屏）；ResizeObserver 跟随窗口 / sidebar 变化
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const update = () => {
      const w = Math.max(ZOOM_DEFAULT, el.clientWidth)
      setMaxW(w)
      setCellW((prev) => Math.min(prev, w))
    }
    update()
    // jsdom 没有 ResizeObserver；测试 env 直接降级为 window resize 监听
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(update)
      ro.observe(el)
      return () => ro.disconnect()
    }
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  // wheel 必须 native + passive=false 才能 preventDefault
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      // shift+wheel 让浏览器原生横滚，不 zoom
      if (e.shiftKey) return
      e.preventDefault()
      const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
      setCellW((prev) => clamp(prev + delta, ZOOM_MIN, maxW))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [maxW])

  const onMouseDown: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (e.button !== 0) return
    // 所有区域（含 cell button）都进 pan；普通 click 已改成 Ctrl+click 才选
    // cell，普通点击让位给拖动手势。preventDefault 阻止浏览器原生 img drag。
    e.preventDefault()
    if (!scrollRef.current) return
    dragRef.current = {
      startX: e.clientX, startY: e.clientY,
      sX: scrollRef.current.scrollLeft,
      sY: scrollRef.current.scrollTop,
    }
    movedRef.current = false
  }

  const onMouseMove: React.MouseEventHandler<HTMLDivElement> = (e) => {
    const d = dragRef.current
    if (!d || !scrollRef.current) return
    const dx = e.clientX - d.startX
    const dy = e.clientY - d.startY
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) movedRef.current = true
    scrollRef.current.scrollLeft = d.sX - dx
    scrollRef.current.scrollTop = d.sY - dy
  }

  const onMouseUp = () => {
    dragRef.current = null
  }

  // capture 阶段拦截 click：拖动过的 mouseup 不让 cell button 触发 click
  const onClickCapture: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (movedRef.current) {
      e.stopPropagation()
      e.preventDefault()
      movedRef.current = false
    }
  }

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

  const fullscreenNeighbors = useMemo(() => {
    if (fullscreenIdx == null) return null
    const sample = samples[fullscreenIdx]
    if (!sample?.xy) return null
    const at = (dx: number, dy: number): number | null =>
      cellIndex.get(`${sample.xy!.yi + dy}_${sample.xy!.xi + dx}`) ?? null
    return {
      left: at(-1, 0),
      right: at(1, 0),
      up: at(0, -1),
      down: at(0, 1),
    }
  }, [cellIndex, fullscreenIdx, samples])

  const selSet = new Set(selectedIndices ?? [])

  // grid 列：固定 cellW（zoom 调它），yDraft 时左侧多一列 axis label。
  // 用 ${cellW}px 而非 minmax(MIN, 1fr) —— 后者在容器宽时按 1fr 均分，
  // zoom 就看不出来；固定列宽让滚轮 zoom 视觉立即生效。
  const labelColW = yDraft ? 60 : 0
  const gridCols = yDraft
    ? `${labelColW}px repeat(${xLen}, ${cellW}px)`
    : `repeat(${xLen}, ${cellW}px)`

  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0">
      <div className="flex items-center justify-between shrink-0">
        <span className="caption">
          {xLen}{yDraft ? ` × ${yLen}` : ''} = {xLen * yLen} 张
          {samples.length < xLen * yLen && samples.length > 0 && (
            <span className="text-fg-tertiary"> · 已出 {samples.length}</span>
          )}
        </span>
        <div className="flex items-center gap-2 text-2xs text-fg-tertiary font-mono">
          <span>滚轮缩放 · 拖动平移 · Ctrl+点击选中 · 双击全屏</span>
          <button
            onClick={() => setCellW(ZOOM_DEFAULT)}
            className="btn btn-ghost text-xs"
            title="重置缩放"
          >
            {Math.round((cellW / ZOOM_DEFAULT) * 100)}%
          </button>
          <button
            onClick={() => void handleExport()}
            disabled={exporting || samples.length === 0}
            className="btn btn-secondary text-xs"
            title="把整个 XY 矩阵 + 轴标签合并成一张 PNG 下载"
          >
            {exporting ? '导出中…' : '导出 PNG'}
          </button>
          {exportMsg && (
            <span className={`text-2xs ${exportMsg.startsWith('失败') ? 'text-err' : 'text-ok'}`}>
              {exportMsg}
            </span>
          )}
        </div>
      </div>

      {/* grid 自带横向滚动（X 列太多撑爆容器时）+ 滚轮 zoom + 拖动 pan */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-auto"
        style={{ cursor: dragRef.current ? 'grabbing' : 'grab' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onClickCapture={onClickCapture}
      >
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
              onCellDoubleClick={(idx) => setFullscreenIdx(idx)}
            />
          ))}
        </div>
      </div>

      {fullscreenIdx != null && samples[fullscreenIdx] && (() => {
        const s = samples[fullscreenIdx]
        const fn = s.path.split(/[\\/]/).pop() ?? ''
        const captionParts: string[] = []
        if (s.xy) {
          captionParts.push(`${AXIS_LABELS[xDraft.axis]}=${formatAxisValue(xDraft.axis, String(s.xy.xv ?? ''))}`)
          if (yDraft && s.xy.yv != null) {
            captionParts.push(`${AXIS_LABELS[yDraft.axis]}=${formatAxisValue(yDraft.axis, String(s.xy.yv))}`)
          }
        }
        return (
          <FullscreenViewer
            src={api.generateSampleUrl(taskId, fn)}
            alt={fn}
            caption={captionParts.join(' · ')}
            onClose={() => setFullscreenIdx(null)}
            hasPrev={fullscreenNeighbors?.left != null}
            hasNext={fullscreenNeighbors?.right != null}
            hasUp={fullscreenNeighbors?.up != null}
            hasDown={fullscreenNeighbors?.down != null}
            onPrev={() => {
              if (fullscreenNeighbors?.left != null) setFullscreenIdx(fullscreenNeighbors.left)
            }}
            onNext={() => {
              if (fullscreenNeighbors?.right != null) setFullscreenIdx(fullscreenNeighbors.right)
            }}
            onUp={() => {
              if (fullscreenNeighbors?.up != null) setFullscreenIdx(fullscreenNeighbors.up)
            }}
            onDown={() => {
              if (fullscreenNeighbors?.down != null) setFullscreenIdx(fullscreenNeighbors.down)
            }}
            shortcutHint="←/→/↑/↓ 切换单元格 · ESC / 点击遮罩关闭"
          />
        )
      })()}
    </div>
  )
}

function Row({
  yi, yv, xValues, xDraft, yDraft, cellIndex, samples, taskId, selSet,
  onCellClick, onCellDoubleClick,
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
  onCellDoubleClick?: (sampleIdx: number) => void
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
        const filename = sample ? sample.path.split(/[\\/]/).pop() ?? null : null
        const isSel = idx != null && selSet.has(idx)
        const tooltip = (!yDraft
          ? `${AXIS_LABELS[xDraft.axis]}=${formatAxisValue(xDraft.axis, xv)}`
          : `${AXIS_LABELS[xDraft.axis]}=${formatAxisValue(xDraft.axis, xv)} · ${AXIS_LABELS[yDraft.axis]}=${formatAxisValue(yDraft.axis, yv ?? '')}`) + ' · 双击全屏 · Ctrl+点击选中'
        return (
          <GridCell
            key={`c-${yi}-${xi}`}
            taskId={taskId}
            filename={filename}
            sampleIdx={idx ?? null}
            isSelected={isSel}
            tooltip={tooltip}
            onClick={onCellClick}
            onDoubleClick={onCellDoubleClick}
          />
        )
      })}
    </>
  )
}

function GridCell({
  taskId, filename, sampleIdx, isSelected, tooltip, onClick, onDoubleClick,
}: {
  taskId: number
  filename: string | null
  sampleIdx: number | null
  isSelected: boolean
  tooltip: string
  onClick?: (idx: number) => void
  onDoubleClick?: (idx: number) => void
}) {
  const [errored, setErrored] = useState(false)

  // 切 task / filename 变（如点击历史回看）时 reset errored，让 img
  // 重新尝试加载。否则上次 errored=true 残留，新 src 来了仍显示 "..."
  useEffect(() => {
    setErrored(false)
  }, [taskId, filename])

  // 占位（无 sample / cache miss）：minHeight 撑高让 grid 行不塌缩
  if (!filename || errored) {
    return (
      <div
        className="grid place-items-center rounded-sm border border-subtle bg-sunken text-fg-tertiary text-2xs"
        style={{ minHeight: 80 }}
      >
        {errored ? '原图已释放' : '…'}
      </div>
    )
  }
  return (
    <button
      onClick={(e) => {
        // 普通 click 让位给 pan（拖动场景）；Ctrl/Cmd+click 才选 cell
        if (sampleIdx == null) return
        if (e.ctrlKey || e.metaKey) onClick?.(sampleIdx)
      }}
      onDoubleClick={() => sampleIdx != null && onDoubleClick?.(sampleIdx)}
      className={`block p-0 overflow-hidden rounded-sm border-2 bg-sunken ${
        isSelected ? 'border-accent' : 'border-transparent hover:border-dim'
      }`}
      title={tooltip}
      style={{ minHeight: 80 }}
    >
      {/* key 加 taskId+filename 让 src 变化时 React 强制重挂载 img，避免
          上次失败的浏览器缓存或 onError 状态残留 */}
      <img
        key={`${taskId}-${filename}`}
        src={api.generateSampleUrl(taskId, filename)}
        className="block w-full h-auto pointer-events-none"
        alt={filename}
        loading="lazy"
        draggable={false}
        onError={() => setErrored(true)}
        onLoad={() => setErrored(false)}
      />
    </button>
  )
}
