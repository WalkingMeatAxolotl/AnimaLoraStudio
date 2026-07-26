// 评估出图的 lora × prompt 矩阵。
//
// 评估第一步本来就为每个候选 × 每张验证图出了一张图，之前只喂给指标计算、用户看不到。
// 这里把它们排成 XY 网格 —— 复用测试页那个组件（滚轮 zoom / 拖动 pan / 双击全屏方向键
// 导航 / 导出合图 PNG 全都白拿），让用户顺手肉眼比一遍，省掉去测试页重跑一次 XY。
// 选 checkpoint 的主路径本来就是视觉对比，指标只是辅助信号。
//
// baseline 是 lora 里的一项（纯底模对照，测试页的 XY 没有这一列），默认勾上。
//
// prompt / lora 都是**逐项勾选**而不是取范围：200 个 checkpoint × N 张验证图会是几千个
// cell，默认勾最近 20 个 lora 和前 3 个 prompt，其余在下拉里随手加减。
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type EvalSampleGrid as GridData } from '../api/client'
import CheckboxDropdown from './CheckboxDropdown'
import PreviewXYGrid, { type XYSample } from '../pages/tools/generate/PreviewXYGrid'
import type { XYAxisView } from '../pages/tools/generate/xy'

const DEFAULT_PROMPT_COUNT = 3
const DEFAULT_CKPT_COUNT = 20

export default function EvalSampleGrid({
  pid, vid, sessionId,
}: {
  pid: number
  vid: number
  sessionId: number
}) {
  const [grid, setGrid] = useState<GridData | null>(null)
  const [error, setError] = useState<string | null>(null)
  // 勾选集。null = 还没按数据初始化过（拿到 grid 后勾默认那批）
  const [pickedRows, setPickedRows] = useState<Set<string> | null>(null)
  const [pickedCols, setPickedCols] = useState<Set<string> | null>(null)

  const load = useCallback(async () => {
    try {
      setGrid(await api.getEvalSessionGrid(pid, vid, sessionId))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [pid, vid, sessionId])

  useEffect(() => { void load() }, [load])
  // 换 Session 时重置勾选（行列都变了）
  useEffect(() => { setPickedRows(null); setPickedCols(null); setGrid(null) }, [sessionId])

  // 默认：前 N 个 prompt；baseline + 最近 N 个 lora（「最近」= 列表末尾，后端按
  // ordinal 给，ordinal 大的训练更晚）
  useEffect(() => {
    if (!grid) return
    if (pickedRows == null) {
      setPickedRows(new Set(
        grid.rows.slice(0, DEFAULT_PROMPT_COUNT).map((r) => String(r.index)),
      ))
    }
    if (pickedCols == null) {
      const base = grid.columns.filter((c) => c.role === 'baseline')
      const ckpts = grid.columns.filter((c) => c.role !== 'baseline')
      setPickedCols(new Set(
        [...base, ...ckpts.slice(-DEFAULT_CKPT_COUNT)].map((c) => String(c.candidate_id)),
      ))
    }
  }, [grid, pickedRows, pickedCols])

  const columns = useMemo(
    () => (grid && pickedCols
      ? grid.columns.filter((c) => pickedCols.has(String(c.candidate_id)))
      : []),
    [grid, pickedCols],
  )
  const rows = useMemo(
    () => (grid && pickedRows
      ? grid.rows.filter((r) => pickedRows.has(String(r.index)))
      : []),
    [grid, pickedRows],
  )

  const xAxis: XYAxisView = useMemo(
    () => ({ label: 'lora', values: columns.map((c) => c.label || `#${c.candidate_id}`) }),
    [columns],
  )
  const yAxis: XYAxisView = useMemo(
    () => ({
      label: 'prompt',
      values: rows.map((r) => r.prompt || r.image || `#${r.index}`),
      // 网格的行标签只有 60px 宽，prompt 常是一长串 booru tag —— 截短，完整内容
      // 靠 cell tooltip / 全屏 caption 看
      format: (v) => (v.length > 28 ? `${v.slice(0, 27)}…` : v),
    }),
    [rows],
  )

  /** grid.cells → PreviewXYGrid 的 samples。图走 session 作用域的 URL（不在
   *  generate cache 里，所以填 imageUrl 让组件优先用它）。 */
  const samples = useMemo<XYSample[]>(() => {
    if (!grid) return []
    const out: XYSample[] = []
    rows.forEach((row, yi) => {
      columns.forEach((col, xi) => {
        const cell = grid.cells[`${col.candidate_id}:${row.index}`]
        if (!cell?.filename) return
        out.push({
          path: cell.filename,
          xy: { xi, yi, xv: xAxis.values[xi], yv: yAxis.values[yi] },
          imageUrl: api.evalSampleImageUrl(pid, vid, sessionId, cell.run_id, cell.filename),
        })
      })
    })
    return out
  }, [grid, rows, columns, xAxis.values, yAxis.values, pid, vid, sessionId])

  const promptOptions = useMemo(
    () => (grid?.rows ?? []).map((r) => ({
      value: String(r.index),
      label: r.prompt || r.image || `#${r.index}`,
      title: r.prompt || r.image || '',
    })),
    [grid],
  )
  const loraOptions = useMemo(
    () => (grid?.columns ?? []).map((c) => ({
      value: String(c.candidate_id),
      label: c.label || `#${c.candidate_id}`,
      title: c.checkpoint_path || c.label || '',
    })),
    [grid],
  )

  if (error) {
    return (
      <div className="rounded-md border border-err bg-err-soft px-3 py-2 text-sm text-err">
        样图矩阵读取失败：{error}
      </div>
    )
  }
  if (!grid) {
    return (
      <div className="rounded-md border border-dashed border-subtle px-3 py-3 text-sm text-fg-tertiary">
        读取样图矩阵…
      </div>
    )
  }

  return (
    <div className="card p-4 flex flex-col gap-3 flex-1 min-h-0">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="text-sm font-semibold">样图</div>
        <span className="text-xs text-fg-tertiary">
          评估出的图按 lora × prompt 排成矩阵，可直接肉眼比
        </span>
        <span className="flex-1" />
        <CheckboxDropdown
          label="prompt"
          options={promptOptions}
          selected={pickedRows ?? new Set()}
          onChange={setPickedRows}
          emptyHint="这次评估没有验证图"
        />
        <CheckboxDropdown
          label="lora"
          options={loraOptions}
          selected={pickedCols ?? new Set()}
          onChange={setPickedCols}
          emptyHint="这次评估没有候选"
        />
      </div>

      {samples.length === 0 ? (
        <div className="rounded-md border border-dashed border-subtle px-3 py-3 text-sm text-fg-tertiary">
          {rows.length === 0
            ? '勾选至少一个 prompt 才能显示矩阵。'
            : columns.length === 0
              ? '勾选至少一个 lora 才能显示矩阵。'
              : '这次评估还没有出图（出图阶段可能仍在跑，或已失败）。'}
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex" style={{ minHeight: 420 }}>
          <PreviewXYGrid
            samples={samples}
            taskId={-1 /* 图走 imageUrl，不会回退到 generate cache */}
            xAxis={xAxis}
            yAxis={yAxis}
          />
        </div>
      )}
    </div>
  )
}
