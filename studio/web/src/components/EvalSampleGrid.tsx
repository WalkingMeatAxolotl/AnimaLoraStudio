// 评估出图的 checkpoint × prompt 矩阵。
//
// 评估第一步本来就为每个候选 × 每张验证图出了一张图，之前只喂给指标计算、用户看不到。
// 这里把它们排成 XY 网格 —— 复用测试页那个组件（滚轮 zoom / 拖动 pan / 双击全屏方向键
// 导航 / 导出合图 PNG 全都白拿），让用户顺手肉眼比一遍，省掉去测试页重跑一次 XY。
// 选 checkpoint 的主路径本来就是视觉对比，指标只是辅助信号。
//
// baseline 在第一列，是 eval 独有的对照：测试页的 XY 没有「纯底模」这一列。
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type EvalSampleGrid as GridData } from '../api/client'
import PreviewXYGrid, { type XYSample } from '../pages/tools/generate/PreviewXYGrid'
import type { XYAxisView } from '../pages/tools/generate/xy'

// 默认只显示一部分 —— 200 个 checkpoint × N 张验证图会是几千个 cell。
const DEFAULT_PROMPT_COUNT = 3
const DEFAULT_CKPT_COUNT = 20
const CKPT_COUNT_OPTIONS = [10, 20, 50, 100, 0] as const  // 0 = 全部

export default function EvalSampleGrid({
  pid, vid, sessionId,
}: {
  pid: number
  vid: number
  sessionId: number
}) {
  const [grid, setGrid] = useState<GridData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  // 显示范围。null = 还没按数据初始化过（拿到 grid 后取默认前 N 个）
  const [pickedRows, setPickedRows] = useState<Set<number> | null>(null)
  const [ckptLimit, setCkptLimit] = useState<number>(DEFAULT_CKPT_COUNT)

  const load = useCallback(async () => {
    if (!open) return
    try {
      setGrid(await api.getEvalSessionGrid(pid, vid, sessionId))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [open, pid, vid, sessionId])

  useEffect(() => { void load() }, [load])
  // 换 Session 时重置范围选择（行数 / checkpoint 都变了）
  useEffect(() => { setPickedRows(null); setGrid(null) }, [sessionId])

  // 默认勾前 N 个 prompt
  useEffect(() => {
    if (!grid || pickedRows != null) return
    setPickedRows(new Set(grid.rows.slice(0, DEFAULT_PROMPT_COUNT).map((r) => r.index)))
  }, [grid, pickedRows])

  const toggleRow = (index: number) => {
    setPickedRows((prev) => {
      const next = new Set(prev ?? [])
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  /** 列：baseline 恒在（对照基准），checkpoint 取**最近** ckptLimit 个。
   *  「最近」= 列表末尾 —— 后端按 ordinal 给，ordinal 大的是训练更晚的。 */
  const columns = useMemo(() => {
    if (!grid) return []
    const base = grid.columns.filter((c) => c.role === 'baseline')
    const ckpts = grid.columns.filter((c) => c.role !== 'baseline')
    const picked = ckptLimit > 0 ? ckpts.slice(-ckptLimit) : ckpts
    return [...base, ...picked]
  }, [grid, ckptLimit])

  const rows = useMemo(
    () => (grid && pickedRows ? grid.rows.filter((r) => pickedRows.has(r.index)) : []),
    [grid, pickedRows],
  )

  const xAxis: XYAxisView = useMemo(
    () => ({ label: 'checkpoint', values: columns.map((c) => c.label || `#${c.candidate_id}`) }),
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

  const totalCkpts = grid ? grid.columns.filter((c) => c.role !== 'baseline').length : 0

  return (
    <div className="card p-4 flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="text-sm font-semibold flex items-center gap-1.5 bg-transparent border-none cursor-pointer text-fg-primary p-0"
        >
          <span className="text-fg-tertiary text-xs">{open ? '▾' : '▸'}</span>
          样图对比
        </button>
        <span className="text-xs text-fg-tertiary">
          评估出的图按 checkpoint × prompt 排成矩阵，可直接肉眼比
        </span>
      </div>

      {open && error && (
        <div className="rounded-md border border-err bg-err-soft px-3 py-2 text-sm text-err">
          样图矩阵读取失败：{error}
        </div>
      )}

      {open && grid && !error && (
        <>
          {/* 显示范围：prompt 多选 + checkpoint 数量 */}
          <div className="rounded-md border border-subtle bg-overlay px-3 py-2.5 flex flex-col gap-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-semibold shrink-0">显示 prompt</span>
              <span className="text-[11px] text-fg-tertiary shrink-0">
                （{pickedRows?.size ?? 0}/{grid.rows.length}）
              </span>
              <div className="flex items-center gap-2 flex-wrap">
                {grid.rows.map((r) => (
                  <label
                    key={r.index}
                    className="flex items-center gap-1 text-[11px] cursor-pointer max-w-[240px]"
                    title={r.prompt || r.image || ''}
                  >
                    <input
                      type="checkbox"
                      checked={pickedRows?.has(r.index) ?? false}
                      onChange={() => toggleRow(r.index)}
                    />
                    <span className="truncate font-mono text-fg-secondary">
                      {r.prompt || r.image || `#${r.index}`}
                    </span>
                  </label>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold shrink-0">显示 checkpoint</span>
              <select
                className="font-mono cursor-pointer"
                style={{
                  fontSize: 11,
                  padding: '4px 8px',
                  borderRadius: 'var(--r-md)',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-sunken)',
                  color: 'var(--fg-secondary)',
                }}
                value={ckptLimit}
                onChange={(e) => setCkptLimit(Number(e.target.value))}
                aria-label="显示多少个 checkpoint"
              >
                {CKPT_COUNT_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n === 0 ? `全部 ${totalCkpts} 个` : `最近 ${n} 个`}
                  </option>
                ))}
              </select>
              <span className="text-[11px] text-fg-tertiary">
                baseline 恒在第一列（纯底模对照）
              </span>
            </div>
          </div>

          {samples.length === 0 ? (
            <div className="rounded-md border border-dashed border-subtle px-3 py-3 text-sm text-fg-tertiary">
              {rows.length === 0
                ? '勾选至少一个 prompt 才能显示矩阵。'
                : '这次评估还没有出图（出图阶段可能仍在跑，或已失败）。'}
            </div>
          ) : (
            <div style={{ height: 520 }} className="flex">
              <PreviewXYGrid
                samples={samples}
                taskId={-1 /* 图走 imageUrl，不会回退到 generate cache */}
                xAxis={xAxis}
                yAxis={yAxis}
              />
            </div>
          )}
        </>
      )}
    </div>
  )
}
