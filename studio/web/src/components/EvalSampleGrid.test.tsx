import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type EvalSampleGrid as GridData } from '../api/client'
import EvalSampleGrid from './EvalSampleGrid'

/** 造一个 cols 列（含 baseline）× rows 行的矩阵，cell 全满。 */
function makeGrid(ckpts: number, rows: number): GridData {
  const columns: GridData['columns'] = [{
    candidate_id: 1000, role: 'baseline', label: 'baseline',
    checkpoint_path: 'output/a.safetensors', epoch: null, step: null,
    status: 'done', run_id: 'run-base',
  }]
  for (let i = 0; i < ckpts; i++) {
    columns.push({
      candidate_id: i + 1, role: 'checkpoint', label: `epoch ${i + 1}`,
      checkpoint_path: `output/e${i + 1}.safetensors`, epoch: i + 1, step: null,
      status: 'done', run_id: `run-${i + 1}`,
    })
  }
  const gridRows: GridData['rows'] = Array.from({ length: rows }, (_, i) => ({
    index: i, image: `validation/1_data/v${i}.png`, folder: '1_data',
    prompt: `prompt ${i}`,
  }))
  const cells: GridData['cells'] = {}
  for (const c of columns) {
    for (const r of gridRows) {
      cells[`${c.candidate_id}:${r.index}`] = {
        run_id: c.run_id!, filename: `sample_000${r.index}_s42.png`, status: 'done',
      }
    }
  }
  return { session_id: 7, columns, rows: gridRows, cells }
}

function open() {
  fireEvent.click(screen.getByRole('button', { name: /样图对比/ }))
}

describe('EvalSampleGrid', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getEvalSessionGrid').mockResolvedValue(makeGrid(30, 5))
  })
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('折叠时不拉数据（几千个 cell 的矩阵不该在进页面时就取）', () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    expect(api.getEvalSessionGrid).not.toHaveBeenCalled()
  })

  it('展开后默认只显示前 3 个 prompt 和最近 20 个 checkpoint', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    open()
    await waitFor(() => expect(api.getEvalSessionGrid).toHaveBeenCalled())

    // 5 个 prompt 全部列为 checkbox，但只勾前 3 个
    const boxes = await screen.findAllByRole('checkbox')
    expect(boxes.length).toBe(5)
    expect(boxes.filter((b) => (b as HTMLInputElement).checked).length).toBe(3)

    // 列 = baseline + 最近 20 个 checkpoint = 21；行 3 → 63 张图
    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 3))
  })

  it('baseline 恒在第一列，checkpoint 取最近的（末尾）', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    open()
    await waitFor(() => expect(screen.getAllByRole('img').length).toBeGreaterThan(0))

    // 表头第一格是 baseline；最近 20 个 = epoch 11..30，所以 epoch 1 不在
    expect(screen.getByTitle('baseline')).toBeInTheDocument()
    expect(screen.getByTitle('epoch 30')).toBeInTheDocument()
    expect(screen.queryByTitle('epoch 1')).toBeNull()
  })

  it('切换 checkpoint 数量到全部 → 列数跟着变', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    open()
    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 3))

    fireEvent.change(
      screen.getByLabelText('显示多少个 checkpoint'), { target: { value: '0' } },
    )

    // 全部 30 个 + baseline = 31 列 × 3 行
    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(31 * 3))
    expect(screen.getByTitle('epoch 1')).toBeInTheDocument()
  })

  it('取消勾选 prompt → 行数跟着变', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    open()
    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 3))

    const boxes = screen.getAllByRole('checkbox')
    fireEvent.click(boxes[0])

    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 2))
  })

  it('全部取消勾选 → 提示要选至少一个，不渲染网格', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    open()
    const boxes = await screen.findAllByRole('checkbox')
    boxes.filter((b) => (b as HTMLInputElement).checked).forEach((b) => fireEvent.click(b))

    await waitFor(() =>
      expect(screen.getByText(/勾选至少一个 prompt/)).toBeInTheDocument())
    expect(screen.queryAllByRole('img').length).toBe(0)
  })

  it('图走 session 作用域 URL，不回退 generate cache', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    open()
    await waitFor(() => expect(screen.getAllByRole('img').length).toBeGreaterThan(0))

    const img = screen.getAllByRole('img')[0] as HTMLImageElement
    expect(img.src).toContain('/api/projects/1/versions/2/eval/samples/')
    expect(img.src).toContain('session_id=7')
    expect(img.src).not.toContain('/api/generate/')
  })

  it('还没出图（cells 空）→ 提示而不是空网格', async () => {
    const empty = makeGrid(3, 2)
    empty.cells = {}
    vi.spyOn(api, 'getEvalSessionGrid').mockResolvedValue(empty)
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    open()

    await waitFor(() =>
      expect(screen.getByText(/还没有出图/)).toBeInTheDocument())
  })

  it('拉取失败 → 显示错误，不炸整个面板', async () => {
    vi.spyOn(api, 'getEvalSessionGrid').mockRejectedValue(new Error('boom'))
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    open()

    await waitFor(() =>
      expect(screen.getByText(/样图矩阵读取失败：.*boom/)).toBeInTheDocument())
  })
})
