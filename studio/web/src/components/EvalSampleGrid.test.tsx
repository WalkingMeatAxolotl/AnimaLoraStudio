import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type EvalSampleGrid as GridData } from '../api/client'
import EvalSampleGrid from './EvalSampleGrid'

/** 造一个 ckpts 个 checkpoint（+baseline）× rows 行的矩阵，cell 全满。 */
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
    index: i, image: `validation/1_data/img_00${i}.png`, folder: '1_data',
    prompt: `1girl, solo, a very long booru caption number ${i}`,
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

/** 打开某个 dropdown 并返回里面的 checkbox（勾选项收在 popover 里）。 */
async function openDropdown(label: string) {
  fireEvent.click(await screen.findByRole('button', { name: label }))
  return screen.getAllByRole('checkbox')
}

describe('EvalSampleGrid', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getEvalSessionGrid').mockResolvedValue(makeGrid(30, 5))
  })
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('prompt 显示验证图 id，完整 prompt 交给 hover', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    await waitFor(() => expect(screen.getAllByRole('img').length).toBeGreaterThan(0))

    // 行标签是 id（文件名去扩展名），不是那一长串 caption
    expect(screen.getByText('img_000')).toBeInTheDocument()
    expect(screen.queryByText(/very long booru caption number 0/)).not.toBeInTheDocument()
    // hover 才给完整 prompt
    expect(screen.getByTitle('1girl, solo, a very long booru caption number 0'))
      .toBeInTheDocument()

    // 下拉选项同款：显示 id，title 是 prompt
    const boxes = await openDropdown('prompt')
    expect(boxes.length).toBe(5)
    expect(screen.getAllByText('img_001').length).toBeGreaterThan(0)
  })

  it('默认勾前 3 个 prompt + baseline 和最近 20 个 lora', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    // 列 = baseline + 最近 20 = 21；行 3 → 63 张图
    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 3))
    expect(screen.getByRole('button', { name: 'prompt' })).toHaveTextContent('3/5')
    expect(screen.getByRole('button', { name: 'lora' })).toHaveTextContent('21/31')
  })

  it('lora 下拉列出**全部**候选（不是范围档位），默认勾最近那批', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    await waitFor(() => expect(screen.getAllByRole('img').length).toBeGreaterThan(0))

    const boxes = await openDropdown('lora')
    // 31 = baseline + 30 个 checkpoint，一个不少地都能勾
    expect(boxes.length).toBe(31)
    expect(boxes.filter((b) => (b as HTMLInputElement).checked).length).toBe(21)
    // 最近 20 = epoch 11..30，所以 epoch 1 在列表里但没勾
    expect(screen.getByTitle('output/e1.safetensors')).toBeInTheDocument()
  })

  it('勾上一个更早的 lora → 列数跟着变', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 3))

    const boxes = await openDropdown('lora')
    fireEvent.click(boxes.find((b) => !(b as HTMLInputElement).checked)!)

    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(22 * 3))
  })

  it('lora 下拉里「全选」→ 31 列全上', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 3))

    fireEvent.click(screen.getByRole('button', { name: 'lora' }))
    fireEvent.click(screen.getByText('全选'))

    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(31 * 3))
  })

  it('取消勾选 prompt → 行数跟着变', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 3))

    const boxes = await openDropdown('prompt')
    fireEvent.click(boxes.find((b) => (b as HTMLInputElement).checked)!)

    await waitFor(() => expect(screen.getAllByRole('img').length).toBe(21 * 2))
  })

  it('prompt 全取消 → 提示要选至少一个，不渲染网格', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
    await waitFor(() => expect(screen.getAllByRole('img').length).toBeGreaterThan(0))

    const boxes = await openDropdown('prompt')
    boxes.filter((b) => (b as HTMLInputElement).checked).forEach((b) => fireEvent.click(b))

    await waitFor(() =>
      expect(screen.getByText(/勾选至少一个 prompt/)).toBeInTheDocument())
    expect(screen.queryAllByRole('img').length).toBe(0)
  })

  it('图走 session 作用域 URL，不回退 generate cache', async () => {
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)
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

    await waitFor(() =>
      expect(screen.getByText(/还没有出图/)).toBeInTheDocument())
  })

  it('拉取失败 → 显示错误，不炸整个面板', async () => {
    vi.spyOn(api, 'getEvalSessionGrid').mockRejectedValue(new Error('boom'))
    render(<EvalSampleGrid pid={1} vid={2} sessionId={7} />)

    await waitFor(() =>
      expect(screen.getByText(/样图矩阵读取失败：.*boom/)).toBeInTheDocument())
  })
})
