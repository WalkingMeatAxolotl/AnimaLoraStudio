import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type LoraCkpt } from '../api/client'
import CreateEvalModal from './CreateEvalModal'

const ckpts: LoraCkpt[] = [
  { path: 'output/e1.safetensors', label: 'epoch 1', kind: 'epoch', epoch: 1, step: null, mtime: 1 },
  { path: 'output/e2.safetensors', label: 'epoch 2', kind: 'epoch', epoch: 2, step: null, mtime: 2 },
] as never

describe('CreateEvalModal', () => {
  beforeEach(() => {
    vi.spyOn(api, 'listVersionLoraCkpts').mockResolvedValue(ckpts)
    vi.spyOn(api, 'getEvalScale').mockResolvedValue({
      validation_images: 4, baseline_enabled: true, metric_runners: ['clip', 'dino'],
    } as never)
    vi.spyOn(api, 'runTaskEval').mockResolvedValue({ session: { id: 99 } } as never)
  })
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('没选 LoRA 时不能提交 —— 空评估没有意义', async () => {
    render(<CreateEvalModal pid={1} vid={2} onClose={() => {}} onCreated={() => {}} />)
    const submit = await screen.findByRole('button', { name: /创建评估/ })
    expect(submit).toBeDisabled()
  })

  it('选中后给出规模预估：候选含 baseline，作业恒为 1', async () => {
    render(<CreateEvalModal pid={1} vid={2} onClose={() => {}} onCreated={() => {}} />)
    fireEvent.click(await screen.findByTitle('output/e1.safetensors'))

    // 候选 = 1 选中 + baseline = 2；出图 = 2 × 4 张验证图 = 8
    await waitFor(() => expect(screen.getByText('8')).toBeInTheDocument())
    // 阶段 = 1 出图 + 2 个指标 runner
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('从概览发起时不带 task_id（评估不必挂在某次训练下）', async () => {
    const onCreated = vi.fn()
    render(<CreateEvalModal pid={1} vid={2} onClose={() => {}} onCreated={onCreated} />)
    fireEvent.click(await screen.findByTitle('output/e2.safetensors'))
    fireEvent.click(screen.getByRole('button', { name: /创建评估/ }))

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith(99))
    expect(api.runTaskEval).toHaveBeenCalledWith(1, 2, {
      task_id: undefined, checkpoints: ['output/e2.safetensors'],
    })
  })

  it('从训练详情发起时带上 task_id 作溯源', async () => {
    render(
      <CreateEvalModal pid={1} vid={2} taskId={55} onClose={() => {}} onCreated={() => {}} />,
    )
    fireEvent.click(await screen.findByTitle('output/e1.safetensors'))
    fireEvent.click(screen.getByRole('button', { name: /创建评估/ }))

    await waitFor(() => expect(api.runTaskEval).toHaveBeenCalledWith(1, 2, {
      task_id: 55, checkpoints: ['output/e1.safetensors'],
    }))
  })

  it('提交失败 → 就地显示错误，modal 不关', async () => {
    vi.spyOn(api, 'runTaskEval').mockRejectedValue(new Error('验证集为空'))
    const onCreated = vi.fn()
    render(<CreateEvalModal pid={1} vid={2} onClose={() => {}} onCreated={onCreated} />)
    fireEvent.click(await screen.findByTitle('output/e1.safetensors'))
    fireEvent.click(screen.getByRole('button', { name: /创建评估/ }))

    await waitFor(() => expect(screen.getByText(/验证集为空/)).toBeInTheDocument())
    expect(onCreated).not.toHaveBeenCalled()
  })
})
