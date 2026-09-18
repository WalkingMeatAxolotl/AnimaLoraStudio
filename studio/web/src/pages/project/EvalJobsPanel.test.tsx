import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type LoraCkpt } from '../../api/client'
import EvalJobsPanel from './EvalJobsPanel'

vi.mock('../../components/TaskLogDrawer', () => ({
  default: () => null,
}))

vi.mock('../../lib/useTaskLog', () => ({
  useTaskLog: () => ({
    lines: [],
    downloadUrl: null,
    hasMoreBefore: false,
    loadingAll: false,
    loadAll: vi.fn(),
  }),
}))

const checkpoint: LoraCkpt = {
  path: 'output/e1.safetensors',
  label: 'epoch 1',
  kind: 'epoch',
  epoch: 1,
  step: null,
  mtime: 1,
} as never

describe('EvalJobsPanel', () => {
  beforeEach(() => {
    vi.spyOn(api, 'listEvalSessions').mockResolvedValue({ sessions: [] })
    vi.spyOn(api, 'listVersionLoraCkpts').mockResolvedValue([checkpoint])
    vi.spyOn(api, 'getEvalScale').mockResolvedValue({
      validation_images: 4,
      baseline_enabled: true,
      metric_runners: ['clip'],
    } as never)
    vi.spyOn(api, 'runTaskEval').mockResolvedValue({
      session: {
        id: 99,
        task_id: 42,
        parent_task_id: null,
        project_id: 1,
        version_id: 2,
        trigger: 'manual',
        status: 'pending',
        stage: null,
        created_at: 1,
        started_at: null,
        finished_at: null,
        error: null,
      },
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('创建成功后留在列表，并提供查看任务入口', async () => {
    render(
      <MemoryRouter>
        <EvalJobsPanel pid={1} vid={2} />
      </MemoryRouter>,
    )

    const createButtons = await screen.findAllByRole('button', { name: '创建新评估' })
    fireEvent.click(createButtons[0])
    fireEvent.click(await screen.findByTitle('output/e1.safetensors'))
    fireEvent.click(screen.getByRole('button', { name: '创建评估 (1)' }))

    await waitFor(() => expect(screen.getByText('评估已排队')).toBeInTheDocument())
    expect(screen.getByRole('link', { name: '查看任务' })).toHaveAttribute('href', '/queue/42')
  })
})
