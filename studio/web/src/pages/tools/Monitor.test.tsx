import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { api, type Task } from '../../api/client'
import MonitorPage from './Monitor'

vi.mock('../../components/MonitorDashboard', () => ({
  default: ({ taskId, taskStatus }: { taskId: number; taskStatus?: string }) => (
    <div data-testid="monitor-dashboard" data-task-id={taskId} data-task-status={taskStatus} />
  ),
}))

vi.mock('../../lib/useEventStream', () => ({ useEventStream: vi.fn() }))
vi.mock('../../lib/useTaskLog', () => ({
  useTaskLog: () => ({
    lines: [], status: 'ready', error: null, hasMoreBefore: false, loadingAll: false,
    loadAll: vi.fn(), refresh: vi.fn(), downloadUrl: null,
  }),
}))

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 1,
    name: 'portrait-v1',
    config_name: 'portrait-v1',
    task_type: 'train',
    status: 'running',
    priority: 0,
    created_at: 1000,
    started_at: 1100,
    finished_at: null,
    pid: 123,
    exit_code: null,
    output_dir: null,
    error_msg: null,
    monitor_state_path: 'studio_data/tasks/1/state.json',
    ...overrides,
  }
}

function renderPage(initialEntry = '/tools/monitor') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <MonitorPage />
    </MemoryRouter>,
  )
}

describe('MonitorPage task responsibility', () => {
  beforeEach(() => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 100,
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('defaults to the running train and exposes task details without task mutations', async () => {
    const running = makeTask({ id: 42 })
    vi.mocked(api.listQueueLive).mockResolvedValue([running])
    vi.mocked(api.listQueueHistory).mockResolvedValue({
      items: [
        makeTask({ id: 41, status: 'done', finished_at: 1200 }),
        makeTask({ id: 40, status: 'done', finished_at: 1190, monitor_state_path: null }),
      ],
      total: 2,
      page: 1,
      page_size: 100,
    })

    renderPage()

    expect(await screen.findByRole('heading', { name: '训练监控' })).toBeInTheDocument()
    const dashboard = await screen.findByTestId('monitor-dashboard')
    expect(dashboard).toHaveAttribute('data-task-id', '42')
    expect(screen.getByRole('link', { name: '查看任务详情' })).toHaveAttribute('href', '/queue/42')
    expect(screen.queryByRole('button', { name: /暂停|取消|重试/ })).not.toBeInTheDocument()
    expect(api.listQueueLive).toHaveBeenCalledWith(undefined, 'train')
    expect(api.listQueueHistory).toHaveBeenCalledWith(expect.objectContaining({ type: 'train' }))
    const options = screen.getAllByRole('option')
    expect(options.map((option) => option.textContent)).toEqual([
      '#42 · portrait-v1 · 运行中',
      '#41 · portrait-v1 · 已完成',
    ])
  })

  it('resolves a linked train outside the bounded history list without replacing it', async () => {
    const linked = makeTask({ id: 999, status: 'done', finished_at: 900 })
    const getTask = vi.spyOn(api, 'getTask').mockResolvedValue(linked)

    renderPage('/tools/monitor?task=999')

    const dashboard = await screen.findByTestId('monitor-dashboard')
    expect(dashboard).toHaveAttribute('data-task-id', '999')
    expect(getTask).toHaveBeenCalledWith(999)
    expect(screen.getByRole('option', { name: /#999/ })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '查看任务详情' })).toHaveAttribute('href', '/queue/999')
  })

  it('rejects a linked non-training task instead of showing a false empty monitor', async () => {
    vi.spyOn(api, 'getTask').mockResolvedValue(
      makeTask({ id: 77, task_type: 'download', status: 'done' }),
    )

    renderPage('/tools/monitor?task=77')

    expect(await screen.findByText('任务 #77 不是训练任务，无法在训练监控中打开。')).toBeInTheDocument()
    expect(screen.queryByTestId('monitor-dashboard')).not.toBeInTheDocument()
  })

  it('shows a recoverable list error instead of treating failure as an empty queue', async () => {
    vi.mocked(api.listQueueLive).mockRejectedValue(new Error('network down'))

    renderPage()

    expect(await screen.findByText('任务列表暂时不可用。请重试；这不会重新运行任何任务。')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '重试' })).toBeInTheDocument()
    expect(screen.queryByText('暂无训练监控记录')).not.toBeInTheDocument()
    await waitFor(() => expect(api.listQueueLive).toHaveBeenCalledTimes(1))
  })
})
