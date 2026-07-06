/** R-5 — 数据任务视图（与 GPU 视图同源 /api/queue，resource_class=data）：
 *  分区渲染 / 行点击进统一详情 /queue/:id / 取消走 cancelTask（confirm）。 */
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../../components/Dialog'
import { ToastProvider } from '../../components/Toast'
import { api, type Task, type TaskType } from '../../api/client'
import DataJobsPanel from './DataJobsPanel'

function makeJobTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 1, name: 'tag', config_name: 'tag', task_type: 'tag',
    status: 'running', priority: 0,
    created_at: 900, started_at: 1000, finished_at: null,
    pid: 123, exit_code: null, output_dir: null, error_msg: null,
    project_id: 1, version_id: 2,
    params: '{}', params_decoded: { tagger: 'wd14' },
    ...overrides,
  }
}

class FakeEventSource {
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  close(): void { this.readyState = 2 }
}

beforeEach(() => {
  vi.stubGlobal('EventSource', FakeEventSource)
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
    ok: false, status: 404, json: async () => null, text: async () => '',
    headers: new Headers(),
  } as Response)))
  vi.spyOn(api, 'listProjects').mockResolvedValue([
    { id: 1, title: 'MyProj' } as never,
  ])
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function renderPanel(kind: TaskType | null = null, q?: string) {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <Routes>
            <Route
              path="/"
              element={
                <DataJobsPanel
                  kind={kind} q={q} historyPage={1} pageSize={20}
                  onHistoryTotal={() => {}} refreshToken={0}
                />
              }
            />
            <Route path="/queue/:id" element={<div data-testid="task-detail-route" />} />
          </Routes>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('DataJobsPanel', () => {
  it('渲染进行中/历史分区，行显示 kind 标签和项目名', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeJobTask({ id: 10, task_type: 'tag', status: 'running' }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [makeJobTask({
        id: 9, task_type: 'download', status: 'done', finished_at: 2000,
      })],
      total: 1, page: 1, page_size: 20,
    })

    renderPanel()

    await waitFor(() => expect(screen.getByTestId('job-row-10')).toBeInTheDocument())
    expect(within(screen.getByTestId('job-row-10')).getByText('打标')).toBeInTheDocument()
    expect(within(screen.getByTestId('job-row-9')).getByText('素材下载')).toBeInTheDocument()
    await waitFor(() => expect(screen.getAllByText(/MyProj/).length).toBeGreaterThan(0))
  })

  it('数据源 = /api/queue resource_class=data（kind/q 透传）', async () => {
    const liveSpy = vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    const histSpy = vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderPanel('download', 'usa')
    await waitFor(() =>
      expect(liveSpy).toHaveBeenCalledWith('usa', 'download', 'data'),
    )
    expect(histSpy).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'download', q: 'usa', resourceClass: 'data' }),
    )
  })

  it('点行 → 跳统一详情页 /queue/{id}（与 GPU 任务同构）', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([makeJobTask({ id: 10 })])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-row-10')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('job-row-10'))
    await waitFor(() => expect(screen.getByTestId('task-detail-route')).toBeInTheDocument())
  })

  it('取消需 confirm，确认后调 cancelTask（统一队列取消端点）', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([makeJobTask({ id: 10 })])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })
    const cancelSpy = vi.spyOn(api, 'cancelTask').mockResolvedValue({
      task_id: 10, canceled: true,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-cancel-btn-10')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('job-cancel-btn-10'))

    await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument())
    expect(cancelSpy).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('取消任务', { selector: 'button[type="submit"]' }))
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith(10))
  })

  it('done 行没有取消按钮，有跳转按钮', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [makeJobTask({ id: 9, status: 'done', finished_at: 2000 })],
      total: 1, page: 1, page_size: 20,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-row-9')).toBeInTheDocument())
    expect(screen.queryByTestId('job-cancel-btn-9')).not.toBeInTheDocument()
    expect(screen.getByTestId('job-jump-btn-9')).toBeInTheDocument()
  })
})
