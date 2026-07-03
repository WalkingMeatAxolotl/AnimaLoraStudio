/** 0.17 P-G — 数据作业只读区：分区渲染 / 行内展开日志 / 取消（confirm）。 */
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../../components/Dialog'
import { ToastProvider } from '../../components/Toast'
import { api, type Job } from '../../api/client'
import DataJobsPanel from './DataJobsPanel'

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: 1, project_id: 1, version_id: 2, kind: 'tag',
    params: '{}', params_decoded: { tagger: 'wd14' },
    status: 'running', started_at: 1000, finished_at: null,
    pid: 123, log_path: '/x/1.log', error_msg: null,
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

function renderPanel() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <DataJobsPanel />
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('DataJobsPanel', () => {
  it('渲染进行中/历史分区，行显示 kind 标签和项目名', async () => {
    vi.spyOn(api, 'listJobsLive').mockResolvedValue([
      makeJob({ id: 10, kind: 'tag', status: 'running' }),
    ])
    vi.spyOn(api, 'listJobsHistory').mockResolvedValue({
      items: [makeJob({ id: 9, kind: 'download', status: 'done', finished_at: 2000 })],
      total: 1, page: 1, page_size: 20,
    })

    renderPanel()

    await waitFor(() => expect(screen.getByTestId('job-row-10')).toBeInTheDocument())
    // kind 下拉里也有同名 option，用 within 限定到行内断言
    expect(within(screen.getByTestId('job-row-10')).getByText('打标')).toBeInTheDocument()
    expect(within(screen.getByTestId('job-row-9')).getByText('素材下载')).toBeInTheDocument()
    await waitFor(() => expect(screen.getAllByText(/MyProj/).length).toBeGreaterThan(0))
  })

  it('点行 → 展开显示 params + 日志（getJobLog tail）', async () => {
    vi.spyOn(api, 'listJobsLive').mockResolvedValue([makeJob({ id: 10 })])
    vi.spyOn(api, 'listJobsHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })
    const logSpy = vi.spyOn(api, 'getJobLog').mockResolvedValue({
      job_id: 10, content: 'hello log line', size: 14,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-row-10')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('job-row-10'))

    await waitFor(() => expect(screen.getByTestId('job-expand-10')).toBeInTheDocument())
    expect(logSpy).toHaveBeenCalledWith(10, 300)
    await waitFor(() => expect(screen.getByText(/hello log line/)).toBeInTheDocument())
    expect(screen.getByText(/tagger:/)).toBeInTheDocument()
  })

  it('取消作业需 confirm，确认后调 cancelJob', async () => {
    vi.spyOn(api, 'listJobsLive').mockResolvedValue([makeJob({ id: 10 })])
    vi.spyOn(api, 'listJobsHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })
    const cancelSpy = vi.spyOn(api, 'cancelJob').mockResolvedValue({
      job_id: 10, canceled: true,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-cancel-btn-10')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('job-cancel-btn-10'))

    await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument())
    expect(cancelSpy).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('取消作业'))
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith(10))
  })

  it('done 行没有取消按钮，有跳转按钮', async () => {
    vi.spyOn(api, 'listJobsLive').mockResolvedValue([])
    vi.spyOn(api, 'listJobsHistory').mockResolvedValue({
      items: [makeJob({ id: 9, status: 'done', finished_at: 2000 })],
      total: 1, page: 1, page_size: 20,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-row-9')).toBeInTheDocument())
    expect(screen.queryByTestId('job-cancel-btn-9')).not.toBeInTheDocument()
    expect(screen.getByTestId('job-jump-btn-9')).toBeInTheDocument()
  })
})
