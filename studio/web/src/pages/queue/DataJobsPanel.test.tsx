/** R-5 — 数据任务视图（与 GPU 视图同源 /api/queue，resource_class=data）：
 *  分区渲染 / 行点击进统一详情 /queue/:id / 取消走 cancelTask（confirm）。 */
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../../components/Dialog'
import { ToastProvider } from '../../components/Toast'
import { api, type Task, type TaskType } from '../../api/client'
import i18n from '../../i18n'
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
  static instances: FakeEventSource[] = []
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  constructor() { FakeEventSource.instances.push(this) }
  close(): void { this.readyState = 2 }
  emit(evt: unknown): void { this.onmessage?.({ data: JSON.stringify(evt) }) }
}

beforeEach(() => {
  FakeEventSource.instances = []
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

describe('DataJobsPanel 加载恢复', () => {
  afterEach(async () => {
    await act(async () => { await i18n.changeLanguage('zh') })
  })

  function deferred<T>() {
    let resolve!: (value: T) => void
    let reject!: (reason: Error) => void
    const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
    return { promise, resolve, reject }
  }

  const emptyHistory = { items: [], total: 0, page: 1, page_size: 20 }
  const copies = [
    { lang: 'zh', title: '数据任务加载失败', reload: '重新加载', empty: '暂无数据任务', loading: '加载中...' },
    { lang: 'en', title: 'Failed to load data tasks', reload: 'Reload', empty: 'No data tasks yet', loading: 'Loading...' },
  ]

  for (const copy of copies) {
    it.each(['live', 'history'] as const)(`${copy.lang}：%s 失败不显示空态，重复失败后可恢复到真实空列表`, async (side) => {
      await i18n.changeLanguage(copy.lang)
      const user = userEvent.setup()
      const initial = deferred<never>()
      const repeated = deferred<never>()
      const liveSpy = vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
      const historySpy = vi.spyOn(api, 'listQueueHistory').mockResolvedValue(emptyHistory)
      const failedRead = side === 'live' ? liveSpy : historySpy
      failedRead.mockReturnValueOnce(initial.promise).mockReturnValueOnce(repeated.promise)
      const cancelSpy = vi.spyOn(api, 'cancelTask')
      renderPanel()
      expect(screen.getByText(copy.loading)).toBeInTheDocument()
      expect(screen.queryByText(copy.empty)).not.toBeInTheDocument()

      await act(async () => { initial.reject(new Error('offline')) })
      const alert = await screen.findByRole('alert')
      expect(alert).toHaveTextContent(copy.title)
      expect(alert).toHaveTextContent('offline')
      expect(screen.queryByText(copy.loading)).not.toBeInTheDocument()
      expect(screen.getByTestId('data-jobs-panel').querySelector('.empty-state')).toBeNull()
      const reload = within(alert).getByRole('button', { name: copy.reload })
      await user.click(reload)
      expect(reload).toBeDisabled()
      expect(reload).toHaveAttribute('aria-busy', 'true')
      await user.click(reload)
      expect(liveSpy).toHaveBeenCalledTimes(2)
      expect(historySpy).toHaveBeenCalledTimes(2)

      await act(async () => { repeated.reject(new Error('still offline')) })
      await waitFor(() => expect(reload).toBeEnabled())
      expect(alert).toHaveTextContent('still offline')
      expect(screen.getByTestId('data-jobs-panel').querySelector('.empty-state')).toBeNull()
      await user.click(reload)
      expect(await screen.findByText(copy.empty)).toBeInTheDocument()
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
      expect(liveSpy).toHaveBeenCalledTimes(3)
      expect(historySpy).toHaveBeenCalledTimes(3)
      expect(cancelSpy).not.toHaveBeenCalled()
    })
  }

  it.each([true, false])('刷新失败保留已有行，但不把旧空列表当作当前结果（有行=%s）', async (hasRows) => {
    const user = userEvent.setup()
    const liveSpy = vi.spyOn(api, 'listQueueLive').mockResolvedValue(hasRows ? [makeJobTask({ id: 10 })] : [])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue(hasRows ? {
      items: [makeJobTask({ id: 9, status: 'done' })], total: 1, page: 1, page_size: 20,
    } : emptyHistory)
    renderPanel()
    if (hasRows) await screen.findByTestId('job-row-10')
    else await screen.findByText('暂无数据任务')

    liveSpy.mockRejectedValueOnce(new Error('refresh offline'))
    act(() => {
      FakeEventSource.instances[FakeEventSource.instances.length - 1].emit({ type: 'task_state_changed', task_id: 10 })
    })
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('refresh offline')
    expect(screen.queryByText('加载中...')).not.toBeInTheDocument()
    expect(screen.getByTestId('data-jobs-panel').querySelector('.empty-state')).toBeNull()
    if (hasRows) {
      expect(screen.getByTestId('job-row-10')).toBeInTheDocument()
      expect(screen.getByTestId('job-row-9')).toBeInTheDocument()
    }
    await user.click(within(alert).getByRole('button', { name: '重新加载' }))
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
    if (!hasRows) expect(screen.getByText('暂无数据任务')).toBeInTheDocument()
  })

  it('较早的重新加载响应不覆盖较新的列表', async () => {
    const user = userEvent.setup()
    const oldRead = deferred<Task[]>()
    vi.spyOn(api, 'listQueueLive')
      .mockRejectedValueOnce(new Error('initial offline'))
      .mockReturnValueOnce(oldRead.promise)
      .mockResolvedValue([makeJobTask({ id: 20 })])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue(emptyHistory)
    renderPanel()
    const alert = await screen.findByRole('alert')
    await user.click(within(alert).getByRole('button', { name: '重新加载' }))
    act(() => {
      FakeEventSource.instances[FakeEventSource.instances.length - 1].emit({ type: 'task_state_changed', task_id: 20 })
    })
    await screen.findByTestId('job-row-20')
    await act(async () => { oldRead.resolve([]) })
    expect(screen.getByTestId('job-row-20')).toBeInTheDocument()
    expect(screen.queryByText('暂无数据任务')).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

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
    expect(screen.getByTestId('data-jobs-panel')).toHaveClass('gap-section')
    expect(screen.getByTestId('job-row-10').firstElementChild)
      .toHaveClass('ui-queue-job-grid')
    expect(screen.getByTestId('job-row-10').querySelector('.ui-queue-job-timing'))
      .toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 3, name: /进行中/ }))
      .toHaveClass('type-section-label')
    expect(screen.getByRole('heading', { level: 3, name: /进行中/ }).parentElement)
      .toHaveClass('gap-related')
    expect(screen.getByRole('heading', { level: 3, name: /历史/ }))
      .toHaveClass('type-section-label')
    expect(within(screen.getByTestId('job-row-10')).getByText('打标')).toBeInTheDocument()
    expect(within(screen.getByTestId('job-row-9')).getByText('素材下载')).toBeInTheDocument()
    await waitFor(() => expect(screen.getAllByText(/MyProj/).length).toBeGreaterThan(0))
  })

  it('空数据视图使用共享的主空状态层级', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderPanel()

    const title = await screen.findByText('暂无数据任务')
    expect(title.closest('.empty-state')).toHaveClass('card', 'empty-state')
    expect(screen.getByText('下载 / 打标 / 正则构建 / 评估等数据任务会显示在这里'))
      .toHaveClass('empty-state-description')
  })

  it('加载失败使用共享 danger Alert 与即时播报语义', async () => {
    vi.spyOn(api, 'listQueueLive').mockRejectedValue(new Error('offline'))
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderPanel()

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveClass('alert', 'alert-danger', 'alert-sm')
    expect(within(alert).getByText('Error: offline')).toHaveClass('font-mono')
    expect(alert).toHaveTextContent('offline')
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
    const row = screen.getByTestId('job-row-10')
    const link = within(row).getByRole('link', { name: '任务 #10：tag' })
    expect(link).toHaveAttribute('href', '/queue/10')
    expect(link.querySelector('button')).toBeNull()
    expect(row.querySelector('button button')).toBeNull()
    fireEvent.click(link)
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
    expect(screen.queryByTestId('task-detail-route')).not.toBeInTheDocument()
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
