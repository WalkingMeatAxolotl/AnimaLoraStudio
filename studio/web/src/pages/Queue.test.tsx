/** Queue page — taskKind 契约测试（0.17 P-D）。
 *
 *  taskKind 从后端权威 `task.task_type`（train/reg_ai/generate）派生队列行的类型，
 *  取代旧 inferKind 的 config_name 子串猜测。核心契约：
 *   1) 直接返回后端 task_type；
 *   2) 缺字段（老 mock / 极老行）兜底 'train'；
 *   3) 不再受 config_name 影响 —— 修掉旧 inferKind 把名字含 "reg"/"tag" 的
 *      训练任务误判成别的类型的 latent bug。 */
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../components/Dialog'
import { ToastProvider } from '../components/Toast'
import { api, type QueueHistoryPage, type Task } from '../api/client'
import i18n from '../i18n'
import QueuePage, { QueueTaskRow, taskKind } from './Queue'

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 1, name: 'train', config_name: 'train',
    status: 'running', priority: 0,
    created_at: 1000, started_at: 1100, finished_at: null,
    pid: 1234, exit_code: null, output_dir: null, error_msg: null,
    ...overrides,
  }
}

describe('taskKind', () => {
  it('直接返回后端 task_type', () => {
    expect(taskKind(makeTask({ task_type: 'train' }))).toBe('train')
    expect(taskKind(makeTask({ task_type: 'reg_ai' }))).toBe('reg_ai')
    expect(taskKind(makeTask({ task_type: 'generate' }))).toBe('generate')
  })

  it('缺 task_type（老行 / 老 mock）兜底 train', () => {
    expect(taskKind(makeTask())).toBe('train')
  })

  it('不看 config_name —— 名字含 reg/tag 的训练任务仍是 train', () => {
    expect(taskKind(makeTask({ task_type: 'train', config_name: 'my_reg_lora' }))).toBe('train')
    expect(taskKind(makeTask({ task_type: 'train', config_name: 'wd14_tag_run' }))).toBe('train')
  })
})

// --- 0.17 P-A/P-C/P-E 页面级：分区 + 分页 ------------------------------------

class FakeEventSource {
  static instances: FakeEventSource[] = []
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  constructor(public url: string) { FakeEventSource.instances.push(this) }
  close(): void { this.readyState = 2 }
}

beforeEach(() => {
  // 队列过滤持久化到 localStorage（0.17 item4）→ 清掉避免测试间泄漏。
  localStorage.clear()
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
  // monitor / eval 等次要拉取统一 404 安静失败；queue 数据源单独 spy。
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
    ok: false, status: 404, json: async () => null, text: async () => '',
    headers: new Headers(),
  } as Response)))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function renderQueue() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <QueuePage />
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('QueueTaskRow 状态与信息', () => {
  function renderRow(task: Task, monitor: { step?: number | null; total_steps?: number | null } | null = null) {
    return render(<MemoryRouter><QueueTaskRow task={task} runningTaskId={task.id} monitor={monitor}
      isWaitingForRelease={false} onResume={vi.fn()} onCancelPaused={vi.fn()}
      onCancelPending={vi.fn()} onStartNow={vi.fn()} onCancelScheduled={vi.fn()} /></MemoryRouter>)
  }

  it.each([
    { monitor: null, value: undefined },
    { monitor: { step: 0, total_steps: 100 }, value: '0' },
    { monitor: { step: 40, total_steps: 100 }, value: '40' },
    { monitor: { step: 4, total_steps: 0 }, value: undefined },
    { monitor: { step: -1, total_steps: 100 }, value: undefined },
    { monitor: { step: Number.NaN, total_steps: 100 }, value: undefined },
  ])('仅有效步数显示定量进度：$monitor', ({ monitor, value }) => {
    renderRow(makeTask(), monitor)
    const progress = screen.getByRole('progressbar', { name: '任务 #1 进度' })
    if (value == null) {
      expect(progress).not.toHaveAttribute('aria-valuenow')
      expect(progress).toHaveAttribute('data-state', 'indeterminate')
      expect(progress).toHaveAttribute('aria-valuetext', '进度未知')
    } else {
      expect(progress).toHaveAttribute('aria-valuenow', value)
      expect(progress).toHaveAttribute('aria-valuemax', '100')
    }
  })

  it('详情入口为原生链接，与局部按钮同级且键盘可达', async () => {
    const user = userEvent.setup()
    renderRow(makeTask({ id: 41, status: 'paused', is_resumable: true }))
    const link = screen.getByRole('link', { name: '任务 #41：train' })
    expect(link).toHaveAttribute('href', '/queue/41')
    expect(link.querySelector('button')).toBeNull()
    expect(link.closest('button')).toBeNull()
    const resume = screen.getByTestId('resume-btn-41')
    expect(link.contains(resume)).toBe(false)
    await user.tab()
    expect(link).toHaveFocus()
    await user.tab()
    expect(resume).toHaveFocus()
  })

  it.each(['done', 'failed', 'canceled'] as const)('%s 的结束时间不再一律称为完成', (status) => {
    renderRow(makeTask({ status, finished_at: 1200 }))
    expect(screen.getByText('结束')).toBeInTheDocument()
    if (status !== 'done') expect(screen.queryByText('完成')).not.toBeInTheDocument()
  })

  it('等待任务不再声称前方任务数量；长名称、配置和错误保留完整提示', () => {
    const name = 'Long task '.repeat(25).trim()
    const config = 'long-config-'.repeat(25)
    const error = 'Full error '.repeat(25).trim()
    const view = renderRow(makeTask({ status: 'pending', name, config_name: config }))
    expect(screen.queryByText(/前面.*个/)).not.toBeInTheDocument()
    expect(screen.getByTitle(name)).toHaveTextContent(name.trim())
    expect(screen.getByTitle(config)).toHaveTextContent(config)
    view.unmount()
    renderRow(makeTask({ status: 'failed', error_msg: error }))
    expect(screen.getByTitle(error)).toHaveTextContent(error.trim())
  })
})

describe('QueuePage 取消当前任务提示', () => {
  afterEach(async () => {
    await act(async () => { await i18n.changeLanguage('zh') })
  })

  const languages = [
    {
      language: 'zh', open: '取消当前任务', dismiss: '取消',
      train: '取消当前任务 #42？将发送停止请求。已有恢复点会保留；仅在存在可用恢复点时才能继续训练。',
      generate: '取消当前任务 #42？将发送停止请求，终止本次任务。',
    },
    {
      language: 'en', open: 'Cancel current task', dismiss: 'Cancel',
      train: 'Cancel current task #42? This sends a stop request. Existing recovery checkpoints are kept; training can resume only if a usable checkpoint is available.',
      generate: 'Cancel current task #42? This sends a stop request to end this task.',
    },
  ] as const

  for (const copy of languages) {
    it.each(['train', 'generate'] as const)(`${copy.language}：%s 提示准确且仅确认后发送取消请求`, async (taskType) => {
      await i18n.changeLanguage(copy.language)
      localStorage.setItem('studio:queue:typeFilter', JSON.stringify(taskType))
      const user = userEvent.setup()
      vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false, pending_waiting: 0 })
      vi.spyOn(api, 'listQueueLive').mockResolvedValue([
        makeTask({ id: 42, task_type: taskType, is_resumable: false }),
      ])
      vi.spyOn(api, 'listQueueHistory').mockResolvedValue({ items: [], total: 0, page: 1, page_size: 20 })
      const cancelSpy = vi.spyOn(api, 'cancelTask').mockResolvedValue({ task_id: 42, canceled: true })
      renderQueue()

      const open = await screen.findByRole('button', { name: copy.open })
      await user.click(open)
      const dialog = await screen.findByRole('alertdialog')
      expect(dialog).toHaveTextContent(copy[taskType])
      expect(cancelSpy).not.toHaveBeenCalled()
      await user.click(within(dialog).getByRole('button', { name: copy.dismiss }))
      expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
      expect(cancelSpy).not.toHaveBeenCalled()

      await user.click(open)
      await user.click(within(await screen.findByRole('alertdialog')).getByRole('button', { name: copy.open }))
      await waitFor(() => expect(cancelSpy).toHaveBeenCalledTimes(1))
      expect(cancelSpy).toHaveBeenCalledWith(42)
    })
  }
})

describe('QueuePage 加载状态隔离', () => {
  const emptyHistory: QueueHistoryPage = { items: [], total: 0, page: 1, page_size: 20 }
  function deferred<T>() {
    let resolve!: (value: T) => void
    let reject!: (reason: Error) => void
    const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
    return { promise, resolve, reject }
  }
  beforeEach(() => {
    // 本组显式控制读取顺序；浏览器 EventSource 创建时仍在 CONNECTING。
    vi.stubGlobal('EventSource', class extends FakeEventSource { readyState = 0 })
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false, pending_waiting: 0 })
  })

  it.each(['live', 'history'] as const)('%s 失败不会被另一数据源的成功清除，局部重试只读失败的数据源', async (side) => {
    const user = userEvent.setup()
    const firstLive = deferred<Task[]>()
    const firstHistory = deferred<QueueHistoryPage>()
    const retry = deferred<never>()
    const liveSpy = vi.spyOn(api, 'listQueueLive').mockReturnValueOnce(firstLive.promise).mockResolvedValue([])
    const historySpy = vi.spyOn(api, 'listQueueHistory').mockReturnValueOnce(firstHistory.promise).mockResolvedValue(emptyHistory)
    const failedRead = side === 'live' ? liveSpy : historySpy
    failedRead.mockReturnValueOnce(retry.promise)
    renderQueue()
    expect(screen.queryByText('暂无训练任务')).not.toBeInTheDocument()
    await act(async () => { (side === 'live' ? firstLive : firstHistory).reject(new Error('source offline')) })
    const alert = await screen.findByTestId(`queue-${side}-error`)
    await act(async () => {
      if (side === 'live') firstHistory.resolve(emptyHistory)
      else firstLive.resolve([])
    })
    expect(alert).toHaveTextContent('source offline')
    expect(screen.queryByText('暂无训练任务')).not.toBeInTheDocument()
    expect(screen.queryByTestId('queue-loading')).not.toBeInTheDocument()
    const reload = within(alert).getByRole('button', { name: '重新加载' })
    await user.click(reload)
    expect(reload).toBeDisabled()
    expect(reload).toHaveAttribute('aria-busy', 'true')
    await user.click(reload)
    expect(failedRead).toHaveBeenCalledTimes(2)
    expect(side === 'live' ? historySpy : liveSpy).toHaveBeenCalledTimes(1)
    await act(async () => { retry.reject(new Error('still offline')) })
    await waitFor(() => expect(reload).toBeEnabled())
    expect(alert).toHaveTextContent('still offline')
    await user.click(reload)
    await screen.findByText('暂无训练任务')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(side === 'live' ? historySpy : liveSpy).toHaveBeenCalledTimes(1)
  })

  it('两路都失败时，恢复一路不清除另一路错误', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'listQueueLive').mockRejectedValueOnce(new Error('live offline')).mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockRejectedValue(new Error('history offline'))
    renderQueue()
    await screen.findByTestId('queue-history-error')
    await user.click(within(screen.getByTestId('queue-live-error')).getByRole('button', { name: '重新加载' }))
    await waitFor(() => expect(screen.queryByTestId('queue-live-error')).not.toBeInTheDocument())
    expect(screen.getByTestId('queue-history-error')).toHaveTextContent('history offline')
    expect(screen.queryByText('暂无训练任务')).not.toBeInTheDocument()
  })

  it('历史刷新失败保留已加载的行', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    const historySpy = vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      ...emptyHistory, items: [makeTask({ id: 8, name: 'Retained history', status: 'failed' })], total: 1,
    })
    renderQueue()
    await screen.findByText('Retained history')
    historySpy.mockRejectedValueOnce(new Error('refresh failed'))
    await user.click(screen.getByRole('button', { name: '刷新' }))
    await screen.findByTestId('queue-history-error')
    expect(screen.getByText('Retained history')).toBeInTheDocument()
  })

  it('旧的列表响应不覆盖新结果', async () => {
    const user = userEvent.setup()
    const old = deferred<Task[]>()
    const liveSpy = vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue(emptyHistory)
    renderQueue()
    await screen.findByText('暂无训练任务')
    liveSpy.mockReturnValueOnce(old.promise).mockResolvedValue([makeTask({ id: 42, name: 'Latest row', status: 'pending' })])
    await user.click(screen.getByRole('button', { name: '刷新' }))
    await user.click(screen.getByRole('button', { name: '刷新' }))
    await screen.findByText('Latest row')
    await act(async () => { old.resolve([]) })
    expect(screen.getByText('Latest row')).toBeInTheDocument()
  })

  it.each([null, 'generate'] as const)('空结果说明当前范围（type=%s）', async (type) => {
    localStorage.setItem('studio:queue:typeFilter', JSON.stringify(type))
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue(emptyHistory)
    renderQueue()
    await screen.findByText(type ? i18n.t('queue.noMatch') : 'GPU 队列为空')
    expect(screen.queryByText('暂无训练任务')).not.toBeInTheDocument()
  })

  it('GPU读取错误不泄漏到数据任务视图', async () => {
    localStorage.setItem('studio:queue:tab', JSON.stringify('jobs'))
    vi.spyOn(api, 'listProjects').mockResolvedValue([])
    vi.spyOn(api, 'listQueueLive').mockImplementation((_q, _type, resource) => resource === 'data'
      ? Promise.resolve([]) : Promise.reject(new Error('GPU offline')))
    vi.spyOn(api, 'listQueueHistory').mockImplementation((opts) => opts.resourceClass === 'data'
      ? Promise.resolve(emptyHistory) : Promise.reject(new Error('GPU offline')))
    renderQueue()
    await screen.findByText('暂无数据任务')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

describe('QueuePage 分区 + 分页', () => {
  it('空队列使用共享的主空状态层级', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderQueue()

    const title = await screen.findByText('暂无训练任务')
    expect(title.closest('.empty-state')).toHaveClass('card', 'empty-state')
    expect(screen.getByText('当前仅显示训练任务。可展开筛选切换类型，或从项目训练页入队。'))
      .toHaveClass('empty-state-description')
  })

  it('全局调度入口在 GPU 与数据视图都可见，并明确影响全部调度', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderQueue()

    expect(await screen.findByRole('button', { name: '挂起队列' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('radio', { name: '数据任务' }))
    expect(screen.getByRole('button', { name: '挂起队列' })).toBeInTheDocument()
  })

  it('队列挂起使用共享 warning Alert，并保留恢复操作', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({
      held: true, pending_waiting: 2,
    })
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderQueue()

    const banner = await screen.findByTestId('queue-hold-banner')
    expect(banner).toHaveClass('alert', 'alert-warning', 'alert-sm')
    expect(banner).not.toHaveClass('sticky')
    expect(within(banner).getByRole('button', { name: '恢复调度' }))
      .toHaveClass('btn', 'btn-ghost', 'btn-xs')
  })

  it('渲染进行中/等待/历史三分区，历史超过一页时固定分页器', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeTask({ id: 10, name: 'run', status: 'running', started_at: 1000 }),
      makeTask({ id: 11, name: 'pend', status: 'pending' }),
    ])
    const historySpy = vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [makeTask({ id: 9, name: 'old', status: 'done', finished_at: 900 })],
      total: 25, page: 1, page_size: 20,
    })

    renderQueue()

    expect(screen.getByTestId('queue-page'))
      .toHaveClass('h-full', 'min-h-0', 'flex', 'flex-col', 'overflow-hidden')
    expect(screen.getByTestId('queue-page'))
      .toHaveAttribute('data-app-shell-scroll', 'contained')
    expect(screen.getByTestId('queue-page')).not.toHaveClass('min-h-full')
    expect(screen.getByTestId('queue-scroll-region'))
      .toHaveClass('ui-queue-scroll-region', 'flex-1', 'min-h-0')
    expect(screen.getByTestId('queue-scroll-region'))
      .toHaveAttribute('role', 'region')
    expect(screen.getByTestId('queue-scroll-region'))
      .toHaveAccessibleName(/任务队列/)
    expect(screen.getByTestId('queue-scroll-region')).toHaveAttribute('tabindex', '0')
    expect(screen.getByTestId('queue-page-content'))
      .toHaveClass('px-page', 'py-section')
    expect(screen.getByTestId('queue-page-content'))
      .not.toHaveClass('overflow-y-auto', 'overflow-hidden')
    expect(screen.getByRole('heading', { level: 1 }).closest('.ui-page-header'))
      .not.toHaveClass('sticky')

    await waitFor(() => expect(screen.getByRole('heading', { level: 3, name: /进行中/ })).toBeInTheDocument())
    expect(screen.getByRole('heading', { level: 3, name: /进行中/ }))
      .toHaveClass('type-section-label')
    const activeGuide = screen.getByTestId('queue-task-section-header-active')
    expect(activeGuide).toHaveClass('ui-queue-task-grid', 'ui-queue-section-header')
    expect(within(activeGuide).getByText('类型')).toHaveClass('ui-queue-column-label')
    expect(within(activeGuide).getByText('状态')).toHaveClass('ui-queue-column-label')
    expect(within(activeGuide).getByText('进度 / 结果')).toHaveClass('ui-queue-column-label')
    expect(within(activeGuide).getByText('时间')).toHaveClass('ui-queue-task-timing')
    expect(within(activeGuide).getByText('操作')).toHaveClass('ui-queue-column-label')
    expect(screen.getByRole('heading', { level: 3, name: /进行中/ }).closest('section'))
      .toHaveClass('gap-related')
    expect(screen.getByRole('heading', { level: 3, name: /等待入队/ }))
      .toHaveClass('type-section-label')
    expect(screen.getByRole('heading', { level: 3, name: /历史/ }))
      .toHaveClass('type-section-label')
    expect(screen.getByTestId('queue-task-grid-10'))
      .toHaveClass('ui-queue-task-grid')
    expect(screen.getByTestId('queue-task-grid-10').querySelector('.ui-queue-task-timing'))
      .toBeInTheDocument()
    // 历史 total=25 > page_size=20 → 分页器 + 页码指示
    expect(screen.getByText(/第 1 \/ 2 页/)).toBeInTheDocument()
    expect(screen.getByTestId('history-prev')).toBeDisabled()
    expect(screen.getByTestId('history-next')).not.toBeDisabled()
    expect(screen.getByRole('combobox', { name: '每页任务数' })).toHaveValue('20')
    const pagination = screen.getByTestId('queue-pagination')
    expect(pagination).toHaveClass('shrink-0', 'px-page', 'border-t')
    expect(pagination).not.toHaveClass('mt-section', '-mx-page', '-mb-page')
    expect(screen.getByTestId('queue-scroll-region').nextElementSibling)
      .toBe(pagination)
    historySpy.mockClear()

    // 点下一页 → 以 page=2 重新请求后端
    fireEvent.click(screen.getByTestId('history-next'))
    await waitFor(() =>
      expect(historySpy).toHaveBeenCalledWith(expect.objectContaining({ page: 2 })),
    )
  })

  it('输入搜索 → 防抖后带 q 请求后端（live + history）', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    const liveSpy = vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    const historySpy = vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderQueue()
    await waitFor(() => expect(screen.getByTestId('queue-filter-toggle')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('queue-filter-toggle'))
    expect(screen.getByTestId('queue-filterbar'))
      .toHaveClass('list-toolbar')
    expect(screen.getByTestId('queue-filterbar'))
      .toHaveAttribute('role', 'region')
    expect(screen.getByTestId('queue-filter-toggle'))
      .toHaveAttribute('aria-controls', 'queue-tasks-list-toolbar')
    expect(screen.getByTestId('queue-search'))
      .toHaveClass('form-control', 'form-control-sm', 'form-control-surface')
    fireEvent.change(screen.getByTestId('queue-search'), { target: { value: 'abc' } })

    await waitFor(
      () => expect(historySpy).toHaveBeenCalledWith(expect.objectContaining({ q: 'abc' })),
      { timeout: 2000 },
    )
    // 默认类型过滤是 train，搜索时随行带上。
    expect(liveSpy).toHaveBeenCalledWith('abc', 'train', 'exclusive')
  })

  it('选类型过滤 → 带 type 请求后端（live + history）', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    const liveSpy = vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    const historySpy = vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderQueue()
    await waitFor(() => expect(screen.getByTestId('queue-filter-toggle')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('queue-filter-toggle'))
    fireEvent.change(screen.getByTestId('queue-type-filter'), { target: { value: 'generate' } })

    await waitFor(
      () => expect(historySpy).toHaveBeenCalledWith(expect.objectContaining({ type: 'generate' })),
    )
    expect(liveSpy).toHaveBeenCalledWith(undefined, 'generate', 'exclusive')
  })

  it('默认类型过滤为训练；generate 行有「出图结果」跳转按钮', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    const liveSpy = vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeTask({ id: 88, name: 'gen', status: 'running', started_at: 1000, task_type: 'generate' }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderQueue()
    // 默认按训练过滤（第二个参数）
    await waitFor(() => expect(liveSpy).toHaveBeenCalledWith(undefined, 'train', 'exclusive'))
    // generate 行渲染跳转按钮（mock 忽略过滤参数，照返 generate 任务）
    await waitFor(() => expect(screen.getByTestId('jump-btn-88')).toBeInTheDocument())
  })

  it('scheduled 任务渲染独立「计划任务」分区，带立即开始/取消计划按钮', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeTask({ id: 20, name: 'pend', status: 'pending', started_at: null, pid: null }),
      makeTask({
        id: 21, name: 'sched', status: 'scheduled', started_at: null, pid: null,
        scheduled_at: Date.now() / 1000 + 3600,
      }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderQueue()

    await waitFor(() =>
      expect(screen.getByTestId('queue-scheduled-section')).toBeInTheDocument(),
    )
    expect(screen.getByText(/计划任务/)).toBeInTheDocument()
    // scheduled 行有专属操作；pending 行只提供取消
    expect(screen.getByTestId('startnow-btn-21')).toBeInTheDocument()
    expect(screen.getByTestId('cancel-scheduled-btn-21')).toBeInTheDocument()
    expect(screen.queryByTestId('startnow-btn-20')).not.toBeInTheDocument()
    expect(screen.getByTestId('cancel-pending-btn-20')).toBeInTheDocument()
  })

  it('GPU pending 行取消先说明不会影响运行任务，确认后调 cancelTask', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeTask({ id: 20, name: 'pend', status: 'pending', started_at: null, pid: null }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })
    const cancelSpy = vi.spyOn(api, 'cancelTask').mockResolvedValue({ task_id: 20, canceled: true })

    renderQueue()
    fireEvent.click(await screen.findByTestId('cancel-pending-btn-20'))
    const dialog = await screen.findByRole('alertdialog')
    expect(dialog).toHaveTextContent('任务尚未开始，不会影响当前运行中的任务')
    expect(cancelSpy).not.toHaveBeenCalled()
    fireEvent.click(within(dialog).getByRole('button', { name: '取消任务' }))
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith(20))
  })

  it('点「立即开始」→ confirm 后调 startTaskNow', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeTask({
        id: 21, name: 'sched', status: 'scheduled', started_at: null, pid: null,
        scheduled_at: Date.now() / 1000 + 3600,
      }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })
    const startSpy = vi.spyOn(api, 'startTaskNow').mockResolvedValue({
      task_id: 21, status: 'pending',
    })

    renderQueue()
    await waitFor(() => expect(screen.getByTestId('startnow-btn-21')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('startnow-btn-21'))
    // confirm modal 弹出（行按钮已 icon 化，「立即开始」文案只在 dialog 确认键上）
    await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument())
    expect(startSpy).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('立即开始'))
    await waitFor(() => expect(startSpy).toHaveBeenCalledWith(21))
  })

  it('paused 行：恢复/取消操作在 action 列，恢复走 confirm 后调 resumeTask', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeTask({
        id: 40, name: 'paused-task', status: 'paused', pid: null,
        paused_step: 120, paused_at: 1200, is_resumable: true,
      }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })
    const resumeSpy = vi.spyOn(api, 'resumeTask').mockResolvedValue({
      task_id: 40, status: 'pending',
    })

    renderQueue()
    await waitFor(() => expect(screen.getByTestId('resume-btn-40')).toBeInTheDocument())
    expect(screen.getByTestId('cancel-paused-btn-40')).toBeInTheDocument()
    fireEvent.click(screen.getByTestId('resume-btn-40'))
    // 状态转移走 confirm（行按钮已 icon 化，「恢复」文案只在 dialog 确认键上）
    await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument())
    expect(resumeSpy).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('恢复'))
    await waitFor(() => expect(resumeSpy).toHaveBeenCalledWith(40))
  })

  it('失败/取消且恢复点在盘的历史行：继续训练在 action 列，confirm 后调 resumeTask', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [
        makeTask({
          id: 30, name: 'dead', status: 'canceled', pid: null,
          finished_at: 900, is_resumable: true,
        }),
        makeTask({
          id: 31, name: 'dead-no-backup', status: 'canceled', pid: null,
          finished_at: 800,
        }),
      ],
      total: 2, page: 1, page_size: 20,
    })
    const resumeSpy = vi.spyOn(api, 'resumeTask').mockResolvedValue({
      task_id: 30, status: 'pending',
    })

    renderQueue()
    await waitFor(() => expect(screen.getByTestId('resume-btn-30')).toBeInTheDocument())
    // 无恢复点的终态行没有继续训练按钮
    expect(screen.queryByTestId('resume-btn-31')).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId('resume-btn-30'))
    await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument())
    expect(resumeSpy).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('继续训练'))
    await waitFor(() => expect(resumeSpy).toHaveBeenCalledWith(30))
  })

  it('GPU / 数据任务使用共享双选切换，切换后漏斗变 kind 过滤（P-G）', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeTask({ id: 10, name: 'run', status: 'running', started_at: 1000 }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })
    vi.spyOn(api, 'listProjects').mockResolvedValue([])

    renderQueue()
    await waitFor(() => expect(screen.getByText(/进行中/)).toBeInTheDocument())

    const viewSwitcher = screen.getByRole('radiogroup', { name: '队列视图' })
    const tasksOption = within(viewSwitcher).getByRole('radio', { name: 'GPU 任务' })
    const jobsOption = within(viewSwitcher).getByRole('radio', { name: '数据任务' })
    expect(tasksOption).toHaveAttribute('aria-checked', 'true')
    expect(jobsOption).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByRole('button', { name: '刷新' }).nextElementSibling).toBe(viewSwitcher)
    const gpuActions = Array.from(viewSwitcher.parentElement?.children ?? [])
    expect(gpuActions.indexOf(screen.getByRole('button', { name: '取消当前任务' })))
      .toBeLessThan(gpuActions.indexOf(screen.getByTestId('queue-hold-btn')))
    expect(gpuActions.indexOf(screen.getByTestId('queue-hold-btn')))
      .toBeLessThan(gpuActions.indexOf(screen.getByTestId('queue-filter-toggle')))
    expect(gpuActions.indexOf(screen.getByTestId('queue-filter-toggle')))
      .toBeLessThan(gpuActions.indexOf(screen.getByRole('button', { name: '刷新' })))

    fireEvent.click(jobsOption)
    await waitFor(() => expect(screen.getByTestId('data-jobs-panel')).toBeInTheDocument())
    expect(tasksOption).toHaveAttribute('aria-checked', 'false')
    expect(jobsOption).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('button', { name: '刷新' }).nextElementSibling).toBe(viewSwitcher)
    const dataActions = Array.from(viewSwitcher.parentElement?.children ?? [])
    expect(dataActions.indexOf(screen.getByTestId('queue-hold-btn')))
      .toBeLessThan(dataActions.indexOf(screen.getByTestId('queue-filter-toggle')))
    expect(dataActions.indexOf(screen.getByTestId('queue-filter-toggle')))
      .toBeLessThan(dataActions.indexOf(screen.getByRole('button', { name: '刷新' })))
    // 任务分区没了；漏斗还在（数据作业视图的 kind 过滤），点开出 kind select
    expect(screen.queryByText(/等待入队/)).not.toBeInTheDocument()
    expect(screen.getByTestId('queue-filter-toggle'))
      .toHaveAttribute('aria-controls', 'queue-jobs-list-toolbar')
    expect(screen.getByTestId('queue-jobs-filterbar')).toHaveAttribute('hidden')
    fireEvent.click(screen.getByTestId('queue-filter-toggle'))
    expect(screen.getByTestId('queue-jobs-filterbar')).not.toHaveAttribute('hidden')
    expect(screen.getByTestId('jobs-kind-filter')).toBeInTheDocument()
    expect(screen.getByTestId('jobs-kind-filter'))
      .toHaveClass('form-control', 'form-control-sm', 'form-control-surface')
    expect(screen.getByTestId('jobs-search')).toBeInTheDocument()
    // 任务视图专属的搜索框不在
    expect(screen.queryByTestId('queue-search')).not.toBeInTheDocument()

    // 切回 GPU 任务视图
    fireEvent.click(tasksOption)
    await waitFor(() => expect(screen.getByText(/进行中/)).toBeInTheDocument())
  })

  it('过滤行默认收起，点漏斗才显示搜索框（与项目页一致）', async () => {
    vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false } as never)
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeTask({ id: 10, name: 'run', status: 'running', started_at: 1000 }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderQueue()

    await waitFor(() => expect(screen.getByTestId('queue-filter-toggle')).toBeInTheDocument())
    const toggle = screen.getByTestId('queue-filter-toggle')
    const toolbar = screen.getByTestId('queue-filterbar')
    expect(toggle).toHaveAttribute('aria-controls', 'queue-tasks-list-toolbar')
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(toolbar).toHaveAttribute('hidden')
    expect(screen.getByTestId('queue-search')).not.toBeVisible()
    // 点漏斗 → 同一个具名 region 展开，控件恢复可见
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(toolbar).not.toHaveAttribute('hidden')
    expect(screen.getByTestId('queue-search')).toBeVisible()
  })
})
