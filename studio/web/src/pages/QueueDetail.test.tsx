/** QueueDetail page 组件级 regression test。
 *
 *  目前只覆盖 SnapshotConfigTab 的 refetch trap：父组件每 2s 浅 clone task
 *  做 elapsed time tick，旧实现 [task] 作 deps 会让 snapshot config 也跟着
 *  2s 重拉 —— 浏览器卡顿、loading flash。 */
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../components/Dialog'
import { ToastProvider } from '../components/Toast'
import type { Task } from '../api/client'
import i18n from '../i18n'
import QueueDetailPage, { OutputsTab, SnapshotConfigTab } from './QueueDetail'

const SNAPSHOT_URL_PREFIX = '/api/queue/'
const SNAPSHOT_URL_SUFFIX = '/snapshot/config'

const fetchMock = vi.fn()

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 1, name: 'train', config_name: 'train',
    status: 'running', priority: 0,
    created_at: 1000, started_at: 1100, finished_at: null,
    pid: 1234, exit_code: null, output_dir: null, error_msg: null,
    ...overrides,
  }
}

function snapshotResponse() {
  const body = { yaml: 'key: val\n', config: { key: 'val' } }
  return {
    ok: true, status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
    headers: new Headers({ 'content-type': 'application/json' }),
  } as Response
}

function snapshotCallCount(): number {
  return fetchMock.mock.calls.filter(([url]) =>
    typeof url === 'string'
    && url.startsWith(SNAPSHOT_URL_PREFIX)
    && url.endsWith(SNAPSHOT_URL_SUFFIX),
  ).length
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string) => {
    if (url.startsWith(SNAPSHOT_URL_PREFIX) && url.endsWith(SNAPSHOT_URL_SUFFIX)) {
      return Promise.resolve(snapshotResponse())
    }
    return Promise.resolve({
      ok: false, status: 404, json: async () => null, text: async () => '',
      headers: new Headers(),
    } as Response)
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function setup(task: Task | null) {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <SnapshotConfigTab task={task} />
      </ToastProvider>
    </MemoryRouter>
  )
}

describe('SnapshotConfigTab', () => {
  it('父组件 2s 浅 clone task 不会触发重拉 — snapshot 是不可变的', async () => {
    const task = makeTask()
    const view = setup(task)

    await waitFor(() => expect(snapshotCallCount()).toBe(1))

    // 模拟父组件的 2s tick：shallow clone 出新引用，id / started_at 不变
    for (let i = 0; i < 5; i++) {
      view.rerender(
        <MemoryRouter>
          <ToastProvider>
            <SnapshotConfigTab task={{ ...task }} />
          </ToastProvider>
        </MemoryRouter>
      )
    }

    // 等一下让任何额外 useEffect 走完
    await new Promise((r) => setTimeout(r, 20))
    expect(snapshotCallCount()).toBe(1)
  })

  it('pending → running 转换（started_at null→number）触发一次重拉', async () => {
    const view = setup(makeTask({ status: 'pending', started_at: null }))
    await waitFor(() => expect(snapshotCallCount()).toBe(1))

    view.rerender(
      <MemoryRouter>
        <ToastProvider>
          <SnapshotConfigTab task={makeTask({ status: 'running', started_at: 1234 })} />
        </ToastProvider>
      </MemoryRouter>
    )

    await waitFor(() => expect(snapshotCallCount()).toBe(2))
  })
})

describe('OutputsTab 响应式输出表', () => {
  it('为两个横向滚动区提供独立名称，并披露被截断的完整目录', async () => {
    const outputDir = 'C:/very/long/output/path'
    const body = {
      task_id: 119,
      output_dir: outputDir,
      exists: true,
      supports_open_folder: false,
      archive_basename: 'project-version',
      files: [
        { name: 'model.safetensors', path: 'model.safetensors', size: 100, mtime: 2, kind: 'lora', is_lora: true },
        { name: 'state.pt', path: 'state.pt', size: 200, mtime: 1, kind: 'training_state', is_lora: false },
      ],
    }
    fetchMock.mockResolvedValue({
      ok: true, status: 200,
      json: async () => body, text: async () => JSON.stringify(body),
      headers: new Headers({ 'content-type': 'application/json' }),
    } as Response)

    render(
      <MemoryRouter>
        <ToastProvider>
          <DialogProvider>
            <OutputsTab taskId={119} />
          </DialogProvider>
        </ToastProvider>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('region', { name: '输出文件' }))
      .toHaveClass('ui-queue-output-table')
    expect(screen.getByRole('region', { name: '训练状态' }))
      .toHaveClass('ui-queue-output-table')
    expect(screen.getByText(outputDir, { selector: 'code' }))
      .toHaveAttribute('title', outputDir)
  })
})

// ── 暂停按钮的 SSE 刷新（QueueDetailPage header）─────────────────────────────
//
// regression：恢复 / 启动后 is_pausable 由 train_loop_started + auto_epoch_backup_written
// 翻 true，但 header 共享的 task 之前只在 task_state_changed 时 reload，漏听这两个
// 事件 → 暂停按钮一直不出现，必须切到 /queue 再回来（整页重挂）才有。
//
// jsdom 默认没有 EventSource（useEventStream 内部 typeof 守卫会短路），这里塞个
// fake 让 hook 真订阅，再手动驱动一条事件验证组件会重新 getTask 并显示按钮。
class FakeEventSource {
  static instances: FakeEventSource[] = []
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  constructor(public url: string) { FakeEventSource.instances.push(this) }
  close(): void { this.readyState = 2 }
  emit(evt: unknown): void { this.onmessage?.({ data: JSON.stringify(evt) }) }
}

const QUEUE_ITEM_URL = '/api/queue/119'

function queueItemResponse(task: Task) {
  return {
    ok: true, status: 200,
    json: async () => task,
    text: async () => JSON.stringify(task),
    headers: new Headers({ 'content-type': 'application/json' }),
  } as Response
}

function renderDetailPage() {
  return render(
    <MemoryRouter initialEntries={['/queue/119']}>
      <ToastProvider>
        <Routes>
          <Route path="/queue/:id" element={<QueueDetailPage />} />
        </Routes>
      </ToastProvider>
    </MemoryRouter>
  )
}

function getTaskCalls(): number {
  return fetchMock.mock.calls.filter(([u]) => u === QUEUE_ITEM_URL).length
}

describe('QueueDetailPage 完整参数', () => {
  it('详情展示超过200字符的参数全文', async () => {
    const value = `${'长参数'.repeat(100)} END-OF-PARAM`
    fetchMock.mockImplementation((url: string) => url === QUEUE_ITEM_URL
      ? Promise.resolve(queueItemResponse(makeTask({ id: 119, task_type: 'tag', status: 'done', params_decoded: { tag: value } })))
      : Promise.resolve(new Response('', { status: 404 })))
    renderDetailPage()
    expect(await screen.findByText(value)).toHaveClass('break-all')
  })
})

describe('QueueDetailPage 加载失败恢复', () => {
  beforeEach(() => {
    FakeEventSource.instances = []
    vi.stubGlobal('EventSource', FakeEventSource)
  })

  afterEach(async () => {
    await act(async () => { await i18n.changeLanguage('zh') })
  })

  it.each([
    { language: 'zh', title: '任务详情加载失败', reload: '重新加载', loading: '加载中...' },
    { language: 'en', title: 'Failed to load task details', reload: 'Reload', loading: 'Loading...' },
  ])('$language：加载失败停止显示加载中，重复失败后仍能重新加载且不重跑任务', async (copy) => {
    await i18n.changeLanguage(copy.language)
    const user = userEvent.setup()
    let settle!: { resolve: (response: Response) => void; reject: (error: Error) => void }
    fetchMock.mockImplementation((url: string) => url === QUEUE_ITEM_URL
      ? new Promise<Response>((resolve, reject) => { settle = { resolve, reject } })
      : Promise.resolve(new Response('', { status: 404 })))
    renderDetailPage()
    expect(screen.getByText(copy.loading)).toBeInTheDocument()

    await act(async () => { settle.reject(new Error('network offline')) })
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(copy.title)
    expect(alert).toHaveTextContent('network offline')
    expect(screen.queryByText(copy.loading)).not.toBeInTheDocument()
    const reload = within(alert).getByRole('button', { name: copy.reload })
    expect(reload).toBeEnabled()

    await user.click(reload)
    expect(getTaskCalls()).toBe(2)
    expect(reload).toBeDisabled()
    expect(reload).toHaveAttribute('aria-busy', 'true')
    await user.click(reload)
    expect(getTaskCalls()).toBe(2)
    await act(async () => {
      settle.resolve(new Response(JSON.stringify({ error: { message: 'Service unavailable' } }), {
        status: 503, headers: { 'content-type': 'application/json' },
      }))
    })
    await waitFor(() => expect(reload).toBeEnabled())
    expect(alert).toHaveTextContent('Service unavailable')
    expect(screen.queryByText(copy.loading)).not.toBeInTheDocument()

    await user.click(reload)
    await act(async () => { settle.resolve(queueItemResponse(makeTask({ id: 119, name: 'Recovered task', status: 'done' }))) })
    expect(await screen.findByTitle('Recovered task')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(getTaskCalls()).toBe(3)
    expect(fetchMock.mock.calls.every(([, options]) => !options?.method || options.method === 'GET')).toBe(true)
  })

  it('已加载详情刷新失败时保留内容，重新加载成功后清除错误', async () => {
    const user = userEvent.setup()
    let fail = false
    fetchMock.mockImplementation((url: string) => url === QUEUE_ITEM_URL
      ? fail
        ? Promise.reject(new Error('refresh offline'))
        : Promise.resolve(queueItemResponse(makeTask({ id: 119, name: 'Keep this task', status: 'done' })))
      : Promise.resolve(new Response('', { status: 404 })))
    renderDetailPage()
    await screen.findByTitle('Keep this task')

    fail = true
    act(() => {
      FakeEventSource.instances[FakeEventSource.instances.length - 1].emit({ type: 'task_state_changed', task_id: 119 })
    })
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('refresh offline')
    expect(screen.getByTitle('Keep this task')).toBeInTheDocument()
    expect(screen.queryByText('加载中...')).not.toBeInTheDocument()

    fail = false
    await user.click(within(alert).getByRole('button', { name: '重新加载' }))
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
    expect(screen.getByTitle('Keep this task')).toBeInTheDocument()
    expect(getTaskCalls()).toBe(3)
  })
})

describe('QueueDetailPage 重试状态恢复', () => {
  function mockRetry(retryResponse: () => Promise<Response>, taskType: 'train' | 'generate' = 'train') {
    fetchMock.mockImplementation((url: string) => {
      if (url === `${QUEUE_ITEM_URL}/retry`) return retryResponse()
      if (url === QUEUE_ITEM_URL) {
        return Promise.resolve(queueItemResponse(makeTask({
          id: 119, task_type: taskType, status: 'failed', finished_at: 1200,
          is_resumable: taskType === 'train',
        })))
      }
      if (url === '/api/queue/120') {
        return Promise.resolve(queueItemResponse(makeTask({
          id: 120, name: 'new task', task_type: taskType, status: 'pending', started_at: null,
        })))
      }
      return Promise.resolve(new Response('', { status: 404 }))
    })
  }

  it.each([
    { taskType: 'train', label: '重新训练' },
    { taskType: 'generate', label: '重试' },
  ] as const)('$label 请求失败后释放按钮，并允许再次重试', async ({ taskType, label }) => {
    const user = userEvent.setup()
    let rejectRetry!: (reason: Error) => void
    const retryResponse = vi.fn<() => Promise<Response>>()
      .mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectRetry = reject }))
      .mockResolvedValueOnce(queueItemResponse(makeTask({ id: 120, status: 'pending' })))
    mockRetry(retryResponse, taskType)
    renderDetailPage()

    const retryButton = await screen.findByRole('button', { name: label })
    const deleteButton = screen.getByRole('button', { name: '删除记录' })
    await user.click(retryButton)
    expect(retryResponse).toHaveBeenCalledTimes(1)
    expect(retryButton).toBeDisabled()
    expect(deleteButton).toBeDisabled()
    if (taskType === 'train') expect(screen.getByTestId('detail-resume-btn')).toBeDisabled()
    await user.click(retryButton)
    expect(retryResponse).toHaveBeenCalledTimes(1)

    await act(async () => { rejectRetry(new Error('retry offline')) })
    expect(await screen.findByText('Error: retry offline')).toBeInTheDocument()
    await waitFor(() => expect(retryButton).toBeEnabled())
    expect(deleteButton).toBeEnabled()
    if (taskType === 'train') expect(screen.getByTestId('detail-resume-btn')).toBeEnabled()

    await user.click(retryButton)
    expect(retryResponse).toHaveBeenCalledTimes(2)
    expect(await screen.findByRole('heading', { name: '#120 new task', level: 1 })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: '取消任务' })).toBeEnabled())
  })

  it('首次重试成功后跳到新任务，且新任务操作不继承 busy', async () => {
    const user = userEvent.setup()
    const retryResponse = vi.fn().mockResolvedValue(queueItemResponse(makeTask({ id: 120, status: 'pending' })))
    mockRetry(retryResponse)
    renderDetailPage()

    await user.click(await screen.findByRole('button', { name: '重新训练' }))

    expect(retryResponse).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(`${QUEUE_ITEM_URL}/retry`, expect.objectContaining({ method: 'POST' }))
    expect(await screen.findByRole('heading', { name: '#120 new task', level: 1 })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: '取消任务' })).toBeEnabled())
  })
})

describe('QueueDetailPage 删除范围提示', () => {
  afterEach(async () => {
    await act(async () => { await i18n.changeLanguage('zh') })
  })

  const languages = [
    {
      language: 'zh', open: '删除记录', title: '删除任务记录', cancel: '取消',
      common: '同时删除本任务目录中的日志、监控、配置快照、样图，以及本任务的恢复点。此操作无法撤销。',
      train: '版本级 LoRA 训练产物不受影响。',
      generate: '还会删除本次生成的图片、XY 输出目录和临时缓存。',
    },
    {
      language: 'en', open: 'Delete record', title: 'Delete task record', cancel: 'Cancel',
      common: "This also deletes the logs, monitor data, config snapshots and samples in this task's folder, along with its recovery checkpoints. This cannot be undone.",
      train: 'Version-level LoRA training outputs are not affected.',
      generate: 'This also deletes the images, XY output folders and temporary cache from this render task.',
    },
  ] as const

  for (const copy of languages) {
    it.each(['train', 'generate', 'tag'] as const)(`${copy.language}：%s 的删除提示准确，取消不删除任务`, async (taskType) => {
      await i18n.changeLanguage(copy.language)
      const user = userEvent.setup()
      fetchMock.mockImplementation((url: string) => url === QUEUE_ITEM_URL
        ? Promise.resolve(queueItemResponse(makeTask({
          id: 119, name: 'example-task-119', task_type: taskType, status: 'done', finished_at: 1200,
        })))
        : Promise.resolve(new Response('', { status: 404 })))
      renderDetailPage()

      const opener = await screen.findByRole('button', { name: copy.open })
      await user.click(opener)
      const panel = screen.getByRole('alertdialog', { name: copy.title })
      expect(panel).toHaveAttribute('aria-modal', 'true')
      expect(panel).toHaveTextContent('#119 example-task-119')
      expect(panel).toHaveTextContent(copy.common)
      expect(panel.textContent!.includes(copy.train)).toBe(taskType === 'train')
      expect(panel.textContent!.includes(copy.generate)).toBe(taskType === 'generate')
      const deleteCalls = () => fetchMock.mock.calls.filter(([, options]) => options?.method === 'DELETE')
      expect(deleteCalls()).toHaveLength(0)

      const cancel = within(panel).getByRole('button', { name: copy.cancel })
      await waitFor(() => expect(cancel).toHaveFocus())
      await user.tab({ shift: true })
      expect(within(panel).getByRole('button', { name: copy.language === 'zh' ? '删除' : 'Delete' })).toHaveFocus()
      await user.tab()
      expect(cancel).toHaveFocus()
      await user.click(cancel)
      expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
      expect(opener).toHaveFocus()
      await user.click(opener)
      await user.keyboard('{Escape}')
      expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
      expect(opener).toHaveFocus()
      expect(screen.getByRole('heading', { name: '#119 example-task-119', level: 1 })).toBeInTheDocument()
      expect(deleteCalls()).toHaveLength(0)
    })
  }
})

describe('QueueDetailPage 确认框忙碌保护', () => {
  afterEach(async () => { await act(async () => { await i18n.changeLanguage('zh') }) })

  it('删除请求期间禁止重复提交和Escape/遮罩关闭，失败后恢复入口', async () => {
    const user = userEvent.setup()
    let rejectDelete!: (error: Error) => void
    fetchMock.mockImplementation((url: string, options?: RequestInit) => {
      if (options?.method === 'DELETE') return new Promise((_resolve, reject) => { rejectDelete = reject })
      return Promise.resolve(url === QUEUE_ITEM_URL
        ? queueItemResponse(makeTask({ id: 119, status: 'done' }))
        : new Response('', { status: 404 }))
    })
    renderDetailPage()
    const opener = await screen.findByRole('button', { name: '删除记录' })
    await user.click(opener)
    const dialog = screen.getByRole('alertdialog', { name: '删除任务记录' })
    const confirm = within(dialog).getByRole('button', { name: '删除' })
    await user.click(confirm)
    expect(confirm).toBeDisabled()
    expect(confirm).toHaveAttribute('aria-busy', 'true')
    expect(within(dialog).getByRole('button', { name: '取消' })).toBeDisabled()
    await user.click(confirm)
    await user.keyboard('{Escape}')
    await user.click(screen.getByTestId('queue-detail-confirm'))
    expect(dialog).toBeInTheDocument()
    expect(fetchMock.mock.calls.filter(([, opts]) => opts?.method === 'DELETE')).toHaveLength(1)
    await act(async () => { rejectDelete(new Error('delete offline')) })
    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument())
    expect(opener).toBeEnabled()
    expect(opener).toHaveFocus()
  })

  it.each(['zh', 'en'])('%s：快照确认键盘退出、失败重试与成功提交', async (lang) => {
    await i18n.changeLanguage(lang)
    const user = userEvent.setup()
    let settle!: { resolve: (response: Response) => void; reject: (error: Error) => void }
    fetchMock.mockImplementation((url: string, options?: RequestInit) => {
      if (options?.method === 'PUT') return new Promise<Response>((resolve, reject) => { settle = { resolve, reject } })
      return Promise.resolve(url.endsWith('/snapshot/config') ? snapshotResponse() : new Response('', { status: 404 }))
    })
    setup(makeTask({ id: 119, status: 'done', project_id: 7, version_id: 9 }))
    const label = i18n.t('snapshot.applyBtn')
    const title = i18n.t('snapshot.applyConfirmTitle')
    const opener = await screen.findByRole('button', { name: label })
    await user.click(opener)
    let dialog = screen.getByRole('alertdialog', { name: title })
    const cancel = within(dialog).getByRole('button', { name: lang === 'zh' ? '取消' : 'Cancel' })
    await waitFor(() => expect(cancel).toHaveFocus())
    await user.tab({ shift: true })
    expect(within(dialog).getByRole('button', { name: label })).toHaveFocus()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    expect(opener).toHaveFocus()
    expect(fetchMock.mock.calls.filter(([, opts]) => opts?.method === 'PUT')).toHaveLength(0)

    await user.click(opener)
    dialog = screen.getByRole('alertdialog', { name: title })
    const confirm = within(dialog).getByRole('button', { name: label })
    await user.click(confirm)
    expect(confirm).toBeDisabled()
    await user.keyboard('{Escape}')
    await user.click(screen.getByTestId('queue-detail-confirm'))
    expect(dialog).toBeInTheDocument()
    await act(async () => { settle.reject(new Error('apply offline')) })
    await waitFor(() => expect(confirm).toBeEnabled())
    expect(dialog).toBeInTheDocument()
    await user.click(confirm)
    await act(async () => { settle.resolve(new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } })) })
    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument())
    expect(fetchMock.mock.calls.filter(([, opts]) => opts?.method === 'PUT')).toHaveLength(2)
  })
})

describe('QueueDetailPage Overview 语义排版', () => {
  it('按任务、时间与技术信息分组，长路径跨列且保留全部字段', async () => {
    const configPath = 'G:/AnimaLoraStudio/studio_data/projects/67-yu-hydra/versions/K2_v1/config.yaml'
    const monitorPath = 'G:/AnimaLoraStudio/studio_data/tasks/119/monitor/state.json'
    fetchMock.mockImplementation((url: string) => url === QUEUE_ITEM_URL
      ? Promise.resolve(queueItemResponse(makeTask({
        id: 119,
        name: 'yu-hydra_K2_v1',
        config_name: 'k2_0823',
        status: 'failed',
        priority: 3,
        scheduled_at: 1050,
        started_at: 1100,
        finished_at: 4700,
        exit_code: 1,
        pid: null,
        project_id: 67,
        version_id: 111,
        config_path: configPath,
        monitor_state_path: monitorPath,
        error_msg: 'supervisor restart while task was running',
      })))
      : Promise.resolve(new Response('', { status: 404 })))

    renderDetailPage()

    const overview = await screen.findByTestId('queue-overview-grid')
    expect(overview).toHaveClass('ui-queue-overview-grid')
    expect(within(overview).getByRole('heading', { name: '任务与状态', level: 2 }))
      .toHaveClass('type-section-label')
    expect(within(overview).getByRole('heading', { name: '时间', level: 2 }))
      .toHaveClass('type-section-label')
    expect(within(overview).getByRole('heading', { name: '来源与技术信息', level: 2 }))
      .toHaveClass('type-section-label')

    const taskGroup = within(overview).getByTestId('queue-overview-group-task')
    expect(within(taskGroup).getByText('名称')).toHaveClass('type-data-label')
    expect(within(taskGroup).getByText('yu-hydra_K2_v1')).not.toHaveClass('font-mono')
    expect(within(taskGroup).getByText('k2_0823.yaml')).toHaveClass('font-mono')

    const configField = within(overview).getByTestId('queue-overview-field-config-path')
    expect(configField).toHaveClass('ui-queue-overview-field--wide')
    expect(within(configField).getByText(configPath)).toHaveClass('font-mono', 'break-all')
    expect(within(overview).getByText(monitorPath)).toBeInTheDocument()
    expect(within(overview).getByText('supervisor restart while task was running')).toBeInTheDocument()
    expect(within(overview).getByRole('link', { name: '项目 #67 / v#111' })).toBeInTheDocument()
  })
})

describe('QueueDetailPage 暂停按钮 SSE 刷新', () => {
  beforeEach(() => {
    FakeEventSource.instances = []
    vi.stubGlobal('EventSource', FakeEventSource)
  })

  it('auto_epoch_backup_written 事件触发重拉 → 暂停按钮出现', async () => {
    // getTask：首拉 is_pausable=false（train loop / 首个 epoch backup 未就绪），
    // 之后拉 is_pausable=true（首个 epoch backup 已落盘）。
    let pausable = false
    fetchMock.mockImplementation((url: string) => {
      if (url === QUEUE_ITEM_URL) {
        return Promise.resolve(queueItemResponse(makeTask({
          id: 119, status: 'running', is_pausable: pausable,
          started_at: Math.floor(Date.now() / 1000) - 125,
        })))
      }
      return Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    renderDetailPage()

    // 运行摘要进入紧凑身份栏，不再用四张常驻卡挤压日志/监控空间。
    await waitFor(() => expect(screen.getByText('取消任务')).toBeInTheDocument())
    expect(screen.queryByTestId('queue-detail-stats')).not.toBeInTheDocument()
    const header = screen.getByTestId('queue-detail-header')
    expect(within(header).getByRole('heading', { name: '#119 train', level: 1 }))
      .toHaveClass('ui-queue-detail-heading')
    expect(within(header).getByText('train.yaml'))
      .toHaveClass('ui-queue-detail-config')
    expect(within(header).getByTestId('queue-detail-running-duration'))
      .toHaveTextContent(/^· 2m/)
    expect(screen.queryByRole('link', { name: '← 队列' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '取消任务' })).toHaveClass('btn-warn', 'btn-sm')
    for (const statusBadge of screen.getAllByText('运行中')) {
      expect(statusBadge).toHaveClass('badge-accent')
    }
    expect(screen.queryByTestId('detail-pause-btn')).not.toBeInTheDocument()

    // 后端首个 epoch backup 落盘 → is_pausable 升级；推一条 SSE 事件
    pausable = true
    await waitFor(() => expect(FakeEventSource.instances.length).toBeGreaterThan(0))
    FakeEventSource.instances[0].emit({ type: 'auto_epoch_backup_written', task_id: 119 })

    // 重拉后暂停按钮出现（不依赖切页重挂）
    await waitFor(() => expect(screen.getByTestId('detail-pause-btn')).toBeInTheDocument())
  })

  it('其它 task 的事件不会触发重拉', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url === QUEUE_ITEM_URL) {
        return Promise.resolve(queueItemResponse(makeTask({
          id: 119, status: 'running', is_pausable: false,
        })))
      }
      return Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    renderDetailPage()
    await waitFor(() => expect(screen.getByText('取消任务')).toBeInTheDocument())
    const before = getTaskCalls()

    await waitFor(() => expect(FakeEventSource.instances.length).toBeGreaterThan(0))
    // 别的 task（task_id=999）的 backup 事件 — 不应触发本页重拉
    FakeEventSource.instances[0].emit({ type: 'auto_epoch_backup_written', task_id: 999 })
    await new Promise((r) => setTimeout(r, 150))

    expect(getTaskCalls()).toBe(before)
  })
})

// ── P-H 类型差异化 tab ───────────────────────────────────────────────────────
describe('QueueDetailPage 类型差异化 tab（P-H）', () => {
  it('generate 任务只留 overview+log，隐藏 monitor/eval/snapshot，带查看出图结果深链', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url === QUEUE_ITEM_URL) {
        return Promise.resolve(queueItemResponse(makeTask({
          id: 119, task_type: 'generate', status: 'done', finished_at: 1200,
        })))
      }
      return Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    renderDetailPage()

    // task 加载后深链按钮出现（证明是 generate 且已 hydrate）
    await waitFor(() => expect(screen.getByTestId('detail-view-generate')).toBeInTheDocument())
    // 训练专用 tab 隐藏
    expect(screen.queryByText('监控')).not.toBeInTheDocument()
    expect(screen.queryByText('指标')).not.toBeInTheDocument()
    expect(screen.queryByText('关联配置')).not.toBeInTheDocument()
    expect(screen.queryByText('输出')).not.toBeInTheDocument()
    // overview tab 仍在，并与当前面板建立 ARIA 关联
    const tablist = screen.getByRole('tablist', { name: '任务详情分区' })
    const overview = screen.getByRole('tab', { name: '详情' })
    const log = screen.getByRole('tab', { name: '日志' })
    expect(tablist).toHaveClass('ui-selection-underline')
    expect(overview).toHaveAttribute('aria-selected', 'true')
    expect(overview).toHaveAttribute('aria-controls', 'queue-detail-panel')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', overview.id)

    const user = userEvent.setup()
    overview.focus()
    await user.keyboard('{ArrowRight}')
    expect(log).toHaveFocus()
    expect(log).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', log.id)

    // QueueDetail 把当前 tab 同步到 window.location.hash；切回概览，避免污染后续用例。
    await user.keyboard('{ArrowLeft}')
    expect(overview).toHaveAttribute('aria-selected', 'true')
  })
})

// ── 评估 tab（#465）──────────────────────────────────────────────────────────
describe('QueueDetailPage 评估 tab', () => {
  function mockTask(task: Task, sessions: unknown[]) {
    fetchMock.mockImplementation((url: string) => {
      if (url === QUEUE_ITEM_URL) return Promise.resolve(queueItemResponse(task))
      if (url.includes('/eval/sessions')) {
        const body = { sessions }
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => body, text: async () => JSON.stringify(body),
          headers: new Headers({ 'content-type': 'application/json' }),
        } as Response)
      }
      return Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })
  }

  it('评估作业：指标 / 样图分成两个 tab，且「关联训练」显示溯源', async () => {
    mockTask(
      makeTask({
        id: 119, task_type: 'eval_session', status: 'done',
        project_id: 3, version_id: 7, params_decoded: { session_id: 42 },
      }),
      [{ id: 42, task_id: 119, parent_task_id: 88, status: 'done', trigger: 'after_training' }],
    )
    renderDetailPage()

    // task=null 时按 train 基线暂时也会显示「样图」和「监控」；等待 monitor
    // 消失才能证明 eval_session 已加载且类型差异化 tab 已收敛。
    await waitFor(() => expect(screen.queryByText('监控')).not.toBeInTheDocument())
    expect(screen.getByText('样图')).toBeInTheDocument()
    expect(screen.getByText('指标')).toBeInTheDocument()
    // 溯源进了概览的键值表，不是单独一块进度卡。parent 是异步认领的（params 里
    // 只有 session_id），所以等它出现而不是等「关联训练」那一格。
    await waitFor(() => expect(screen.getByText('#88')).toBeInTheDocument())
    expect(screen.getByText('关联训练')).toBeInTheDocument()
  })

  it('手动发起的评估没有关联训练 → n/a，不藏起来', async () => {
    mockTask(
      makeTask({
        id: 119, task_type: 'eval_session', status: 'done',
        project_id: 3, version_id: 7, params_decoded: { session_id: 42 },
      }),
      [{ id: 42, task_id: 119, parent_task_id: null, status: 'done', trigger: 'manual' }],
    )
    renderDetailPage()

    await waitFor(() => expect(screen.getByText('关联训练')).toBeInTheDocument())
    expect(screen.getByText('n/a')).toBeInTheDocument()
  })

  it('训练没评估过 → 指标 / 样图两个 tab 都不显示', async () => {
    mockTask(
      makeTask({ id: 119, task_type: 'train', status: 'done', project_id: 3, version_id: 7 }),
      [],
    )
    renderDetailPage()

    // 等 tab 收敛（listEvalSessions 回来之前不收，避免抖动）
    await waitFor(() => expect(screen.queryByText('指标')).not.toBeInTheDocument())
    expect(screen.queryByText('样图')).not.toBeInTheDocument()
    // 其余训练 tab 不受影响
    expect(screen.getByText('监控')).toBeInTheDocument()
  })

  it('训练评估过 → 两个 tab 都在', async () => {
    mockTask(
      makeTask({ id: 119, task_type: 'train', status: 'done', project_id: 3, version_id: 7 }),
      [{ id: 42, task_id: 500, parent_task_id: 119, status: 'done', trigger: 'after_training' }],
    )
    renderDetailPage()

    await waitFor(() => expect(screen.getByText('指标')).toBeInTheDocument())
    expect(screen.getByText('样图')).toBeInTheDocument()
  })
})
