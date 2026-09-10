import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode } from 'react'
import { createMemoryRouter, Outlet, RouterProvider } from 'react-router-dom'
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type ProjectDetail, type Task, type Version } from '../../../api/client'
import i18n from '../../../i18n'
import TrainPage from './Train'

// Node's Request rejects jsdom's AbortSignal. Preserve the router's signal on
// a real Request while avoiding the cross-realm constructor check.
const NativeRequest = globalThis.Request
beforeAll(() => {
  globalThis.Request = class extends NativeRequest {
    constructor(input: RequestInfo | URL, init?: RequestInit) {
      super(input, { ...init, signal: undefined })
      if (init?.signal) Object.defineProperty(this, 'signal', { value: init.signal })
    }
  }
})
afterAll(() => { globalThis.Request = NativeRequest })

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  prompt: vi.fn<() => Promise<string | null>>(async () => null),
  guard: null as null | (() => Promise<boolean>),
  setVersionSwitchGuard: vi.fn((guard: null | (() => Promise<boolean>)) => { mocks.guard = guard }),
  reload: vi.fn(async () => undefined),
  onEvent: undefined as undefined | ((event: unknown) => void),
}))

vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../components/Dialog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Dialog')>()
  return {
    ...actual,
    useDialog: () => ({
      confirm: vi.fn(async () => true),
      prompt: mocks.prompt,
    }),
  }
})
vi.mock('../../../lib/SettingsDrawer', () => ({
  useSettingsDrawer: () => ({ open: vi.fn() }),
}))
vi.mock('../../../lib/useEventStream', () => ({
  useEventStream: (callback: (event: unknown) => void) => {
    mocks.onEvent = callback
  },
}))

const config = {
  model_family: 'anima',
  transformer_path: 'G:/models/anima-base-v1.0.safetensors',
  lora_type: 'lora',
  lora_rank: 32,
  resolution: [1024],
  navit_packing: false,
  batch_size: 2,
  grad_accum: 1,
  epochs: 10,
  optimizer_type: 'adamw',
  learning_rate: 0.0001,
  mixed_precision: 'bf16',
  output_name: 'demo',
}

function renderPage(liveTask: Task | null = null, strict = false) {
  vi.mocked(api.listQueueLive).mockResolvedValue(liveTask ? [liveTask] : [])
  const project = { id: 7, title: 'Demo' } as ProjectDetail
  const activeVersion = {
    id: 11,
    label: 'v1',
    stats: {
      train_image_count: 4,
      train_folders: [{ name: '1_data', image_count: 4 }],
      masked_count: 0,
    },
  } as unknown as Version

  const router = createMemoryRouter([
    { element: <Outlet context={{ project, activeVersion, reload: mocks.reload, setVersionSwitchGuard: mocks.setVersionSwitchGuard }} />,
      children: [{ path: '/train', element: <TrainPage /> }] },
    { path: '/queue/:id', element: <div>Task detail route</div> },
    { path: '/away', element: <div>Away route</div> },
  ], { initialEntries: ['/train'], future: { v7_relativeSplatPath: true } })
  const ui = <RouterProvider router={router} future={{ v7_startTransition: true }} />
  return { ...render(strict ? <StrictMode>{ui}</StrictMode> : ui), router }

}

beforeEach(async () => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  localStorage.clear()
  await i18n.changeLanguage('zh')
  mocks.onEvent = undefined
  mocks.guard = null
  mocks.prompt.mockResolvedValue(null)

  vi.spyOn(api, 'schema').mockResolvedValue({
    schema: { properties: {} },
    groups: [],
  })
  vi.spyOn(api, 'listPresets').mockResolvedValue([])
  vi.spyOn(api, 'getSecrets').mockResolvedValue({ models: { auto_sync_paths: true } } as never)
  vi.spyOn(api, 'getVersionConfig').mockResolvedValue({ has_config: true, config } as never)
  vi.spyOn(api, 'getRegStatus').mockResolvedValue({ exists: false, meta: null, image_count: 0, files: [] })
  vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
  vi.spyOn(api, 'previewConfigYaml').mockResolvedValue({ yaml: 'model_family: anima\n' })
  vi.spyOn(api, 'getBucketDistribution').mockResolvedValue({
    resolutions: [1024],
    aspect_ratio_limit: 2,
    groups: [],
    navit: null,
  })
  vi.spyOn(api, 'listCropWorkspaceTrain').mockResolvedValue({ images: [] } as never)
  vi.spyOn(api, 'putVersionConfig').mockResolvedValue({ has_config: true, config } as never)
  vi.spyOn(api, 'enqueueVersionTraining').mockResolvedValue({ id: 91, status: 'pending' } as Task)
})

describe('Train workbench', () => {
  it('keeps submission in the header and display mode in the config toolbar', async () => {
    renderPage()

    const mode = await screen.findByRole('radiogroup', { name: '参数显示模式' })
    expect(within(mode).getByRole('radio', { name: '简单' })).toHaveAttribute('aria-checked', 'true')
    const toolbar = screen.getByRole('region', { name: '配置工具栏' })
    const toolbarMain = toolbar.querySelector('.train-config-toolbar-main')
    const modeColumn = mode.closest('[data-train-display-mode]')
    expect(toolbar).toHaveClass('train-config-toolbar')
    expect(toolbarMain).not.toBeNull()
    expect(toolbarMain).not.toContainElement(mode)
    expect(toolbar).toContainElement(mode)
    expect(modeColumn).toHaveClass('train-config-toolbar-mode')
    expect(within(toolbar).getByRole('button', { name: '另存为新预设' })).toBeInTheDocument()

    const actions = screen.getByRole('group', { name: '训练操作' })
    expect(actions.closest('.ui-page-header')).not.toBeNull()
    expect(within(actions).getByRole('button', { name: '开始训练' })).toBeEnabled()
    expect(within(actions).getByRole('button', { name: '定时训练' })).toBeEnabled()
    expect(screen.getAllByRole('button', { name: '开始训练' })).toHaveLength(1)
    expect(screen.getAllByRole('status')).toHaveLength(1)

    const preview = screen.getByRole('complementary', { name: '训练预览' })
    expect(within(toolbar).getByText('参数显示')).toBeVisible()
    expect(within(toolbar).getByRole('button', { name: '配置 Demo / v1 配置' })).toHaveClass('train-preset-trigger')
    const summary = within(preview).getByLabelText('训练摘要')
    expect(summary.tagName).toBe('DIV')
    expect(within(summary).getByText('Anima base 1.0')).toBeInTheDocument()
    expect(within(summary).getByText('lora')).toBeInTheDocument()
    expect(within(summary).getByText('demo')).toBeInTheDocument()
    expect(within(summary).getByText('10')).toBeInTheDocument()
    expect(within(summary).getByText('≈ 20')).toBeInTheDocument()
    expect(within(summary).getAllByRole('term')).toHaveLength(5)
    expect(within(summary).queryByText(/秩|rank/)).not.toBeInTheDocument()
    expect(within(preview).queryByRole('button', { name: '开始训练' })).not.toBeInTheDocument()
    for (const text of ['本次训练方案', '可提交', '训练精度', '优化器 · 学习率', '训练时长', 'adamw', 'bf16']) {
      expect(within(preview).queryByText(text)).not.toBeInTheDocument()
    }
  })

  it('restores the collapse handle, focus, saved state and content-specific widths', async () => {
    const user = userEvent.setup()
    const page = renderPage()
    const toggle = await screen.findByRole('button', { name: '收起预览' })
    const workspace = document.querySelector('.train-workbench')
    expect(workspace).toHaveAttribute('data-preview-tab', 'stats')
    await user.click(screen.getByRole('tab', { name: 'YAML 预览' }))
    expect(workspace).toHaveAttribute('data-preview-tab', 'config')
    await user.click(toggle)
    expect(workspace).toHaveAttribute('data-preview-open', 'false')
    expect(toggle).toHaveFocus()
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByRole('complementary', { name: '训练预览' })).not.toBeInTheDocument()
    expect(localStorage.getItem('train.previewOpen')).toBe('false')
    page.unmount()

    renderPage()
    await user.click(await screen.findByRole('button', { name: '展开预览' }))
    expect(screen.getByRole('complementary', { name: '训练预览' })).toBeVisible()
    expect(screen.getByRole('tab', { name: 'YAML 预览' })).toHaveAttribute('aria-selected', 'true')
    expect(localStorage.getItem('train.previewOpen')).toBe('true')
  })

  it('names the configured custom model rather than assuming a family default', async () => {
    vi.mocked(api.getVersionConfig).mockResolvedValue({
      has_config: true, config: { ...config, transformer_path: 'C:\\custom\\my-model-v2.safetensors' },
    } as never)
    renderPage()
    expect(await screen.findByText('my-model-v2')).toHaveAttribute('title', 'C:\\custom\\my-model-v2.safetensors')
    expect(screen.queryByText('Anima base 1.0')).not.toBeInTheDocument()
  })

  it('switches peer preview tabs with keyboard navigation and remembers the selection', async () => {
    const user = userEvent.setup()
    const page = renderPage()
    const stats = await screen.findByRole('tab', { name: '数据分布' })
    await screen.findByRole('radiogroup', { name: '参数显示模式' })
    expect(stats).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel', { name: '数据分布' })).toBeVisible()
    expect(screen.queryByRole('tabpanel', { name: 'YAML 预览' })).not.toBeInTheDocument()

    stats.focus()
    await user.keyboard('{ArrowRight}')
    const yaml = screen.getByRole('tab', { name: 'YAML 预览' })
    expect(yaml).toHaveFocus()
    expect(yaml).toHaveAttribute('aria-selected', 'true')
    const panel = screen.getByRole('tabpanel', { name: 'YAML 预览' })
    expect(panel).toHaveAttribute('id', yaml.getAttribute('aria-controls'))
    expect(await within(panel).findByText('model_family: anima')).toBeInTheDocument()
    expect(screen.queryByRole('tabpanel', { name: '数据分布' })).not.toBeInTheDocument()
    expect(localStorage.getItem('train.previewTab')).toBe('"config"')
    expect(screen.getByRole('button', { name: '开始训练' })).toBeEnabled()

    page.unmount()
    renderPage()
    expect(await screen.findByRole('tab', { name: 'YAML 预览' })).toHaveAttribute('aria-selected', 'true')
    await user.click(screen.getByRole('tab', { name: '数据分布' }))
    expect(screen.getByRole('tabpanel', { name: '数据分布' })).toBeVisible()
  })

  it('falls back to dataset stats for an invalid persisted tab', async () => {
    localStorage.setItem('train.previewTab', '"removed-tab"')
    renderPage()
    expect(await screen.findByRole('tab', { name: '数据分布' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel', { name: '数据分布' })).toBeVisible()
  })

  it('restores the dev workload derivation including natural and capped totals', async () => {
    vi.mocked(api.getVersionConfig).mockResolvedValue({
      has_config: true, config: { ...config, max_steps: 15 },
    } as never)
    renderPage()
    const panel = await screen.findByRole('region', { name: '数据与训练规模' })
    expect(within(panel).getByText('÷ (batch × GA)（2 × 1）')).toBeInTheDocument()
    expect(within(panel).getByText('≈ 2 steps/epoch')).toBeInTheDocument()
    expect(within(panel).getByText('× epochs (10)')).toBeInTheDocument()
    expect(within(panel).getByText('≈ 20 steps')).toBeInTheDocument()
    expect(within(panel).getByText('≈ 15')).toBeInTheDocument()
    expect(within(panel).getByText(i18n.t('train.maxStepsLabel', { n: 15 }))).toBeInTheDocument()
    expect(within(screen.getByRole('group', { name: '训练摘要' })).getByText('≈ 15')).toBeInTheDocument()
  })

  it('uses pack estimates instead of ordinary batch size for NaViT', async () => {
    vi.mocked(api.getVersionConfig).mockResolvedValue({
      has_config: true, config: { ...config, navit_packing: true, batch_size: 99, grad_accum: 2 },
    } as never)
    vi.mocked(api.getBucketDistribution).mockResolvedValue({
      resolutions: [1024], aspect_ratio_limit: 2, groups: [],
      navit: { native: false, packs_per_epoch: 8, samples: 12 },
    } as never)
    renderPage()
    const panel = await screen.findByRole('region', { name: '数据与训练规模' })
    expect(await within(panel).findByText('≈ 4 steps/epoch')).toBeInTheDocument()
    expect(within(panel).getByText('× epochs (10)')).toBeInTheDocument()
    expect(within(panel).getByText('≈ 40 steps')).toBeInTheDocument()
    expect(within(panel).getByText('≈ 40')).toBeInTheDocument()
    expect(within(panel).queryByText(/batch × GA/)).not.toBeInTheDocument()
  })

  it('shows unsaved work immediately and persists the next-run draft', async () => {
    vi.mocked(api.schema).mockResolvedValue({
      groups: [{ key: 'training', label: '训练' }],
      schema: {
        properties: {
          epochs: {
            type: 'integer',
            default: 10,
            group: 'training',
            description: 'Epochs',
          },
        },
      },
    })
    const pending = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(pending.promise)
    const user = userEvent.setup()
    renderPage()
    const input = await screen.findByRole('textbox')

    await user.clear(input)
    await user.type(input, '12')
    await user.tab()

    await waitFor(() => {
      expect(api.putVersionConfig).toHaveBeenCalledWith(
        7,
        11,
        expect.objectContaining({ epochs: 12 }),
      )
    }, { timeout: 2000 })
    expect(screen.getByRole('status')).toHaveTextContent('保存中…')
    await act(async () => pending.resolve({
      has_config: true,
      config: { ...config, epochs: 12 },
    } as never))
    expect(await screen.findByText(/已保存/)).toBeInTheDocument()
  })

  it('opens scheduling in the shared accessible dialog', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('button', { name: '定时训练' }))

    expect(screen.getByRole('dialog', { name: '定时训练' })).toBeInTheDocument()
    expect(screen.getByLabelText('指定时间')).toHaveAttribute('type', 'datetime-local')

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: '定时训练' })).not.toBeInTheDocument()
  })

  it('flushes and navigates directly to QueueDetail on immediate submit', async () => {
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole('button', { name: '开始训练' }))

    await waitFor(() => {
      expect(api.enqueueVersionTraining).toHaveBeenCalledWith(7, 11, undefined)
    })
    expect(await screen.findByText('Task detail route')).toBeInTheDocument()
  })

  it('submits an absolute local schedule and opens its QueueDetail', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('button', { name: '定时训练' }))
    await user.type(screen.getByLabelText('指定时间'), '2099-01-02T03:04')
    await user.click(screen.getByRole('button', { name: '定时入队' }))

    await waitFor(() => {
      expect(api.enqueueVersionTraining).toHaveBeenCalledWith(
        7,
        11,
        { scheduledAt: new Date('2099-01-02T03:04').getTime() / 1000 },
      )
    })
    expect(await screen.findByText('Task detail route')).toBeInTheDocument()
  })

  it('offers both preset paths in the first-run empty state', async () => {
    vi.mocked(api.getVersionConfig).mockResolvedValue({ has_config: false, config: null } as never)
    renderPage()

    expect(await screen.findByText('未配置 — 选预设作为起点')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '选择已有预设' })).toBeEnabled()
    expect(screen.getAllByRole('button', { name: '+ 新建预设' }).length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: '开始训练' })).toBeDisabled()
  })

  it('shows an active task snapshot and blocks duplicate submission', async () => {
    const task = {
      id: 42,
      name: 'demo_v1',
      config_name: 'demo',
      task_type: 'train',
      status: 'running',
      priority: 0,
      created_at: 1,
      started_at: 2,
      finished_at: null,
      pid: 123,
      exit_code: null,
      output_dir: null,
      error_msg: null,
      project_id: 7,
      version_id: 11,
    } as Task
    const user = userEvent.setup()
    renderPage(task)

    expect(await screen.findByText('当前训练任务')).toBeInTheDocument()
    expect(screen.getByText('运行中')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始训练' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '定时训练' })).toBeDisabled()
    await user.click(screen.getByRole('tab', { name: 'YAML 预览' }))
    expect(screen.getByText('当前训练任务')).toBeVisible()
    expect(screen.getByText('下一轮草稿')).toBeVisible()
    expect(screen.getByRole('button', { name: '开始训练' })).toBeDisabled()

    await user.click(screen.getByRole('button', { name: '查看任务详情' }))
    expect(await screen.findByText('Task detail route')).toBeInTheDocument()
  })

  it('blocks training while another GPU task owns the version resources', async () => {
    renderPage({
      id: 51,
      name: 'demo_generate',
      config_name: 'demo',
      task_type: 'generate',
      status: 'running',
      priority: 0,
      created_at: 1,
      started_at: 2,
      project_id: 7,
      version_id: 11,
    } as Task)

    expect(await screen.findByText('正在占用训练资源')).toBeInTheDocument()
    expect(screen.getByText(/资源释放前不可提交/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始训练' })).toBeDisabled()
  })

  it('refreshes the current task when SSE announces a different live task', async () => {
    const current = {
      id: 42,
      name: 'demo_current',
      config_name: 'demo',
      task_type: 'train',
      status: 'running',
      created_at: 1,
      project_id: 7,
      version_id: 11,
    } as Task
    renderPage(current)
    expect(await screen.findByText('#42 · demo_current')).toBeInTheDocument()

    const task = {
      id: 77,
      name: 'demo_next',
      config_name: 'demo',
      task_type: 'train',
      status: 'scheduled',
      created_at: 1,
      project_id: 7,
      version_id: 11,
    } as Task
    vi.mocked(api.listQueueLive).mockResolvedValue([task])

    act(() => mocks.onEvent?.({ type: 'task_state_changed', task_id: 77 }))

    expect(await screen.findByText('#77 · demo_next')).toBeInTheDocument()
    expect(screen.getByText('等待入队')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始训练' })).toBeDisabled()
  })

  it('keeps submission enabled during a background live-task refresh', async () => {
    renderPage()
    const start = await screen.findByRole('button', { name: '开始训练' })
    expect(start).toBeEnabled()

    const refresh = deferred<Task[]>()
    vi.mocked(api.listQueueLive).mockReturnValue(refresh.promise)
    act(() => mocks.onEvent?.({ type: 'task_state_changed', task_id: 900 }))

    expect(start).toBeEnabled()
    expect(screen.queryByText('正在检查当前任务…')).not.toBeInTheDocument()

    await act(async () => { refresh.resolve([]); await refresh.promise })
    expect(start).toBeEnabled()
  })

  it('formats active task times with the current interface locale', async () => {
    await i18n.changeLanguage('en')
    const format = vi.spyOn(Date.prototype, 'toLocaleString').mockReturnValue('localized time')
    renderPage({
      id: 61,
      name: 'demo_v1',
      config_name: 'demo',
      task_type: 'train',
      status: 'scheduled',
      priority: 0,
      created_at: 1,
      scheduled_at: 2,
      project_id: 7,
      version_id: 11,
    } as Task)

    expect(await screen.findByText('Current training task')).toBeInTheDocument()
    expect(screen.getByText(/localized time/)).toBeInTheDocument()
    expect(format).toHaveBeenCalledWith('en-US', { hour12: false })
  })

  it('keeps config load failures visible and recoverable', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getVersionConfig)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce({ has_config: true, config } as never)
    renderPage()

    expect(await screen.findByText(/加载训练配置失败.*offline/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '重试' }))
    expect(await screen.findByRole('radiogroup', { name: '参数显示模式' })).toBeInTheDocument()
    expect(screen.queryByText(/加载训练配置失败/)).not.toBeInTheDocument()
  })

  it('uses the shared modal for scheduling and keeps delayed enqueue atomic', async () => {
    const user = userEvent.setup()
    renderPage()
    const schedule = await screen.findByRole('button', { name: '定时训练' })
    await user.click(schedule)

    const dialog = screen.getByRole('dialog', { name: '定时训练' })
    expect(dialog).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: '+1h' }))

    await waitFor(() => expect(api.enqueueVersionTraining).toHaveBeenCalledWith(
      7,
      11,
      expect.objectContaining({ scheduledAt: expect.any(Number) }),
    ))
    expect(await screen.findByText('Task detail route')).toBeInTheDocument()
  })
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

function editableConfig() {
  vi.mocked(api.schema).mockResolvedValue({
    groups: [{ key: 'training', label: '训练' }],
    schema: { properties: { output_name: { type: 'string', default: 'demo', group: 'training', description: 'Output name' } } },
  })
  vi.mocked(api.putVersionConfig).mockImplementation(async (_pid, _vid, body) => ({ has_config: true, config: body }) as never)
}

async function editDraft(value: string) {
  const user = userEvent.setup()
  const input = await screen.findByRole('textbox')
  await user.clear(input)
  await user.type(input, value)
  await user.tab()
  return input
}

describe('Train save lifecycle', () => {
  it('keeps the dev idle indicator on load and a clean version guard', async () => {
    renderPage()
    await screen.findByRole('radiogroup', { name: '参数显示模式' })
    expect(screen.getByRole('status')).toHaveAttribute('data-state', 'idle')
    expect(screen.getByRole('status')).toBeEmptyDOMElement()
    await act(async () => expect(await mocks.guard!()).toBe(true))
    expect(api.putVersionConfig).not.toHaveBeenCalled()
    expect(screen.getByRole('status')).toBeEmptyDOMElement()
  })

  it('does not refresh the saved timestamp when a later boundary has nothing to write', async () => {
    editableConfig()
    renderPage()
    await editDraft('real-save')
    const now = vi.spyOn(Date, 'now').mockReturnValue(1_000_000)
    await act(async () => expect(await mocks.guard!()).toBe(true))
    const savedText = screen.getByRole('status').textContent
    expect(screen.getByRole('status')).toHaveAttribute('data-state', 'saved')
    now.mockReturnValue(2_000_000)
    await act(async () => expect(await mocks.guard!()).toBe(true))
    expect(api.putVersionConfig).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('status').textContent).toBe(savedText)
  })

  it('waits for a dirty draft before SPA departure, including StrictMode', async () => {
    editableConfig()
    const pending = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(pending.promise)
    const { router } = renderPage(null, true)
    const input = await editDraft('latest')
    act(() => { void router.navigate('/away') })
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(1))
    expect(router.state.location.pathname).toBe('/train')
    expect(input).toBeDisabled()
    await act(async () => pending.resolve({ has_config: true, config: { ...config, output_name: 'latest' } } as never))
    expect(await screen.findByText('Away route')).toBeInTheDocument()
    expect(api.putVersionConfig).toHaveBeenCalledTimes(1)
  })

  it('keeps failed navigation on the draft without automatic retries or duplicate announcements, then retries', async () => {
    editableConfig()
    vi.mocked(api.putVersionConfig).mockRejectedValueOnce(new Error('offline'))
    const { router } = renderPage()
    await editDraft('retain-me')
    act(() => { void router.navigate('/away') })
    const error = await screen.findByText(/保存失败.*offline/)
    expect(router.state.location.pathname).toBe('/train')
    expect(screen.getByRole('textbox')).toHaveValue('retain-me')
    expect(error.closest('[role="alert"]')).not.toBeNull()
    expect(mocks.toast).not.toHaveBeenCalled()
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 800)) })
    expect(api.putVersionConfig).toHaveBeenCalledTimes(1)
    await userEvent.setup().click(screen.getByRole('button', { name: '重试' }))
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(2))
    act(() => { void router.navigate('/away') })
    expect(await screen.findByText('Away route')).toBeInTheDocument()
  })

  it('registers the existing version guard, rejects failed saves and allows a successful retry', async () => {
    editableConfig()
    const pending = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(pending.promise)
    const page = renderPage()
    await editDraft('version-draft')
    expect(mocks.guard).toBeTypeOf('function')
    let result!: Promise<boolean>
    act(() => { result = mocks.guard!() })
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(1))
    await act(async () => { pending.reject(new Error('offline')); expect(await result).toBe(false) })
    expect(screen.getByRole('textbox')).toHaveValue('version-draft')
    await act(async () => expect(await mocks.guard!()).toBe(true))
    expect(api.putVersionConfig).toHaveBeenCalledTimes(2)
    page.unmount()
    expect(mocks.guard).toBeNull()
  })

  it('saves the latest edit after an in-flight PUT before saving as a preset', async () => {
    editableConfig()
    mocks.prompt.mockResolvedValue('new-preset')
    vi.spyOn(api, 'saveVersionConfigAsPreset').mockResolvedValue({} as never)
    const first = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    const second = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    renderPage()
    await editDraft('first')
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(1), { timeout: 2000 })
    const input = await editDraft('latest')
    await userEvent.setup().click(screen.getByRole('button', { name: '另存为新预设' }))
    expect(api.saveVersionConfigAsPreset).not.toHaveBeenCalled()
    expect(input).toBeDisabled()
    await act(async () => first.resolve({ has_config: true, config: { ...config, output_name: 'first' } } as never))
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(2))
    expect(input).toHaveValue('latest')
    expect(screen.getByRole('status')).toHaveTextContent('保存中')
    expect(api.saveVersionConfigAsPreset).not.toHaveBeenCalled()
    expect(api.putVersionConfig).toHaveBeenLastCalledWith(7, 11, expect.objectContaining({ output_name: 'latest' }))
    await act(async () => second.resolve({ has_config: true, config: { ...config, output_name: 'latest' } } as never))
    await waitFor(() => expect(api.saveVersionConfigAsPreset).toHaveBeenCalledWith(7, 11, 'new-preset', false))
  })

  it.each(['fork', 'create'])('serializes %s after an in-flight PUT without an old autosave overwriting the preset', async (action) => {
    editableConfig()
    vi.mocked(api.listPresets).mockResolvedValue([{ name: 'template' }] as never)
    vi.spyOn(api, 'savePreset').mockResolvedValue({} as never)
    vi.spyOn(api, 'forkPresetForVersion').mockImplementation(async () => {
      vi.mocked(api.getVersionConfig).mockResolvedValue({ has_config: true, config: { ...config, output_name: 'replacement' } } as never)
      return {} as never
    })
    const pending = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(pending.promise)
    renderPage()
    await editDraft('old-draft')
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(1), { timeout: 2000 })
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: '配置 Demo / v1 配置' }))
    await user.click(screen.getByRole('button', { name: action === 'fork' ? /template/ : '+ 新建预设' }))
    expect(api.forkPresetForVersion).not.toHaveBeenCalled()
    expect(api.savePreset).not.toHaveBeenCalled()
    await act(async () => pending.resolve({ has_config: true, config: { ...config, output_name: 'old-draft' } } as never))
    await waitFor(() => expect(api.forkPresetForVersion).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue('replacement'))
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 800)) })
    expect(api.putVersionConfig).toHaveBeenCalledTimes(1)
  })

  it('flushes a dirty draft before enqueue and opens its exact task', async () => {
    editableConfig()
    const pending = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(pending.promise)
    const { router } = renderPage()
    const input = await editDraft('queued-draft')
    await userEvent.setup().click(screen.getByRole('button', { name: '开始训练' }))
    expect(api.enqueueVersionTraining).not.toHaveBeenCalled()
    expect(input).toBeDisabled()
    await act(async () => pending.resolve({ has_config: true, config: { ...config, output_name: 'queued-draft' } } as never))
    expect(await screen.findByText('Task detail route')).toBeInTheDocument()
    expect(router.state.location.pathname).toBe('/queue/91')
    expect(api.putVersionConfig).toHaveBeenCalledTimes(1)
  })

  it('does not launch a hidden PUT on forced unmount', async () => {
    editableConfig()
    const page = renderPage()
    await editDraft('unsaved')
    page.unmount()
    expect(api.putVersionConfig).not.toHaveBeenCalled()
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 800)) })
    expect(api.putVersionConfig).not.toHaveBeenCalled()
  })
  it('drains newer edits during ordinary autosave without briefly reporting saved', async () => {
    editableConfig()
    const first = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    const second = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    renderPage()
    await editDraft('first')
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(1), { timeout: 2000 })
    const input = await editDraft('newer')
    expect(input).toBeEnabled()
    await act(async () => first.resolve({ has_config: true, config: { ...config, output_name: 'FIRST-normalized' } } as never))
    expect(input).toHaveValue('newer')
    expect(screen.getByRole('status')).toHaveTextContent('保存中')
    expect(api.putVersionConfig).toHaveBeenLastCalledWith(7, 11, expect.objectContaining({ output_name: 'newer' }))
    await act(async () => second.resolve({ has_config: true, config: { ...config, output_name: 'NEWER-normalized' } } as never))
    expect(input).toHaveValue('NEWER-normalized')
    expect(screen.getByRole('status')).toHaveTextContent('已保存')
  })

  it('does not retry a failed in-flight autosave implicitly when a departure is waiting', async () => {
    editableConfig()
    const pending = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(pending.promise)
    const { router } = renderPage(null, true)
    await editDraft('first')
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(1), { timeout: 2000 })
    await editDraft('newer')
    act(() => { void router.navigate('/away') })
    await act(async () => pending.reject(new Error('offline')))
    await screen.findByText(/保存失败.*offline/)
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 800)) })
    expect(api.putVersionConfig).toHaveBeenCalledTimes(1)
    expect(router.state.location.pathname).toBe('/train')
    expect(screen.getByRole('textbox')).toHaveValue('newer')
    expect(mocks.toast).not.toHaveBeenCalled()
    act(() => { void router.navigate('/away') })
    expect(await screen.findByText('Away route')).toBeInTheDocument()
    expect(api.putVersionConfig).toHaveBeenLastCalledWith(7, 11, expect.objectContaining({ output_name: 'newer' }))
  })

  it.each(['enqueue', 'save-as', 'fork', 'create'])('does not execute %s after a save failure', async (action) => {
    editableConfig()
    mocks.prompt.mockResolvedValue('new-preset')
    vi.mocked(api.listPresets).mockResolvedValue([{ name: 'template' }] as never)
    vi.mocked(api.putVersionConfig).mockRejectedValueOnce(new Error('offline'))
    vi.spyOn(api, 'saveVersionConfigAsPreset').mockResolvedValue({} as never)
    vi.spyOn(api, 'savePreset').mockResolvedValue({} as never)
    vi.spyOn(api, 'forkPresetForVersion').mockResolvedValue({} as never)
    renderPage()
    const input = await editDraft('retain')
    const user = userEvent.setup()
    if (action === 'fork' || action === 'create') {
      await user.click(screen.getByRole('button', { name: '配置 Demo / v1 配置' }))
      await user.click(screen.getByRole('button', { name: action === 'fork' ? /template/ : '+ 新建预设' }))
    } else {
      await user.click(screen.getByRole('button', { name: action === 'enqueue' ? '开始训练' : '另存为新预设' }))
    }
    await screen.findByText(/保存失败.*offline/)
    expect(input).toHaveValue('retain')
    expect(input).toBeEnabled()
    expect(api.enqueueVersionTraining).not.toHaveBeenCalled()
    expect(api.saveVersionConfigAsPreset).not.toHaveBeenCalled()
    expect(api.savePreset).not.toHaveBeenCalled()
    expect(api.forkPresetForVersion).not.toHaveBeenCalled()
    expect(mocks.toast).not.toHaveBeenCalled()
  })

  it('keeps the next-run draft editable while a submitted task runs', async () => {
    editableConfig()
    renderPage({ id: 42, task_type: 'train', status: 'running', project_id: 7, version_id: 11 } as Task)
    const input = await editDraft('next-run')
    expect(input).toBeEnabled()
    expect(screen.getByRole('button', { name: '开始训练' })).toBeDisabled()
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledWith(7, 11, expect.objectContaining({ output_name: 'next-run' })), { timeout: 2000 })
  })

  it('retains beforeunload protection until the latest draft is saved', async () => {
    editableConfig()
    renderPage()
    await editDraft('draft')
    const dirtyEvent = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(dirtyEvent)
    expect(dirtyEvent.defaultPrevented).toBe(true)
    await act(async () => expect(await mocks.guard!()).toBe(true))
    const savedEvent = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(savedEvent)
    expect(savedEvent.defaultPrevented).toBe(false)
  })

  it('waits for the preset action boundary before navigation and keeps editing locked through the POST', async () => {
    editableConfig()
    mocks.prompt.mockResolvedValue('new-preset')
    const pending = deferred<Awaited<ReturnType<typeof api.saveVersionConfigAsPreset>>>()
    vi.spyOn(api, 'saveVersionConfigAsPreset').mockReturnValueOnce(pending.promise)
    const { router } = renderPage()
    const input = await editDraft('exported')
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: '另存为新预设' }))
    await waitFor(() => expect(api.saveVersionConfigAsPreset).toHaveBeenCalledTimes(1))
    expect(input).toBeDisabled()
    await user.type(input, 'ignored')
    expect(input).toHaveValue('exported')
    act(() => { void router.navigate('/away') })
    expect(router.state.location.pathname).toBe('/train')
    await act(async () => pending.resolve({} as never))
    expect(await screen.findByText('Away route')).toBeInTheDocument()
    expect(api.putVersionConfig).toHaveBeenCalledTimes(1)
  })

  it('shares a pending save between the version guard and SPA guard', async () => {
    editableConfig()
    const pending = deferred<Awaited<ReturnType<typeof api.putVersionConfig>>>()
    vi.mocked(api.putVersionConfig).mockReturnValueOnce(pending.promise)
    const { router } = renderPage(null, true)
    await editDraft('shared')
    let versionSwitch!: Promise<boolean>
    act(() => {
      versionSwitch = mocks.guard!()
      void router.navigate('/away')
    })
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(1))
    await act(async () => {
      pending.resolve({ has_config: true, config: { ...config, output_name: 'shared' } } as never)
      expect(await versionSwitch).toBe(true)
    })
    expect(await screen.findByText('Away route')).toBeInTheDocument()
    expect(api.putVersionConfig).toHaveBeenCalledTimes(1)
  })

  it('retries a failed forced rewrite even when the draft matches its baseline', async () => {
    editableConfig()
    vi.mocked(api.getVersionConfig).mockResolvedValue({ has_config: true, config, dropped_fields: ['legacy'] } as never)
    vi.mocked(api.putVersionConfig).mockRejectedValueOnce(new Error('offline'))
    renderPage()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: i18n.t('presets.cleanLegacyBtn') }))
    await screen.findByText(/保存失败.*offline/)
    await user.click(screen.getByRole('button', { name: '重试' }))
    await waitFor(() => expect(api.putVersionConfig).toHaveBeenCalledTimes(2))
    expect(mocks.toast).not.toHaveBeenCalled()
  })

})
