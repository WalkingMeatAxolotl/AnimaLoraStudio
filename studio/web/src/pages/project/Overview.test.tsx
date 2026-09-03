/** ProjectOverview 组件级 regression test。
 *
 *  覆盖训练态 StatusBanner 的暂停按钮实时刷新：latestTask 是 Overview 内独立
 *  local state（不是 project prop），Layout 的 version_state_changed reload 碰
 *  不到它。暂停按钮门控的 is_pausable 由 train_loop_started + 首个 epoch 的
 *  auto_epoch_backup_written 翻 true —— Overview 必须自己订阅这些 SSE 事件重拉
 *  latestTask，否则暂停按钮训练期一直不出现，得切版本 / 刷新页面才有。
 *
 *  jsdom 默认没有 EventSource（useEventStream 内部 typeof 守卫会短路），这里塞
 *  个 fake 让 hook 真订阅，再手动驱动事件验证组件会重拉 listQueue 并显示按钮。 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ToastProvider } from '../../components/Toast'
import { api, type ProjectDetail, type Task, type Version } from '../../api/client'
import ProjectOverview from './Overview'

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

function makeVersion(overrides: Partial<Version> = {}): Version {
  return {
    id: 7, project_id: 3, label: 'v1', config_name: 'train',
    status: 'training', phase: 'ready', last_failure_reason: null,
    created_at: 1000, output_lora_path: null, note: null, trigger_word: '',
    ...overrides,
  }
}

function makeProject(overrides: Partial<ProjectDetail> = {}): ProjectDetail {
  return {
    id: 3, slug: 'proj', title: 'Proj', active_version_id: 7,
    active_version_label: 'v1', active_version_status: 'training',
    active_version_phase: null, created_at: 1000, updated_at: 1000,
    archived_at: null, note: null,
    download_image_count: 0, preprocess_image_count: 0,
    versions: [makeVersion()],
    ...overrides,
  }
}

function makeTrainTask(is_pausable: boolean): Task {
  return {
    id: 42, name: 'train', config_name: 'train', status: 'running', priority: 0,
    created_at: 1000, started_at: 1100, finished_at: null, pid: 1234,
    exit_code: null, output_dir: null, error_msg: null,
    project_id: 3, version_id: 7, is_pausable,
  }
}

function renderOverview(project: ProjectDetail) {
  const ctxValue = {
    project,
    activeVersion: project.versions[0] ?? null,
    reload: async () => {},
    onCreateVersion: () => {},
    creatingVersionBusy: false,
  }
  return render(
    <MemoryRouter initialEntries={['/projects/3']}>
      <ToastProvider>
        <Routes>
          <Route element={<Outlet context={ctxValue} />}>
            <Route path="/projects/:id" element={<ProjectOverview />} />
          </Route>
        </Routes>
      </ToastProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
  // detail tab 的 getCuration / listCaptionsFull 等都有 .catch —— 统一 404 让它们
  // 安静失败，不污染测试；listQueue 单独 spy 控制返回。
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
    ok: false, status: 404, json: async () => null, text: async () => '',
    headers: new Headers(),
  } as Response)))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('ProjectOverview 训练态暂停按钮 SSE 刷新', () => {
  it('auto_epoch_backup_written 事件触发重拉 latestTask → 暂停按钮出现', async () => {
    // listQueue：首拉 is_pausable=false（首个 epoch backup 未落盘），之后 true。
    let pausable = false
    const listSpy = vi.spyOn(api, 'listQueue')
      .mockImplementation(async () => [makeTrainTask(pausable)])

    renderOverview(makeProject())

    // 训练 banner 已渲染（取消训练按钮随 taskId 出现），但 is_pausable=false →
    // 暂停按钮不在。
    await waitFor(() => expect(screen.getByText('取消训练')).toBeInTheDocument())
    expect(screen.getByRole('progressbar', { name: 'v1 训练进度' })).toHaveAttribute('data-state', 'indeterminate')
    expect(screen.queryByText('暂停')).not.toBeInTheDocument()
    const callsBefore = listSpy.mock.calls.length

    // 后端首个 epoch backup 落盘 → is_pausable 升级；推一条 SSE 事件。
    pausable = true
    await waitFor(() => expect(FakeEventSource.instances.length).toBeGreaterThan(0))
    FakeEventSource.instances[0].emit({ type: 'auto_epoch_backup_written', task_id: 42 })

    // 重拉后暂停按钮出现（不依赖切版本 / 刷新页面）。
    await waitFor(() => expect(screen.getByText('暂停')).toBeInTheDocument())
    expect(listSpy.mock.calls.length).toBeGreaterThan(callsBefore)
  })
})

describe('ProjectOverview 版本选择', () => {
  it('VersionRail 使用 radiogroup、roving tabindex 与方向键导航', async () => {
    vi.spyOn(api, 'listQueue').mockResolvedValue([])
    const project = makeProject({
      versions: [
        makeVersion(),
        makeVersion({ id: 8, label: 'v2', status: 'completed', phase: 'ready' }),
      ],
    })

    renderOverview(project)

    expect(screen.getByRole('radiogroup', { name: '项目版本' })).toBeInTheDocument()
    const v1 = screen.getByRole('radio', { name: 'v1 训练中' })
    const v2 = screen.getByRole('radio', { name: 'v2 已完成' })
    expect(v1).toHaveAttribute('aria-checked', 'true')
    expect(v1).toHaveAttribute('tabindex', '0')
    expect(v2).toHaveAttribute('tabindex', '-1')

    fireEvent.keyDown(v1, { key: 'ArrowRight' })
    await waitFor(() => expect(v2).toHaveAttribute('aria-checked', 'true'))
    expect(v2).toHaveAttribute('tabindex', '0')
    await waitFor(() => expect(v2).toHaveFocus())

    fireEvent.keyDown(v2, { key: 'Home' })
    await waitFor(() => expect(v1).toHaveAttribute('aria-checked', 'true'))
    expect(screen.getByRole('button', { name: '+ 新版本' })).toHaveClass('btn', 'btn-secondary', 'btn-sm')
  })
})

describe('ProjectOverview 内容 surfaces', () => {
  it('详情统计使用共享 Card，任务与输出空态使用 EmptyState', async () => {
    vi.spyOn(api, 'listQueue').mockResolvedValue([])

    renderOverview(makeProject())

    expect(screen.getByRole('heading', { level: 3, name: '训练集' }).closest('.card')).toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 3, name: '标签分布' }).closest('.card')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: '任务' }))
    expect((await screen.findByText('本项目还没有训练任务')).closest('.empty-state-sm')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'LoRA 文件' }))
    expect((await screen.findByText('此版本尚未产出 LoRA 模型')).closest('.empty-state-sm')).toBeInTheDocument()
  })

  it('任务表使用语义链接与共享状态 Badge', async () => {
    vi.spyOn(api, 'listQueue').mockResolvedValue([
      { ...makeTrainTask(false), status: 'done', finished_at: 1200 },
    ])

    renderOverview(makeProject())
    fireEvent.click(screen.getByRole('tab', { name: '任务' }))

    const taskLink = await screen.findByRole('link', { name: '#42 train' })
    expect(taskLink).toHaveAttribute('href', '/queue/42')
    const taskRow = taskLink.closest('tr')
    const taskBadge = taskRow?.querySelector('.badge')
    expect(taskBadge).toHaveClass('badge', 'badge-ok')
    expect(taskBadge).not.toHaveClass('badge-sm')
    expect(taskBadge).toHaveTextContent('完成')
    expect(taskRow).not.toHaveClass('cursor-pointer')
  })

  it('任务读取失败显示 Alert，不伪装为空状态', async () => {
    vi.spyOn(api, 'listQueue').mockRejectedValue(new Error('offline'))

    renderOverview(makeProject())
    fireEvent.click(screen.getByRole('tab', { name: '任务' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('任务列表读取失败：offline')
    expect(screen.queryByText('本项目还没有训练任务')).not.toBeInTheDocument()
  })
})

describe('ProjectOverview 评估 tab', () => {
  beforeEach(() => {
    vi.spyOn(api, 'listQueue').mockResolvedValue([])
    vi.spyOn(api, 'listEvalSessions').mockResolvedValue({ sessions: [] } as never)
  })

  it('共享 Tabs 暴露 tablist/panel 关联并支持方向键切换', async () => {
    renderOverview(makeProject())

    const tabs = screen.getByRole('tablist', { name: '项目概览内容' })
    const detailsTab = screen.getByRole('tab', { name: '详情' })
    const tasksTab = screen.getByRole('tab', { name: '任务' })
    const panel = screen.getByRole('tabpanel')

    expect(tabs).toContainElement(detailsTab)
    expect(detailsTab).toHaveAttribute('aria-selected', 'true')
    expect(detailsTab).toHaveAttribute('aria-controls', panel.id)

    fireEvent.keyDown(detailsTab, { key: 'ArrowRight' })

    await waitFor(() => expect(tasksTab).toHaveAttribute('aria-selected', 'true'))
    expect(tasksTab).toHaveFocus()
    expect(panel).toHaveAttribute('aria-labelledby', tasksTab.id)
  })

  it('点「评估」tab → 列该 version 的评估作业（不带 task_id）', async () => {
    renderOverview(makeProject())
    fireEvent.click(await screen.findByRole('tab', { name: '评估' }))

    await waitFor(() => expect(api.listEvalSessions).toHaveBeenCalledWith(3, 7))
    expect(await screen.findByText('创建新评估')).toBeInTheDocument()
  })

  it('停在别的 tab 时不拉评估（进项目页不该白跑一次请求）', async () => {
    renderOverview(makeProject())
    await waitFor(() => expect(api.listQueue).toHaveBeenCalled())
    expect(api.listEvalSessions).not.toHaveBeenCalled()
  })
})
