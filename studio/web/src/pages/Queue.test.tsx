/** Queue page — taskKind 契约测试（0.17 P-D）。
 *
 *  taskKind 从后端权威 `task.task_type`（train/reg_ai/generate）派生队列行的类型，
 *  取代旧 inferKind 的 config_name 子串猜测。核心契约：
 *   1) 直接返回后端 task_type；
 *   2) 缺字段（老 mock / 极老行）兜底 'train'；
 *   3) 不再受 config_name 影响 —— 修掉旧 inferKind 把名字含 "reg"/"tag" 的
 *      训练任务误判成别的类型的 latent bug。 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../components/Dialog'
import { ToastProvider } from '../components/Toast'
import { api, type Task } from '../api/client'
import QueuePage, { taskKind } from './Queue'

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

describe('QueuePage 分区 + 分页', () => {
  it('渲染进行中/等待/历史三分区，历史超过一页时出分页器', async () => {
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

    await waitFor(() => expect(screen.getByText(/进行中/)).toBeInTheDocument())
    expect(screen.getByText(/等待入队/)).toBeInTheDocument()
    expect(screen.getByText(/历史/)).toBeInTheDocument()
    // 历史 total=25 > page_size=20 → 分页器 + 页码指示
    expect(screen.getByText(/第 1 \/ 2 页/)).toBeInTheDocument()
    expect(screen.getByTestId('history-prev')).toBeDisabled()
    expect(screen.getByTestId('history-next')).not.toBeDisabled()
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
    // scheduled 行有专属操作；pending 行没有
    expect(screen.getByTestId('startnow-btn-21')).toBeInTheDocument()
    expect(screen.getByTestId('cancel-scheduled-btn-21')).toBeInTheDocument()
    expect(screen.queryByTestId('startnow-btn-20')).not.toBeInTheDocument()
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

  it('右上角「数据作业」toggle → 切换只读区，漏斗变 kind 过滤（P-G）', async () => {
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

    fireEvent.click(screen.getByTestId('queue-jobs-toggle'))
    await waitFor(() => expect(screen.getByTestId('data-jobs-panel')).toBeInTheDocument())
    // 任务分区没了；漏斗还在（数据作业视图的 kind 过滤），点开出 kind select
    expect(screen.queryByText(/等待入队/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId('queue-filter-toggle'))
    expect(screen.getByTestId('jobs-kind-filter')).toBeInTheDocument()
    expect(screen.getByTestId('jobs-search')).toBeInTheDocument()
    // 任务视图专属的搜索框不在
    expect(screen.queryByTestId('queue-search')).not.toBeInTheDocument()

    // 再点 toggle 切回任务视图
    fireEvent.click(screen.getByTestId('queue-jobs-toggle'))
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
    // 默认收起：搜索框不在
    expect(screen.queryByTestId('queue-search')).not.toBeInTheDocument()
    // 点漏斗 → 过滤行展开，搜索框出现
    fireEvent.click(screen.getByTestId('queue-filter-toggle'))
    expect(screen.getByTestId('queue-search')).toBeInTheDocument()
  })
})
