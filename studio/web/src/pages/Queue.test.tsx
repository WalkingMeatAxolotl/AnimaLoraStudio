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
})
