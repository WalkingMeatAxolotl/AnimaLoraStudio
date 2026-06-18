/** QueueDetail page 组件级 regression test。
 *
 *  目前只覆盖 SnapshotConfigTab 的 refetch trap：父组件每 2s 浅 clone task
 *  做 elapsed time tick，旧实现 [task] 作 deps 会让 snapshot config 也跟着
 *  2s 重拉 —— 浏览器卡顿、loading flash。 */
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ToastProvider } from '../components/Toast'
import type { Task } from '../api/client'
import QueueDetailPage, { SnapshotConfigTab } from './QueueDetail'

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
        })))
      }
      return Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    renderDetailPage()

    // 初始：running header 已渲染（PID 卡片），但 is_pausable=false → 暂停按钮不在
    await waitFor(() => expect(screen.getByText('取消任务')).toBeInTheDocument())
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
