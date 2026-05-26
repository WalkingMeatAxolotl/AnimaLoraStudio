/** QueueDetail page 组件级 regression test。
 *
 *  目前只覆盖 SnapshotConfigTab 的 refetch trap：父组件每 2s 浅 clone task
 *  做 elapsed time tick，旧实现 [task] 作 deps 会让 snapshot config 也跟着
 *  2s 重拉 —— 浏览器卡顿、loading flash。 */
import { render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ToastProvider } from '../components/Toast'
import type { Task } from '../api/client'
import { SnapshotConfigTab } from './QueueDetail'

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
