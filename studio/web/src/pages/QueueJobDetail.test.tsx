/** 0.17 P-G — 数据作业详情页：meta/参数（i18n 映射 + 原 key 兜底）/日志。 */
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../components/Dialog'
import { ToastProvider } from '../components/Toast'
import { api, type Job } from '../api/client'
import QueueJobDetailPage from './QueueJobDetail'

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: 7, project_id: 1, version_id: 2, kind: 'tag',
    params: '{}',
    params_decoded: { tagger: 'wd14', mystery_key: true },
    status: 'running', created_at: 900, started_at: 1000, finished_at: null,
    pid: 123, log_path: '/x/7.log', error_msg: null,
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
  vi.spyOn(api, 'getJobLog').mockResolvedValue({
    job_id: 7, content: 'hello log line', size: 14,
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function renderDetail(job: Job) {
  vi.spyOn(api, 'getJob').mockResolvedValue(job)
  return render(
    <MemoryRouter initialEntries={['/queue/jobs/7']}>
      <ToastProvider>
        <DialogProvider>
          <Routes>
            <Route path="/queue/jobs/:jid" element={<QueueJobDetailPage />} />
          </Routes>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('QueueJobDetailPage', () => {
  it('渲染 header + meta（含入队时间）+ 参数（映射标签 / 原 key 兜底）+ 日志', async () => {
    renderDetail(makeJob())

    await waitFor(() => expect(screen.getByText('#7')).toBeInTheDocument())
    // meta：入队时间标签 + 项目名
    expect(screen.getByText('入队时间')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('MyProj')).toBeInTheDocument())
    // 参数：tagger 有映射 → 「打标器」；mystery_key 无映射 → 原 key 兜底
    const params = screen.getByTestId('job-detail-params')
    expect(params).toHaveTextContent('打标器')
    expect(params).toHaveTextContent('wd14')
    expect(params).toHaveTextContent('mystery_key')
    // 布尔值人话化
    expect(params).toHaveTextContent('是')
    // 日志
    await waitFor(() =>
      expect(screen.getByTestId('job-detail-log')).toHaveTextContent('hello log line'),
    )
  })

  it('running 显示取消按钮；done 不显示', async () => {
    renderDetail(makeJob({ status: 'running' }))
    await waitFor(() => expect(screen.getByTestId('job-detail-cancel')).toBeInTheDocument())
  })

  it('终态没有取消按钮，仍有跳转', async () => {
    renderDetail(makeJob({ status: 'done', finished_at: 2000 }))
    await waitFor(() => expect(screen.getByTestId('job-detail-jump')).toBeInTheDocument())
    expect(screen.queryByTestId('job-detail-cancel')).not.toBeInTheDocument()
  })
})
