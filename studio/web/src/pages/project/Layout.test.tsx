/**
 * ProjectLayout 版本切换回归测试（issue #386）。
 *
 * 场景：切换版本后 activate 请求还在途时立即点「开始训练」，Train 页读到的
 * activeVersion 必须已经是新版本（乐观更新），否则会把旧版本入队。
 */
import { act, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useOutletContext } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  ProjectSetterContext,
  type ProjectCtxValue,
} from '../../context/ProjectContext'
import type { ProjectDetail, Version } from '../../api/client'
import { DialogProvider } from '../../components/Dialog'

const toastMock = vi.fn()
vi.mock('../../components/Toast', () => ({
  useToast: () => ({ toast: toastMock }),
}))

vi.mock('../../lib/useEventStream', () => ({
  useEventStream: () => {},
}))

const getProjectMock = vi.fn()
const activateVersionMock = vi.fn()
vi.mock('../../api/client', () => ({
  api: {
    getProject: (pid: number) => getProjectMock(pid),
    activateVersion: (pid: number, vid: number) => activateVersionMock(pid, vid),
  },
}))

import ProjectLayout from './Layout'

function makeVersion(id: number, label: string): Version {
  return {
    id, project_id: 3, label, config_name: null, status: 'preparing',
    phase: 'curating', last_failure_reason: null, created_at: 0,
    output_lora_path: null, note: null, trigger_word: '',
  }
}

const V1 = makeVersion(1, 'v1')
const V2 = makeVersion(2, 'v2')
const V3 = makeVersion(3, 'v3')

function makeProject(activeVid: number, versions: Version[]): ProjectDetail {
  return {
    id: 3, slug: 'ganyu', title: '甘雨', active_version_id: activeVid,
    active_version_label: 'v2', active_version_status: 'preparing',
    active_version_phase: 'curating', created_at: 0, updated_at: 0,
    archived_at: null, note: null, versions,
    download_image_count: 0, preprocess_image_count: 0,
  }
}

/** 模拟 Train 等步骤页：从 Outlet context 读 activeVersion（Train.tsx 入队用的就是它）。 */
function Probe() {
  const { activeVersion } = useOutletContext<{ activeVersion: Version | null }>()
  return <div data-testid="active-vid">{activeVersion ? String(activeVersion.id) : 'none'}</div>
}

let lastCtx: ProjectCtxValue | null = null

function renderLayout() {
  return render(
    <MemoryRouter
      initialEntries={['/projects/3']}
      future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
    >
      <DialogProvider>
        <ProjectSetterContext.Provider value={(v) => { lastCtx = v }}>
          <Routes>
            <Route path="/projects/:pid" element={<ProjectLayout />}>
              <Route index element={<Probe />} />
            </Route>
          </Routes>
        </ProjectSetterContext.Provider>
      </DialogProvider>
    </MemoryRouter>
  )
}

function deferred<T>() {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

describe('ProjectLayout 版本切换（#386）', () => {
  beforeEach(() => {
    lastCtx = null
    toastMock.mockClear()
    getProjectMock.mockReset()
    activateVersionMock.mockReset()
    getProjectMock.mockResolvedValue(makeProject(2, [V1, V2]))
  })

  it('activate 在途时 activeVersion 已乐观切到新版本', async () => {
    const d = deferred<ProjectDetail>()
    activateVersionMock.mockReturnValue(d.promise)
    renderLayout()
    await screen.findByTestId('active-vid')
    expect(screen.getByTestId('active-vid')).toHaveTextContent('2')

    act(() => { lastCtx!.onSelectVersion(1) })
    // 后端未返回，本地已经是 v1 —— 此刻点「开始训练」入队的就是 v1。
    expect(screen.getByTestId('active-vid')).toHaveTextContent('1')

    await act(async () => { d.resolve(makeProject(1, [V1, V2])) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('1')
  })

  it('activate 失败回滚到原版本并 toast', async () => {
    const d = deferred<ProjectDetail>()
    activateVersionMock.mockReturnValue(d.promise)
    renderLayout()
    await screen.findByTestId('active-vid')

    act(() => { lastCtx!.onSelectVersion(1) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('1')

    await act(async () => { d.reject(new Error('boom')) })
    await waitFor(() => expect(screen.getByTestId('active-vid')).toHaveTextContent('2'))
    expect(toastMock).toHaveBeenCalled()
  })

  it('快速连切两次，第一次失败不覆盖第二次的选择', async () => {
    getProjectMock.mockResolvedValue(makeProject(2, [V1, V2, V3]))
    const d1 = deferred<ProjectDetail>()
    const d2 = deferred<ProjectDetail>()
    activateVersionMock.mockImplementation((_pid: number, vid: number) =>
      vid === 1 ? d1.promise : d2.promise)
    renderLayout()
    await screen.findByTestId('active-vid')

    act(() => { lastCtx!.onSelectVersion(1) })
    act(() => { lastCtx!.onSelectVersion(3) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('3')

    // 第一次切换的失败先落地：序号已过期，既不回滚也不 toast。
    await act(async () => { d1.reject(new Error('stale boom')) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('3')
    expect(toastMock).not.toHaveBeenCalled()

    await act(async () => { d2.resolve(makeProject(3, [V1, V2, V3])) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('3')
  })
})
