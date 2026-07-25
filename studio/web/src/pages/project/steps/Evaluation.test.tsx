import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../../api/client'
import EvaluationPage from './Evaluation'

const project = { id: 7, slug: 'nyahu', title: 'Nyahu' }
const version = { id: 9, label: 'v2' }

function renderAt(url: string, activeVersion: unknown = version) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <Routes>
        <Route
          path="/projects/:pid/v/:vid/eval"
          element={<EvaluationPage />}
        />
      </Routes>
    </MemoryRouter>,
    // useOutletContext 在没有 Outlet 时抛错，所以直接 mock 掉它
  )
  void activeVersion
}

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useOutletContext: () => outletCtx }
})

let outletCtx: { project: unknown; activeVersion: unknown } = {
  project, activeVersion: version,
}

describe('版本级评估页', () => {
  beforeEach(() => {
    outletCtx = { project, activeVersion: version }
    vi.spyOn(api, 'listEvalMetrics').mockResolvedValue({
      metric_specs: {}, cache: {}, results: [], session: null,
    } as never)
    vi.spyOn(api, 'listEvalSessions').mockResolvedValue({ sessions: [] } as never)
  })
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('按 version 拉评估，不带 task_id —— 这是它和训练页面板的唯一区别', async () => {
    renderAt('/projects/7/v/9/eval')
    await waitFor(() => expect(api.listEvalMetrics).toHaveBeenCalled())
    expect(api.listEvalMetrics).toHaveBeenCalledWith(7, 9, undefined, undefined)
    expect(api.listEvalSessions).toHaveBeenCalledWith(7, 9, undefined)
  })

  it('?session= 深链直接打开那一次评估', async () => {
    renderAt('/projects/7/v/9/eval?session=42')
    await waitFor(() => expect(api.listEvalMetrics).toHaveBeenCalledWith(7, 9, undefined, 42))
  })

  it('没有选中版本时给提示而不是空面板', () => {
    outletCtx = { project, activeVersion: null }
    renderAt('/projects/7/v/9/eval', null)
    expect(screen.getByText(/先选一个版本/)).toBeInTheDocument()
    expect(api.listEvalMetrics).not.toHaveBeenCalled()
  })
})
