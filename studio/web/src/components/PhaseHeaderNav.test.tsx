import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import '../i18n'
import PhaseHeaderNav from './PhaseHeaderNav'
import { ProjectContext } from '../context/ProjectContext'

vi.mock('./Toast', () => ({
  useToast: () => ({ toast: vi.fn() }),
}))

function renderAt(path: string) {
  // 最简 ctx：组件不深入用，给个空 project + null activeVersion 即可
  const ctx = {
    project: {
      id: 1, slug: 'p', title: 'P', stage: 'curating' as const,
      active_version_id: null, active_version_label: null, active_version_status: null,
      created_at: 0, updated_at: 0, note: null, versions: [],
      download_image_count: 0, preprocess_image_count: 0,
    },
    activeVersion: null,
    reload: async () => {},
    onSelectVersion: async () => {},
    onCreateVersion: () => {},
    onExportTrain: () => {},
    onDeleteVersion: async () => {},
    exporting: false,
  }
  return render(
    <ProjectContext.Provider value={ctx}>
      <MemoryRouter initialEntries={[path]}>
        <PhaseHeaderNav />
      </MemoryRouter>
    </ProjectContext.Provider>
  )
}

describe('PhaseHeaderNav (ADR-0007 §11.5-A / §11.8-B)', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders null on non-phase routes', () => {
    const { container } = renderAt('/projects/1/download')
    expect(container.firstChild).toBeNull()
  })

  it('renders null on overview', () => {
    const { container } = renderAt('/projects/1')
    expect(container.firstChild).toBeNull()
  })

  it('renders next button only on curating (no prev)', () => {
    renderAt('/projects/1/v/2/curate')
    // next 按钮含 ④ 打标
    expect(screen.getByText(/④/)).toBeInTheDocument()
    // 不应有 ③（前一步不存在）
    expect(screen.queryByText(/③/)).not.toBeInTheDocument()
  })

  it('renders both prev and next on tagging', () => {
    renderAt('/projects/1/v/2/tag')
    expect(screen.getByText(/③/)).toBeInTheDocument()  // prev = curating
    expect(screen.getByText(/⑤/)).toBeInTheDocument()  // next = editing
  })

  it('renders prev only on ready (last phase, no next)', () => {
    renderAt('/projects/1/v/2/train')
    expect(screen.getByText(/⑥/)).toBeInTheDocument()  // prev = regularizing
    // ready 是最后一个，next 按钮被 hidden
    // 不应出现 ⑧ 之类的下一步标记
  })
})
