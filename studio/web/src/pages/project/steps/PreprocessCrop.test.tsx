import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type CropWorkspaceItem, type Job } from '../../../api/client'
import PreprocessCropPage from './PreprocessCrop'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  reload: vi.fn(async () => undefined),
  onEvent: null as ((evt: Record<string, unknown>) => void) | null,
}))

vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../lib/useEventStream', () => ({
  useEventStream: (handler: (evt: Record<string, unknown>) => void) => { mocks.onEvent = handler },
}))
vi.mock('../../../components/preprocess/FreeCropEditor', () => ({
  default: ({ onCreate }: { onCreate: (rect: { x: number; y: number; w: number; h: number }) => void }) => (
    <div data-testid="crop-canvas">
      <button type="button" onClick={() => onCreate({ x: 0.1, y: 0.2, w: 0.5, h: 0.6 })}>Draw crop</button>
    </div>
  ),
}))
vi.mock('../../../lib/cropClustering', () => ({
  clusterByAspectRatio: () => ({
    kUsed: 1,
    assignments: [{
      id: '1_data/a.png',
      skipped: false,
      targetAr: { w: 1, h: 1 },
      rect: { x: 0, y: 0, w: 1, h: 1 },
    }],
  }),
}))

const image: CropWorkspaceItem = {
  name: '1_data/a.png', source: 'a.png', w: 1000, h: 800,
  mtime: 1, size: 100, processed: false, mask_mtime: null,
}
const job: Job = {
  id: 7, project_id: 1, version_id: 2, kind: 'preprocess', params: '{}', params_decoded: { stage: 'crop' },
  status: 'running', started_at: null, finished_at: null, pid: null,
  log_path: null, error_msg: null,
}

function renderPage() {
  return render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }} initialEntries={['/crop']}>
      <Routes>
        <Route element={<Outlet context={{ project: { id: 1 }, activeVersion: { id: 2 }, reload: mocks.reload }} />}>
          <Route path="/crop" element={<PreprocessCropPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  vi.spyOn(api, 'listCropWorkspaceTrain').mockResolvedValue({ images: [image] })
  vi.spyOn(api, 'getPreprocessStatusTrain').mockResolvedValue({ job: null, log_tail: '', summary: { image_count: 1 } })
  vi.spyOn(api, 'startPreprocessCropTrain').mockResolvedValue(job)
  vi.spyOn(api, 'cancelJob').mockResolvedValue({ task_id: job.id, canceled: true })
})

describe('Preprocess crop contracts', () => {
  it('keeps the filter in the Filmstrip header and outside its scrollport', async () => {
    const { container } = renderPage()
    const filmstrip = await screen.findByRole('group', { name: '裁剪工作集图片' })
    const filter = within(filmstrip).getByRole('radiogroup', { name: '裁剪图片筛选' })
    const scrollport = container.querySelector('.overflow-y-auto')

    expect(filmstrip).toContainElement(filter)
    expect(scrollport).not.toContainElement(filter)
    expect(filter).not.toHaveClass('ui-selection-content')
  })

  it('demotes the optional note and preserves the current-image crop payload', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('button', { name: '编辑 1_data/a.png 的裁剪框' })
    expect(screen.queryByRole('textbox', { name: '可选备注' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Draw crop' }))
    const note = screen.getByRole('textbox', { name: '可选备注' })
    expect(note).toHaveValue('裁剪 1')
    expect(note).toHaveAccessibleDescription('不会影响输出文件名；输出仍按 _c0、_c1… 编号。')
    expect(screen.getByText('500×480 px')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '裁剪当前图' }))
    expect(api.startPreprocessCropTrain).toHaveBeenCalledWith(1, 2, {
      '1_data/a.png': [{ x: 0.1, y: 0.2, w: 0.5, h: 0.6, label: '裁剪 1' }],
    })
  })

  it.each(['upscale', 'head_mask', undefined])('filters foreign or legacy stage %s before displaying logs/cancel', async (stage) => {
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({ job: { ...job, params_decoded: stage ? { stage } : {} }, log_tail: 'foreign stage log', summary: { image_count: 1 } })
    renderPage()
    await screen.findByRole('group', { name: '裁剪工作集图片' })
    expect(api.getPreprocessStatusTrain).toHaveBeenCalledWith(1, 2, 'crop')
    expect(screen.queryByText('foreign stage log')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '取消' })).not.toBeInTheDocument()
    expect(api.cancelJob).not.toHaveBeenCalled()
  })

  it('cancels only the matching crop job', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({ job, log_tail: 'crop log', summary: { image_count: 1 } })
    renderPage()
    await user.click(await screen.findByRole('button', { name: '取消' }))
    expect(api.cancelJob).toHaveBeenCalledWith(job.id)
  })

  it('keeps cluster cancel non-submitting and submits the modal with Enter', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('button', { name: '编辑 1_data/a.png 的裁剪框' })

    await user.click(screen.getByRole('button', { name: '按比例预填' }))
    let dialog = screen.getByRole('dialog', { name: '按比例预填' })
    await user.click(within(dialog).getByRole('button', { name: '取消' }))
    expect(screen.queryByRole('dialog', { name: '按比例预填' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '裁剪当前图' })).toBeDisabled()

    await user.click(screen.getByRole('button', { name: '按比例预填' }))
    dialog = screen.getByRole('dialog', { name: '按比例预填' })
    const firstField = within(dialog).getAllByRole('spinbutton')[0]
    firstField.focus()
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: '按比例预填' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: '裁剪当前图' })).toBeEnabled()
  })

  it('crop-active job completion keeps unsubmitted rects on other images', async () => {
    const user = userEvent.setup()
    const other: CropWorkspaceItem = { ...image, name: '1_data/b.png', source: 'b.png' }
    vi.mocked(api.listCropWorkspaceTrain).mockResolvedValue({ images: [image, other] })
    renderPage()
    await screen.findByRole('button', { name: '编辑 1_data/a.png 的裁剪框' })

    await user.click(screen.getByRole('button', { name: 'Draw crop' }))
    await user.click(screen.getByRole('button', { name: '编辑 1_data/b.png 的裁剪框' }))
    await user.click(screen.getByRole('button', { name: 'Draw crop' }))
    await user.click(screen.getByRole('button', { name: '编辑 1_data/a.png 的裁剪框' }))
    expect(screen.getByRole('button', { name: '裁剪全部 (2)' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '裁剪当前图' }))
    expect(api.startPreprocessCropTrain).toHaveBeenCalledWith(1, 2, {
      '1_data/a.png': [{ x: 0.1, y: 0.2, w: 0.5, h: 0.6, label: '裁剪 1' }],
    })

    act(() => { mocks.onEvent?.({ type: 'job_state_changed', job_id: job.id, status: 'done' }) })
    expect(await screen.findByRole('button', { name: '裁剪全部 (1)' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '裁剪当前图' })).toBeDisabled()
  })
})
