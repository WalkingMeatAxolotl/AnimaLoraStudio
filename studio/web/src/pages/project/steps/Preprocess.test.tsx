import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type Job, type ModelsCatalog, type TrainImage, type UpscalerVariant } from '../../../api/client'
import PreprocessPage from './Preprocess'

const mocks = vi.hoisted(() => ({ toast: vi.fn(), reload: vi.fn() }))
vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return {
    ...actual,
    useToast: () => ({ toast: mocks.toast }),
    useOptionalToast: () => ({ toast: mocks.toast }),
  }
})
vi.mock('../../../lib/useEventStream', () => ({ useEventStream: vi.fn() }))
// Only the parameter/action contract is under test here. ImageGrid's own tests
// cover selection and virtualization; no synthetic geometry is asserted here.
vi.mock('../../../components/ImageGrid', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/ImageGrid')>()
  return { ...actual, default: ({ items }: { items: { name: string }[] }) => <div role="grid">{items.map(item => <span key={item.name}>{item.name}</span>)}</div> }
})

const variant = (label: string, exists = true, kind: 'preset' | 'custom' = 'preset'): UpscalerVariant => ({
  label, exists, kind, filename: `${label}.pth`, hf_repo: 'models/upscaler', ms_repo: null,
  size_mb: 64, description: '', target_path: `/models/${label}.pth`, is_current: false, size: 0, mtime: 0,
})
const job: Job = { id: 9, project_id: 1, version_id: 2, kind: 'preprocess', params: '{}', status: 'running', started_at: null, finished_at: null, pid: null, log_path: null, error_msg: null }
// Fixtures project only fields consumed by this page; unrelated catalog families
// and project metadata intentionally remain outside this test's scope.
const catalog = (variants = [variant('4x-AnimeSharp'), variant('Other')]): ModelsCatalog => ({
  upscalers: { variants, current: variants[0].label },
}) as ModelsCatalog
const images = ['768px_group/a.png', '1024px_group/b.png'].map(name => ({
  name, origin: name.slice(name.lastIndexOf('/') + 1), size: 100, w: 512, h: 512,
  mtime: 1, duplicate_removed: false, processed: false,
})) as TrainImage[]

function renderPage() {
  return render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }} initialEntries={['/upscale']}>
    <Routes><Route element={<Outlet context={{ project: { id: 1 }, activeVersion: { id: 2 }, reload: mocks.reload }} />}>
      <Route path="/upscale" element={<PreprocessPage />} />
    </Route></Routes>
  </MemoryRouter>)
}

async function ready() {
  await waitFor(() => expect(screen.getByRole('button', { name: '放大全部 2' })).toBeEnabled())
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  vi.spyOn(api, 'listPreprocessFilesTrain').mockResolvedValue({ images, summary: { image_count: 2 } })
  vi.spyOn(api, 'getPreprocessStatusTrain').mockResolvedValue({ job: null, log_tail: '', summary: { image_count: 2 } })
  vi.spyOn(api, 'getModelsCatalog').mockResolvedValue(catalog())
  vi.spyOn(api, 'selectUpscaler').mockImplementation(async (label) => ({ selected: label }))
  vi.spyOn(api, 'startPreprocessTrain').mockResolvedValue(job)
  vi.spyOn(api, 'cancelJob').mockResolvedValue({ task_id: job.id, canceled: true })
  vi.spyOn(api, 'startModelDownload').mockResolvedValue({} as Awaited<ReturnType<typeof api.startModelDownload>>)
})

describe('Preprocess upscale controls', () => {
  it('uses named shared controls and separates the custom input label from its preset', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()
    const settings = screen.getByRole('region', { name: '放大设置' })
    expect(settings).toHaveClass('card', 'card-compact')
    const preset = within(settings).getByRole('combobox', { name: '目标分辨率' })
    await user.selectOptions(preset, 'custom')
    const custom = within(settings).getByRole('spinbutton', { name: '自定义目标边长（像素）' })
    expect(custom).toHaveClass('form-control', 'form-control-sm', 'form-control-mono')
    expect(custom).toHaveAttribute('min', '256')
    expect(custom).toHaveAttribute('max', '4096')
    expect(preset).toHaveAccessibleName('目标分辨率')
    expect(custom).toHaveAttribute('aria-describedby', preset.getAttribute('aria-describedby'))
    for (const select of within(settings).getAllByRole('combobox')) {
      expect(select).toHaveClass('form-control', 'form-control-sm')
      expect(select.style.padding).toBe('')
    }
    const actions = screen.getByRole('group', { name: '放大操作' })
    expect(within(actions).getAllByRole('button').map(button => button.textContent)).toEqual(['放大全部 2', '放大选中 0'])
    expect(within(actions).getByRole('button', { name: '放大选中 0' })).toBeDisabled()
  })

  it('preserves defaults when submitting all and disables controls while the job runs', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()
    await user.click(screen.getByRole('button', { name: '放大全部 2' }))
    expect(api.startPreprocessTrain).toHaveBeenCalledWith(1, 2, {
      mode: 'all', names: undefined, model: '4x-AnimeSharp', tile_size: 256, device: 'auto', target_area: 1024 ** 2,
    })
    await waitFor(() => expect(screen.getByRole('button', { name: '放大全部 2' })).toBeDisabled())
    for (const select of within(screen.getByRole('region', { name: '放大设置' })).getAllByRole('combobox')) expect(select).toBeDisabled()
  })

  it('keeps model, custom area, tile, device and selected rel paths in the submission', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()
    await user.selectOptions(screen.getByRole('combobox', { name: '模型' }), 'Other')
    expect(api.selectUpscaler).toHaveBeenCalledWith('Other')
    await user.selectOptions(screen.getByRole('combobox', { name: '目标分辨率' }), 'custom')
    const custom = screen.getByRole('spinbutton')
    await user.clear(custom)
    await user.type(custom, '1280')
    await user.selectOptions(screen.getByRole('combobox', { name: 'tile' }), '192')
    await user.selectOptions(screen.getByRole('combobox', { name: '设备' }), 'cpu')
    await user.click(screen.getByRole('button', { name: '全选' }))
    await user.click(screen.getByRole('button', { name: '放大选中 2' }))
    expect(api.startPreprocessTrain).toHaveBeenCalledWith(1, 2, {
      mode: 'selected', names: expect.arrayContaining(images.map(image => image.name)), model: 'Other', tile_size: 192, device: 'cpu', target_area: 1280 ** 2,
    })
  })

  it('rejects out-of-range custom edges without submitting a job', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()
    await user.selectOptions(screen.getByRole('combobox', { name: '目标分辨率' }), 'custom')
    await user.clear(screen.getByRole('spinbutton'))
    await user.type(screen.getByRole('spinbutton'), '100')
    await user.click(screen.getByRole('button', { name: '放大全部 2' }))
    expect(api.startPreprocessTrain).not.toHaveBeenCalled()
    expect(mocks.toast).toHaveBeenCalledWith(expect.any(String), 'error')
  })

  it('keeps folder-scoped all mode and follows the folder target resolution', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()
    await user.selectOptions(screen.getByRole('combobox', { name: '文件夹' }), '768px_group')
    expect(screen.getByRole('combobox', { name: '目标分辨率' })).toHaveValue('768')
    await user.click(screen.getByRole('button', { name: '放大全部 1' }))
    expect(api.startPreprocessTrain).toHaveBeenCalledWith(1, 2, expect.objectContaining({ mode: 'selected', names: ['768px_group/a.png'], target_area: 768 ** 2 }))
  })

  it('keeps off mode as null target area', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()
    await user.selectOptions(screen.getByRole('combobox', { name: '目标分辨率' }), 'off')
    await user.click(screen.getByRole('button', { name: '放大全部 2' }))
    expect(api.startPreprocessTrain).toHaveBeenCalledWith(1, 2, expect.objectContaining({ target_area: null }))
  })

  it('presents missing preset guidance and sends the existing model download payload', async () => {
    vi.mocked(api.getModelsCatalog).mockResolvedValue(catalog([variant('4x-AnimeSharp', false)]))
    let rejectDownload!: (reason: Error) => void
    vi.mocked(api.startModelDownload).mockReturnValue(new Promise((_, reject) => { rejectDownload = reject }))
    const user = userEvent.setup()
    renderPage()
    await screen.findByText('models/upscaler · ~64 MB')
    expect(screen.getByText('需要下载模型').closest('.alert')).toHaveClass('alert-warning', 'alert-sm')
    expect(screen.getByRole('button', { name: '放大全部 2' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /下载.*4x-AnimeSharp/ }))
    expect(api.startModelDownload).toHaveBeenCalledWith({ model_id: 'upscaler', variant: '4x-AnimeSharp' })
    expect(screen.getByRole('button', { name: '下载中...' })).toBeDisabled()
    await act(async () => rejectDownload(new Error('offline')))
    expect(mocks.toast).toHaveBeenCalledWith('Error: offline', 'error')
  })

  it('does not offer downloading a missing custom model', async () => {
    vi.mocked(api.getModelsCatalog).mockResolvedValue(catalog([variant('Local', false, 'custom')]))
    renderPage()
    const download = await screen.findByRole('button', { name: /下载.*Local/ })
    expect(download).toBeDisabled()
    expect(api.startModelDownload).not.toHaveBeenCalled()
  })
})
