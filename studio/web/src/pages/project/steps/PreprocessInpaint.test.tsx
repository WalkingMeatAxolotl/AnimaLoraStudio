import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type CropWorkspaceItem } from '../../../api/client'
import PreprocessInpaintPage from './PreprocessInpaint'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(), reload: vi.fn(async () => undefined),
  canvasProps: [] as Array<Record<string, unknown>>,
}))

vi.mock('../../../lib/SettingsData', () => ({ useSettingsData: () => ({
  catalog: {
    head_detector: { current: 'builtin' },
    model_sources: { head_detector: [{
      kind: 'preset', candidate: null, value: 'builtin', label: 'Built-in detector',
      description: '', download_id: 'head_detector', download_variant: null,
      status_key: 'head_detector', exists: true, size: 10, files: null,
      size_estimate: 0, is_current: true, removable: false, deletable: true, extra: {},
    }] },
    downloads: {},
  },
  catalogError: null, downloadBusy: new Set(), downloadErrors: {}, reloadCatalog: vi.fn(),
}) }))

vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../components/preprocess/InpaintCanvas', async () => {
  const React = await import('react')
  const Canvas = React.forwardRef(function Canvas(
    props: {
      mode: 'paint' | 'mask'
      erase: boolean
      maskEdits: object[]
      onStrokeEnd: (stroke: object) => void
      onMaskStrokeEnd: (stroke: object) => void
    },
    ref: React.ForwardedRef<{
      exportBlob: () => Promise<Blob | null>
      exportMaskBlob: () => Promise<{ blob: Blob; coverage: number } | null>
    }>,
  ) {
    React.useImperativeHandle(ref, () => ({
      exportBlob: async () => new Blob(['paint'], { type: 'image/png' }),
      exportMaskBlob: async () => ({ blob: new Blob(['mask'], { type: 'image/png' }), coverage: 0.2 }),
    }))
    mocks.canvasProps.push(props as unknown as Record<string, unknown>)
    const stroke = { color: '#ffffff', size: 24, hardness: 1, points: [{ x: 10, y: 10 }] }
    return (
      <div data-testid="inpaint-canvas" data-mode={props.mode} data-erase={String(props.erase)}
        data-mask-edits={JSON.stringify(props.maskEdits)}>
        <button
        type="button"
        onClick={() => props.mode === 'mask'
          ? props.onMaskStrokeEnd(stroke)
          : props.onStrokeEnd(stroke)}
      >
        Draw stroke
      </button>
      </div>
    )
  })
  return {
    default: Canvas,
    renderInpaintedBlob: vi.fn(async () => new Blob(['paint'], { type: 'image/png' })),
    renderMaskBlob: vi.fn(async () => ({ blob: new Blob(['mask'], { type: 'image/png' }), coverage: 0.2 })),
  }
})

const image: CropWorkspaceItem = {
  name: '1_data/a.png', source: 'a.png', w: 1000, h: 800,
  mtime: 1, size: 100, processed: false, mask_mtime: null,
}

function renderPage() {
  return render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }} initialEntries={['/inpaint']}>
      <Routes>
        <Route element={<Outlet context={{ project: { id: 1 }, activeVersion: { id: 2 }, reload: mocks.reload }} />}>
          <Route path="/inpaint" element={<PreprocessInpaintPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  mocks.canvasProps.length = 0
  window.localStorage.clear()
  vi.spyOn(api, 'getPreprocessStatusTrain').mockResolvedValue({ job: null, log_tail: '', summary: { image_count: 1 } })
  vi.spyOn(api, 'getJob').mockRejectedValue(new Error('not polled'))
  vi.spyOn(api, 'startHeadMaskDetection').mockRejectedValue(new Error('not started'))
  vi.spyOn(api, 'getHeadMaskProposals').mockRejectedValue(new Error('not loaded'))
  vi.spyOn(api, 'listCropWorkspaceTrain').mockResolvedValue({ images: [image] })
  vi.spyOn(api, 'saveInpaintTrain').mockResolvedValue({ ...image, origin: image.source })
  vi.spyOn(api, 'saveMaskTrain').mockResolvedValue({ name: image.name, mtime: 2, size: 10 })
  vi.spyOn(api, 'deleteMaskTrain').mockResolvedValue({ deleted: true })
  vi.spyOn(api, 'applyHeadMaskProposals')
  vi.spyOn(api, 'undoHeadMaskApply')
})

describe('Preprocess inpaint contracts', () => {
  it('keeps the unsaved-state filter in Filmstrip and exposes mode/tool controls', async () => {
    const { container } = renderPage()
    const filmstrip = await screen.findByRole('group', { name: '涂抹工作集图片' })
    const filter = within(filmstrip).getByRole('radiogroup', { name: '按未保存状态筛选' })
    const scrollport = container.querySelector('.overflow-y-auto')

    expect(filmstrip).toContainElement(filter)
    expect(scrollport).not.toContainElement(filter)
    expect(filter).toHaveClass('ui-selection-content')
    expect(screen.getByRole('radiogroup', { name: '模式' })).toBeInTheDocument()
    expect(screen.getByRole('radiogroup', { name: '工具' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存当前图' })).toBeDisabled()
  })

  it('opens setup to explain the save prerequisite while unsaved edits block Start', async () => {
    let finishWorkspace!: (value: { images: CropWorkspaceItem[] }) => void
    vi.mocked(api.listCropWorkspaceTrain).mockReturnValue(new Promise((resolve) => {
      finishWorkspace = resolve
    }))
    const user = userEvent.setup()
    renderPage()
    expect(screen.getByText('加载工作集...')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Draw stroke' })).not.toBeInTheDocument()

    await act(async () => finishWorkspace({ images: [image] }))
    await user.click(await screen.findByRole('button', { name: 'Draw stroke' }))
    const autoMask = screen.getByRole('button', { name: '自动遮罩' })
    expect(autoMask).toBeEnabled()
    await user.click(autoMask)

    expect(screen.getByText('还有 1 张图片存在未保存修改，请先保存再启动自动遮罩。')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始' })).toBeDisabled()
  })

  it('saves a mask-only edit without overwriting the source image', async () => {
    const user = userEvent.setup()
    renderPage()

    const drawStroke = await screen.findByRole('button', { name: 'Draw stroke' })
    await user.click(within(screen.getByRole('radiogroup', { name: '模式' })).getByRole('radio', { name: '训练遮罩' }))
    await user.click(drawStroke)
    expect(screen.getByRole('button', { name: '保存当前图' })).toBeEnabled()
    expect(screen.getByRole('radio', { name: '待保存 1' })).toHaveAttribute('aria-checked', 'false')

    await user.click(screen.getByRole('button', { name: '保存当前图' }))
    await waitFor(() => expect(api.saveMaskTrain).toHaveBeenCalledWith(1, 2, image.name, expect.any(Blob)))
    expect(api.saveInpaintTrain).not.toHaveBeenCalled()
    await waitFor(() => expect(screen.getByRole('button', { name: '保存当前图' })).toBeDisabled())
  })

  it('places setup before save-all/current and removes the transient undetected filter', async () => {
    const user = userEvent.setup()
    renderPage()
    const filmstrip = await screen.findByRole('group', { name: '涂抹工作集图片' })
    const actions = screen.getByRole('group', { name: '涂抹保存操作' })
    expect(within(actions).getAllByRole('button').map((button) => button.textContent)).toEqual([
      '自动遮罩', '保存全部 (0)', '保存当前图',
    ])
    expect(within(filmstrip).getAllByRole('radio')).toHaveLength(3)
    const autoMask = screen.getByRole('button', { name: '自动遮罩' })
    expect(autoMask).toHaveClass('btn-ghost')
    await user.click(autoMask)
    expect(screen.getByRole('dialog', { name: '自动遮罩' })).toBeInTheDocument()
    expect(screen.getByLabelText('识别模型')).toHaveValue('builtin')
    expect(screen.queryByText('高级参数')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('NMS IoU')).not.toBeInTheDocument()
    await user.keyboard('{Escape}')
    expect(screen.getByRole('button', { name: '自动遮罩' })).toHaveFocus()
  })

  it('does not resurrect an already-completed job on initial mount', async () => {
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({ job: {
      id: 40, project_id: 1, version_id: 2, kind: 'preprocess', params_decoded: { stage: 'head_mask' },
      params: '{}', status: 'done', started_at: null, finished_at: null, pid: null, log_path: null, error_msg: null,
    }, log_tail: '', summary: { image_count: 1 } })
    renderPage()
    await screen.findByRole('group', { name: '涂抹工作集图片' })
    await waitFor(() => expect(api.getPreprocessStatusTrain).toHaveBeenCalled())
    expect(api.getHeadMaskProposals).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: '保存全部 (0)' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '保存当前图' })).toBeDisabled()
    expect(mocks.toast).not.toHaveBeenCalled()
  })

  it.each(['save', 'discard'] as const)(
    'adds a session job once and does not resurrect it after %s and remount',
    async (action) => {
    const images = [image, { ...image, name: '1_data/b.png' }, { ...image, name: '1_data/c.png' }]
    const completedJob = {
      id: 41, project_id: 1, version_id: 2, kind: 'preprocess' as const, params_decoded: { stage: 'head_mask' },
      params: '{}', status: 'done' as const, started_at: null, finished_at: null, pid: null, log_path: null, error_msg: null,
    }
    vi.mocked(api.listCropWorkspaceTrain).mockResolvedValue({ images })
    vi.mocked(api.startHeadMaskDetection).mockResolvedValue(completedJob)
    vi.mocked(api.getHeadMaskProposals).mockResolvedValue({
      schema_version: 2, job_id: 41, status: 'partial', created_at: 1, stale_count: 1, undo_available: false,
      model: { revision: 'fixed', path: 'model.onnx', input_size: [640, 640], provider: 'CPU' },
      parameters: { confidence: 0.413, iou_threshold: 0.7, padding_ratio: 0.1, feather_ratio: 0.03 },
      images: [
        { name: image.name, status: 'done', regions: [
          { id: 'a', score: 0.9, box: [10, 10, 20, 20], mask_region: { x1: 8, y1: 8, x2: 22, y2: 22, feather_x: 2, feather_y: 3 } },
          { id: 'b', score: 0.8, box: [30, 30, 40, 40], mask_region: { x1: 28, y1: 28, x2: 42, y2: 42, feather_x: 2, feather_y: 3 } },
        ], size: [1000, 800], source_mtime_ns: 1, source_file_size: 100, stale: false, stale_reason: null },
        { name: '1_data/b.png', status: 'done', regions: [
          { id: 'c', score: 0.7, box: [1, 1, 3, 3], mask_region: { x1: 0, y1: 0, x2: 4, y2: 4, feather_x: 1, feather_y: 1 } },
        ], size: [1000, 800], source_mtime_ns: 1, source_file_size: 100, stale: false, stale_reason: null },
        { name: '1_data/c.png', status: 'done', regions: [
          { id: 'stale', score: 0.9, box: [1, 1, 3, 3], mask_region: { x1: 0, y1: 0, x2: 4, y2: 4, feather_x: 1, feather_y: 1 } },
        ], size: [1000, 800], source_mtime_ns: 1, source_file_size: 100, stale: true, stale_reason: 'changed' },
      ],
    })
    const view = renderPage()
    await screen.findByRole('group', { name: '涂抹工作集图片' })
    await userEvent.click(screen.getByRole('button', { name: '自动遮罩' }))
    await userEvent.click(screen.getByRole('button', { name: '开始' }))

    await waitFor(() => expect(screen.getByRole('button', { name: '保存全部 (2)' })).toBeEnabled())
    expect(screen.getByTestId('inpaint-canvas')).toHaveAttribute('data-mode', 'mask')
    expect(screen.getByRole('radio', { name: '画笔' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByTestId('inpaint-canvas').getAttribute('data-mask-edits')).toContain('"type":"auto"')
    expect(screen.getByTestId('inpaint-canvas').getAttribute('data-mask-edits')).toContain('"feather_y":3')
    expect(mocks.toast).toHaveBeenCalledWith(
      '已为 2 张图片添加 3 个遮罩。 未应用：1 张源图片已变化。',
      'info',
    )

    await userEvent.click(screen.getByRole('button', { name: '撤销' }))
    expect(screen.getByRole('button', { name: '保存全部 (1)' })).toBeEnabled()
    await userEvent.click(screen.getByRole('button', { name: '重做' }))
    expect(screen.getByRole('button', { name: '保存全部 (2)' })).toBeEnabled()

    const { renderMaskBlob } = await import('../../../components/preprocess/InpaintCanvas')
    if (action === 'save') {
      await userEvent.click(screen.getByRole('button', { name: '保存全部 (2)' }))
      await waitFor(() => expect(renderMaskBlob).toHaveBeenCalledTimes(2))
      expect(renderMaskBlob).toHaveBeenCalledWith(null, 1000, 800, [expect.objectContaining({
        type: 'auto', regions: expect.arrayContaining([expect.objectContaining({ feather_x: 2, feather_y: 3 })]),
      })])
    } else {
      expect(renderMaskBlob).not.toHaveBeenCalled()
    }
    expect(api.applyHeadMaskProposals).not.toHaveBeenCalled()
    expect(api.undoHeadMaskApply).not.toHaveBeenCalled()

    view.unmount()
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({
      job: completedJob, log_tail: '', summary: { image_count: 3 },
    })
    renderPage()
    await screen.findByRole('group', { name: '涂抹工作集图片' })
    await waitFor(() => expect(api.getPreprocessStatusTrain).toHaveBeenCalledTimes(2))
    expect(api.getHeadMaskProposals).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: '保存全部 (0)' })).toBeDisabled()
    expect(mocks.toast.mock.calls.filter(([message]) => String(message).includes('已为 2 张图片添加 3 个遮罩'))).toHaveLength(1)
    },
  )

  it('ignores a late workspace response after unmount', async () => {
    let finish!: (value: { images: CropWorkspaceItem[] }) => void
    vi.mocked(api.listCropWorkspaceTrain).mockReturnValue(new Promise((resolve) => { finish = resolve }))
    const view = renderPage()
    view.unmount()
    await act(async () => finish({ images: [image] }))
    expect(mocks.reload).not.toHaveBeenCalled()
    expect(mocks.toast).not.toHaveBeenCalled()
  })

  it('only references the mounted recent-color disclosure region', async () => {
    window.localStorage.setItem('studio:inpaint:recent_colors', JSON.stringify(['#ff0000']))
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('group', { name: '涂抹工作集图片' })

    const recent = screen.getByRole('button', { name: '最近' })
    expect(recent).toHaveAttribute('aria-expanded', 'false')
    expect(recent).not.toHaveAttribute('aria-controls')
    await user.click(recent)
    expect(recent).toHaveAttribute('aria-expanded', 'true')
    expect(recent).toHaveAttribute('aria-controls', 'inpaint-recent-colors')
    expect(screen.getByRole('group', { name: '历史颜色' })).toHaveAttribute('id', 'inpaint-recent-colors')
  })
})
