import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api, type HeadMaskProposals, type Job, type ModelsCatalog } from '../../api/client'
import AutoHeadMaskPanel from './AutoHeadMaskPanel'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  onEvent: undefined as undefined | ((event: Record<string, unknown>) => void),
}))

vi.mock('../Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../lib/useEventStream', () => ({
  useEventStream: (callback: (event: Record<string, unknown>) => void) => {
    mocks.onEvent = callback
  },
}))

const job: Job = {
  id: 41, project_id: 2, version_id: 3, kind: 'preprocess', params: '{}',
  params_decoded: { stage: 'head_mask' }, status: 'running', started_at: null,
  finished_at: null, pid: null, log_path: null, error_msg: null,
}

const catalog = {
  head_detector: {
    id: 'head_detector', name: 'Anime Head Detector', description: '',
    repo: 'deepghs/anime_head_detection', revision: '06604f', target_path: 'model.onnx',
    expected_size: 44_585_386, expected_sha256: 'sha', exists: true, valid: true,
    size: 44_585_386, mtime: 1,
  },
  downloads: {},
} as unknown as ModelsCatalog

const proposals: HeadMaskProposals = {
  schema_version: 1,
  job_id: 41,
  model: { revision: '06604f', path: 'model.onnx', input_size: [640, 640], provider: 'CPUExecutionProvider' },
  parameters: { confidence: 0.413, iou_threshold: 0.7, padding_ratio: 0.1, feather_ratio: 0.03 },
  created_at: 1,
  stale_count: 0,
  undo_available: false,
  images: [
    {
      name: '1_data/A.png', size: [100, 100], source_mtime_ns: 1,
      source_file_size: 2, stale: false, stale_reason: null,
      regions: [
        { id: 'a', score: 0.9, box: [10, 10, 30, 30], mask_region: { x1: 8, y1: 8, x2: 32, y2: 32, feather_x: 1, feather_y: 1 } },
        { id: 'b', score: 0.8, box: [50, 10, 70, 30], mask_region: { x1: 48, y1: 8, x2: 72, y2: 32, feather_x: 1, feather_y: 1 } },
      ],
    },
    {
      name: '1_data/B.png', size: [100, 100], source_mtime_ns: 1,
      source_file_size: 2, stale: false, stale_reason: null, regions: [],
    },
  ],
}

function renderPanel(overrides: Partial<React.ComponentProps<typeof AutoHeadMaskPanel>> = {}) {
  return render(<AutoHeadMaskPanel
    projectId={2}
    versionId={3}
    activeName="1_data/A.png"
    unsavedCount={0}
    onStateChange={vi.fn()}
    onShowUndetected={vi.fn()}
    onWorkspaceChanged={vi.fn().mockResolvedValue(undefined)}
    {...overrides}
  />)
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  vi.spyOn(api, 'getModelsCatalog').mockResolvedValue(catalog)
  vi.spyOn(api, 'getPreprocessStatusTrain').mockResolvedValue({ job: null, log_tail: '', summary: { image_count: 2 } })
  vi.spyOn(api, 'startHeadMaskDetection').mockResolvedValue(job)
  vi.spyOn(api, 'getHeadMaskProposals').mockResolvedValue(proposals)
  vi.spyOn(api, 'applyHeadMaskProposals').mockResolvedValue({ job_id: 41, applied: 1, images: ['1_data/A.png'], undo_available: true })
  vi.spyOn(api, 'undoHeadMaskApply').mockResolvedValue({ job_id: 41, undone: 1, images: ['1_data/A.png'] })
  vi.spyOn(api, 'cancelJob').mockResolvedValue({ task_id: 41, canceled: true })
})

describe('AutoHeadMaskPanel', () => {
  it('blocks detection while manual strokes are unsaved', async () => {
    const user = userEvent.setup()
    renderPanel({ unsavedCount: 2 })
    await user.click(await screen.findByRole('button', { name: '检测全部' }))
    expect(api.startHeadMaskDetection).not.toHaveBeenCalled()
    expect(mocks.toast).toHaveBeenCalledWith(expect.stringContaining('先保存'), 'error')
  })

  it('submits pinned defaults and loads proposals when the job completes', async () => {
    const user = userEvent.setup()
    const onStateChange = vi.fn()
    renderPanel({ onStateChange })
    await user.click(await screen.findByRole('button', { name: '检测全部' }))
    expect(api.startHeadMaskDetection).toHaveBeenCalledWith(2, 3, {
      scope: 'all', confidence: 0.413, iou_threshold: 0.7,
      padding_ratio: 0.1, feather_ratio: 0.03,
    })
    act(() => {
      mocks.onEvent?.({ type: 'head_mask_progress', job_id: 41, idx: 1, total: 2, status: 'done', detections: 2 })
      mocks.onEvent?.({ type: 'job_state_changed', job_id: 41, status: 'done' })
    })
    expect(await screen.findByText(/2 张图 · 2 个头部 · 已选 2 个/)).toBeInTheDocument()
    await waitFor(() => expect(onStateChange).toHaveBeenLastCalledWith(expect.objectContaining({
      selections: { '1_data/A.png': ['a', 'b'], '1_data/B.png': [] },
    })))
  })

  it('lets the user deselect one region, apply the rest, and undo', async () => {
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({
      job: { ...job, status: 'done' }, log_tail: '', summary: { image_count: 2 },
    })
    const user = userEvent.setup()
    const changed = vi.fn().mockResolvedValue(undefined)
    renderPanel({ onWorkspaceChanged: changed })
    const regions = await screen.findAllByRole('checkbox')
    await user.click(regions[0])
    vi.mocked(api.getHeadMaskProposals).mockResolvedValue({ ...proposals, undo_available: true })
    await user.click(screen.getByRole('button', { name: '应用所选（1）' }))
    expect(api.applyHeadMaskProposals).toHaveBeenCalledWith(2, 3, 41, {
      '1_data/A.png': ['b'], '1_data/B.png': [],
    })
    expect(changed).toHaveBeenCalled()

    await waitFor(() => expect(screen.getByRole('button', { name: '撤销本次自动遮罩' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '撤销本次自动遮罩' }))
    expect(api.undoHeadMaskApply).toHaveBeenCalledWith(2, 3, 41)
  })

  it('offers the fixed model download when it is missing', async () => {
    vi.mocked(api.getModelsCatalog).mockResolvedValue({
      ...catalog,
      head_detector: { ...catalog.head_detector!, exists: false, valid: false, size: 0 },
    })
    vi.spyOn(api, 'startModelDownload').mockResolvedValue({ key: 'head_detector', status: 'running' })
    const user = userEvent.setup()
    renderPanel()
    await user.click(await screen.findByRole('button', { name: /下载头部检测模型/ }))
    expect(api.startModelDownload).toHaveBeenCalledWith({ model_id: 'head_detector' })
  })

  it('cancels a running detection and reports terminal failures', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({
      job, log_tail: '', summary: { image_count: 2 },
    })
    renderPanel()
    await user.click(await screen.findByRole('button', { name: '取消' }))
    expect(api.cancelJob).toHaveBeenCalledWith(41)

    act(() => {
      mocks.onEvent?.({ type: 'job_state_changed', job_id: 41, status: 'failed' })
    })
    expect(mocks.toast).toHaveBeenCalledWith(expect.stringContaining('失败'), 'error')
  })

  it('reports a canceled detection without attempting to load partial proposals', async () => {
    vi.mocked(api.getPreprocessStatusTrain).mockResolvedValue({
      job, log_tail: '', summary: { image_count: 2 },
    })
    renderPanel()
    await screen.findByRole('button', { name: '取消' })
    act(() => {
      mocks.onEvent?.({ type: 'job_state_changed', job_id: 41, status: 'canceled' })
    })
    expect(mocks.toast).toHaveBeenCalledWith(expect.stringContaining('取消'), 'info')
    expect(api.getHeadMaskProposals).not.toHaveBeenCalled()
  })
})
