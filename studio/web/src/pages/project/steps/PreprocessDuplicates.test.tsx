import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type DuplicateApplyResult, type DuplicateScanResult } from '../../../api/client'
import PreprocessDuplicatesPage from './PreprocessDuplicates'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  confirm: vi.fn(async () => true),
  reload: vi.fn(async () => undefined),
}))

vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../components/Dialog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Dialog')>()
  return { ...actual, useDialog: () => ({ confirm: mocks.confirm }) }
})
vi.mock('../../../lib/useEventStream', () => ({ useEventStream: () => undefined }))

const result: DuplicateScanResult = {
  target: 'preprocess',
  match_scope: 'both',
  total_images: 2,
  readable_images: 2,
  group_count: 1,
  candidate_count: 1,
  crop_relation_count: 0,
  elapsed_seconds: 0.2,
  stats: {
    total_pairs: 1,
    aspect_skipped_pairs: 0,
    prefiltered_pairs: 1,
    compared_pairs: 1,
  },
  groups: [{
    group_id: 1,
    keep: '1_data/keep.png',
    best: null,
    items: [
      { name: '1_data/keep.png', keep: true, width: 512, height: 512, filesize_kb: 20, metrics: null },
      { name: '1_data/remove.png', keep: false, width: 512, height: 512, filesize_kb: 18, metrics: null },
    ],
  }],
}

function renderPage() {
  return render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }} initialEntries={['/duplicates']}>
      <Routes>
        <Route element={<Outlet context={{ project: { id: 1, download_image_count: 2 }, activeVersion: { id: 2 }, reload: mocks.reload }} />}>
          <Route path="/duplicates" element={<PreprocessDuplicatesPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  mocks.confirm.mockResolvedValue(true)
  vi.spyOn(api, 'scanDuplicatesTrain').mockResolvedValue(result)
  vi.spyOn(api, 'applyDuplicateActionTrain').mockResolvedValue({
    removed: ['1_data/remove.png'], missing: [], skipped: [],
  })
})

describe('Preprocess duplicate review contracts', () => {
  it('uses the primary scan mode control, locks sensitivity, and preserves scan options', async () => {
    const user = userEvent.setup()
    renderPage()

    const scope = screen.getByRole('radiogroup', { name: '匹配范围' })
    await user.click(within(scope).getByRole('radio', { name: '严格重复' }))
    for (const sensitivity of within(screen.getByRole('radiogroup', { name: '灵敏度' })).getAllByRole('radio')) {
      expect(sensitivity).toBeDisabled()
    }
    await user.click(within(scope).getByRole('radio', { name: '重复 + 同分镜差分' }))
    await user.click(within(screen.getByRole('radiogroup', { name: '灵敏度' })).getByRole('radio', { name: '严格' }))
    await user.click(screen.getByRole('button', { name: '扫描重复' }))

    expect(api.scanDuplicatesTrain).toHaveBeenCalledWith(1, 2, {
      match_scope: 'both', sensitivity: 'strict',
    })
    expect(await screen.findByRole('button', { name: '保留 1_data/keep.png' })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('button', { name: '去除 1_data/remove.png' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('keeps the toggle name aligned with its visible Keep/Remove state', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(screen.getByRole('button', { name: '扫描重复' }))

    const keep = await screen.findByRole('button', { name: '保留 1_data/keep.png' })
    expect(keep).toHaveTextContent('保留')
    await user.click(keep)
    const remove = screen.getByRole('button', { name: '去除 1_data/keep.png' })
    expect(remove).toHaveTextContent('去除')
    expect(remove).toHaveAttribute('aria-pressed', 'true')
  })

  it('does not present Apply as another scan and preserves the apply payload', async () => {
    let resolveApply!: (value: DuplicateApplyResult) => void
    vi.mocked(api.applyDuplicateActionTrain).mockReturnValue(new Promise((resolve) => { resolveApply = resolve }))
    const user = userEvent.setup()
    renderPage()
    await user.click(screen.getByRole('button', { name: '扫描重复' }))
    await screen.findByRole('button', { name: '去除 1_data/remove.png' })

    await user.click(screen.getByRole('button', { name: '确认去除 1' }))
    await waitFor(() => expect(api.applyDuplicateActionTrain).toHaveBeenCalledWith(1, 2, { names: ['1_data/remove.png'] }))

    const scan = screen.getByRole('button', { name: '扫描重复' })
    expect(scan).toBeDisabled()
    expect(scan).not.toHaveAttribute('aria-busy')
    expect(screen.queryByRole('button', { name: '扫描中...' })).not.toBeInTheDocument()
    expect(api.scanDuplicatesTrain).toHaveBeenCalledTimes(1)

    await act(async () => resolveApply({ removed: ['1_data/remove.png'], missing: [], skipped: [] }))
  })
})
