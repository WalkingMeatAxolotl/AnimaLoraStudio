import type { ReactNode } from 'react'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type CropWorkspaceItem, type ProjectDetail, type Version } from '../../../api/client'
import PreprocessOverviewPage from './PreprocessOverview'

const mocks = vi.hoisted(() => ({ confirm: vi.fn(), toast: vi.fn(), reload: vi.fn(), onEvent: undefined as undefined | ((event: unknown) => void) }))
vi.mock('../../../components/Dialog', () => ({ useDialog: () => ({ confirm: mocks.confirm }) }))
vi.mock('../../../components/Toast', () => ({ useToast: () => ({ toast: mocks.toast }) }))
vi.mock('../../../lib/useEventStream', () => ({ useEventStream: (callback: (event: unknown) => void) => { mocks.onEvent = callback } }))
vi.mock('../../../components/ImagePreviewModal', () => ({ default: ({ src, compareSrc }: { src: string; compareSrc?: string }) => <div role="dialog" aria-label="Preview" data-src={src} data-compare={compareSrc} /> }))
// JSDOM has no layout measurements. Keep the real ImageGrid wrapper/selection
// and inspect the list inset separately from the mocked virtual scrollport.
vi.mock('react-virtuoso', () => ({
  VirtuosoGrid: ({ totalCount, itemContent, listClassName }: { totalCount: number; itemContent: (index: number) => ReactNode; listClassName: string }) => (
    <div data-testid="virtual-scrollport"><div data-testid="virtual-list" className={listClassName}>
      {Array.from({ length: totalCount }, (_, i) => <div key={i}>{itemContent(i)}</div>)}
    </div></div>
  ),
}))

const image = (name: string, processed: boolean): CropWorkspaceItem => ({ name, processed, source: name.slice(name.lastIndexOf('/') + 1), w: 512, h: 512, size: 100, mtime: 1, mask_mtime: null })
const images = [image('1_data/processed.png', true), image('1_data/original.png', false)]

function renderPage(activeVersion: Version | null = { id: 2 } as Version) {
  return render(<MemoryRouter initialEntries={['/overview']}><Routes>
    <Route element={<Outlet context={{ project: { id: 1 } as ProjectDetail, activeVersion, reload: mocks.reload }} />}>
      <Route path="/overview" element={<PreprocessOverviewPage />} />
    </Route>
  </Routes></MemoryRouter>)
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  mocks.confirm.mockResolvedValue(true)
  mocks.reload.mockResolvedValue(undefined)
  vi.spyOn(api, 'listCropWorkspaceTrain').mockResolvedValue({ images })
  vi.spyOn(api, 'listPreprocessDuplicatesRemovedTrain').mockResolvedValue({ images: [{ name: 'removed.png', source: 'removed.png', w: 512, h: 512, mtime: 1, size: 100 }] })
  vi.spyOn(api, 'restorePreprocessFilesTrain').mockResolvedValue({ restored: ['1_data/processed.png'], no_origin: [], missing: [] })
  vi.spyOn(api, 'resetPreprocessFilesTrain').mockResolvedValue({ ok: true } as Awaited<ReturnType<typeof api.resetPreprocessFilesTrain>>)
})

describe('PreprocessOverview adoption', () => {
  it('keeps ImageGrid directly bounded and puts padding inside its virtual list', async () => {
    renderPage()
    const grid = await screen.findByRole('grid', { name: '处理后数据集' })
    const panel = screen.getByRole('tabpanel')
    expect(grid.parentElement).toBe(panel)
    expect(grid).toHaveClass('flex-1', 'min-h-0', 'h-full')
    expect(panel).toHaveClass('flex', 'flex-col', 'flex-1', 'min-h-0', 'overflow-hidden')
    expect(panel).not.toHaveClass('overflow-y-auto', 'p-3', 'p-2')
    expect(screen.getByTestId('virtual-list')).toHaveClass('p-2')
    expect(screen.getByTestId('virtual-scrollport')).not.toHaveClass('p-2')
    expect(api.listCropWorkspaceTrain).toHaveBeenCalledWith(1, 2)
  })

  it('associates tabs with the stable panel and clears selection when switching by keyboard', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('grid')
    await user.click(screen.getByRole('button', { name: '全选' }))
    expect(screen.getByRole('button', { name: '撤销选中 (1)' })).toBeEnabled()
    const allTab = screen.getByRole('tab', { name: '处理后数据集 (2)' })
    const removedTab = screen.getByRole('tab', { name: '已删除 (1)' })
    allTab.focus()
    await user.keyboard('{ArrowRight}')
    expect(removedTab).toHaveFocus()
    expect(removedTab).toHaveAttribute('aria-selected', 'true')
    expect(removedTab).toHaveAttribute('tabindex', '0')
    const panel = screen.getByRole('tabpanel')
    expect(removedTab).toHaveAttribute('aria-controls', panel.id)
    expect(allTab).toHaveAttribute('aria-controls', panel.id)
    expect(panel).toHaveAttribute('aria-labelledby', removedTab.id)
    expect(screen.getByRole('button', { name: '撤销选中 (0)' })).toBeDisabled()
    await user.keyboard('{Home}')
    expect(allTab).toHaveFocus()
    await user.keyboard('{End}')
    expect(removedTab).toHaveFocus()
  })

  it('preserves processed-only selection, undo confirmation and API payloads', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('grid')
    await user.click(screen.getByRole('button', { name: '全选' }))
    const cells = screen.getAllByRole('gridcell')
    expect(cells[0]).toHaveAttribute('aria-selected', 'true')
    expect(cells[1]).toHaveAttribute('aria-selected', 'false')
    await user.click(screen.getByRole('button', { name: '撤销选中 (1)' }))
    await waitFor(() => expect(api.restorePreprocessFilesTrain).toHaveBeenCalledWith(1, 2, ['1_data/processed.png']))
    expect(mocks.confirm).toHaveBeenCalledWith(expect.any(String), { tone: 'danger', okText: '撤销' })
    await waitFor(() => expect(mocks.reload).toHaveBeenCalled())
  })

  it('preserves split preview and undo-all cancellation', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('grid')
    await user.click(screen.getAllByRole('gridcell')[0])
    expect(screen.getByRole('dialog', { name: 'Preview' })).toHaveAttribute('data-compare', expect.stringContaining('processed.png'))
    mocks.confirm.mockResolvedValueOnce(false)
    await user.click(screen.getByRole('button', { name: '撤销全部' }))
    expect(api.resetPreprocessFilesTrain).not.toHaveBeenCalled()
  })

  it('separates initial loading, load failure and empty state; retry recovers', async () => {
    let reject!: (error: Error) => void
    vi.mocked(api.listCropWorkspaceTrain).mockReturnValueOnce(new Promise((_, fail) => { reject = fail }))
    vi.mocked(api.listPreprocessDuplicatesRemovedTrain).mockResolvedValue({ images: [] })
    const user = userEvent.setup()
    const { container } = renderPage()
    expect(screen.getByRole('status')).toHaveTextContent('加载中')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-busy', 'true')
    expect(container.querySelector('.empty-state')).toBeNull()
    await act(async () => reject(new Error('offline')))
    expect(screen.getByRole('alert')).toHaveTextContent('无法加载数据集')
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(container.querySelector('.empty-state')).toBeNull()
    vi.mocked(api.listCropWorkspaceTrain).mockResolvedValue({ images: [] })
    await user.click(screen.getByRole('button', { name: '重试' }))
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
    const empty = container.querySelector('.empty-state')
    expect(empty).not.toHaveClass('card')
    expect(empty).toHaveTextContent('项目里还没有图片')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-busy', 'false')
    await user.click(screen.getByRole('tab', { name: '已删除 (0)' }))
    expect(within(screen.getByRole('tabpanel')).getByText('没有被去重审核标记移除的图片。')).toBeInTheDocument()
  })

  it('retains loaded images when an SSE refresh fails instead of replacing them with empty state', async () => {
    renderPage()
    const grid = await screen.findByRole('grid')
    vi.mocked(api.listCropWorkspaceTrain).mockRejectedValueOnce(new Error('offline'))
    act(() => mocks.onEvent?.({ type: 'project_state_changed', project_id: 1 }))
    await screen.findByRole('alert')
    expect(grid).toBeInTheDocument()
    expect(screen.getAllByRole('gridcell')).toHaveLength(2)
  })

  it('keeps the version guard without fetching a zero version', () => {
    renderPage(null)
    expect(screen.queryByRole('tabpanel')).not.toBeInTheDocument()
    expect(api.listCropWorkspaceTrain).not.toHaveBeenCalled()
  })
})
