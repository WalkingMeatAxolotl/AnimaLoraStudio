import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type CropWorkspaceItem } from '../../../api/client'
import PreprocessInpaintPage from './PreprocessInpaint'

const mocks = vi.hoisted(() => ({ toast: vi.fn(), reload: vi.fn(async () => undefined) }))

vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../components/preprocess/InpaintCanvas', async () => {
  const React = await import('react')
  const Canvas = React.forwardRef(function Canvas(
    props: {
      mode: 'paint' | 'mask'
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
    const stroke = { color: '#ffffff', size: 24, hardness: 1, points: [{ x: 10, y: 10 }] }
    return (
      <button
        type="button"
        onClick={() => props.mode === 'mask'
          ? props.onMaskStrokeEnd(stroke)
          : props.onStrokeEnd(stroke)}
      >
        Draw stroke
      </button>
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
  window.localStorage.clear()
  vi.spyOn(api, 'listCropWorkspaceTrain').mockResolvedValue({ images: [image] })
  vi.spyOn(api, 'saveInpaintTrain').mockResolvedValue({ ...image, origin: image.source })
  vi.spyOn(api, 'saveMaskTrain').mockResolvedValue({ name: image.name, mtime: 2, size: 10 })
  vi.spyOn(api, 'deleteMaskTrain').mockResolvedValue({ deleted: true })
})

describe('Preprocess inpaint contracts', () => {
  it('keeps the unsaved-state filter in Filmstrip and exposes mode/tool controls', async () => {
    const { container } = renderPage()
    const filmstrip = await screen.findByRole('group', { name: '涂抹工作集图片' })
    const filter = within(filmstrip).getByRole('radiogroup', { name: '按未保存状态筛选' })
    const scrollport = container.querySelector('.overflow-y-auto')

    expect(filmstrip).toContainElement(filter)
    expect(scrollport).not.toContainElement(filter)
    expect(filter).not.toHaveClass('ui-selection-content')
    expect(screen.getByRole('radiogroup', { name: '模式' })).toBeInTheDocument()
    expect(screen.getByRole('radiogroup', { name: '工具' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存当前修改' })).toBeDisabled()
  })

  it('saves a mask-only edit without overwriting the source image', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('group', { name: '涂抹工作集图片' })

    await user.click(within(screen.getByRole('radiogroup', { name: '模式' })).getByRole('radio', { name: '训练遮罩' }))
    await user.click(screen.getByRole('button', { name: 'Draw stroke' }))
    expect(screen.getByRole('button', { name: '保存当前修改' })).toBeEnabled()
    expect(screen.getByRole('radio', { name: '待保存 1' })).toHaveAttribute('aria-checked', 'false')

    await user.click(screen.getByRole('button', { name: '保存当前修改' }))
    await waitFor(() => expect(api.saveMaskTrain).toHaveBeenCalledWith(1, 2, image.name, expect.any(Blob)))
    expect(api.saveInpaintTrain).not.toHaveBeenCalled()
    await waitFor(() => expect(screen.getByRole('button', { name: '保存当前修改' })).toBeDisabled())
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
