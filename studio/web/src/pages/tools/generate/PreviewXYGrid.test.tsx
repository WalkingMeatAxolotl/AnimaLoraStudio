import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import PreviewXYGrid, { type XYSample } from './PreviewXYGrid'
import type { XYAxisDraft } from './xy'

type Sample = XYSample

const xDraft: XYAxisDraft = { axis: 'steps', raw: '20, 25, 30', loraIndex: null }
const yDraft: XYAxisDraft = { axis: 'cfg_scale', raw: '3.0, 5.0', loraIndex: null }

function makeSample(xi: number, yi: number, xv: number, yv: number | null): Sample {
  return {
    path: `/tmp/anima_gen_99/xy_x${String(xi).padStart(2, '0')}_y${String(yi).padStart(2, '0')}_s42.png`,
    step: yi * 3 + xi + 1,
    xy: { xi, yi, xv, yv },
  }
}

describe('PreviewXYGrid', () => {
  it('shows total count for 2D matrix', () => {
    render(
      <PreviewXYGrid
        samples={[]}
        taskId={99}
        xDraft={xDraft}
        yDraft={yDraft}
      />
    )
    // 3 × 2 = 6 张
    expect(screen.getByText(/3 × 2 = 6 张/)).toBeInTheDocument()
  })

  it('shows partial count when generation in progress', () => {
    const samples = [makeSample(0, 0, 20, 3.0), makeSample(1, 0, 25, 3.0)]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={yDraft} />
    )
    expect(screen.getByText(/3 × 2 = 6 张/)).toBeInTheDocument()
    expect(screen.getByText(/已出 2/)).toBeInTheDocument()
  })

  it('renders 1D layout (y=null) with single header row', () => {
    render(
      <PreviewXYGrid samples={[]} taskId={99} xDraft={xDraft} yDraft={null} />
    )
    expect(screen.getByText(/3 张/)).toBeInTheDocument()
  })

  it('renders cell images at the right (yi, xi) positions', () => {
    const samples: Sample[] = [
      makeSample(0, 0, 20, 3.0),
      makeSample(2, 1, 30, 5.0),
    ]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={yDraft} />
    )
    // 已出 cell 显示 img；未出 cell 显示 …
    const imgs = screen.getAllByRole('img')
    expect(imgs.length).toBe(2)
    // 占位灰格至少 4 个（6 总 - 2 已出）
    const placeholders = screen.getAllByText('…')
    expect(placeholders.length).toBe(4)
  })

  it('calls onCellClick only with Ctrl+click (普通点击让位给 pan 拖动)', async () => {
    const user = userEvent.setup()
    const onCellClick = vi.fn()
    const samples = [makeSample(0, 0, 20, null), makeSample(1, 0, 25, null)]
    render(
      <PreviewXYGrid
        samples={samples}
        taskId={99}
        xDraft={xDraft}
        yDraft={null}
        onCellClick={onCellClick}
      />
    )
    const imgs = screen.getAllByRole('img')
    // 普通点击 → 不触发（让位 pan）
    await user.click(imgs[1])
    expect(onCellClick).not.toHaveBeenCalled()
    // Ctrl+点击 → 触发
    await user.keyboard('{Control>}')
    await user.click(imgs[1])
    await user.keyboard('{/Control}')
    expect(onCellClick).toHaveBeenCalledWith(1)
  })

  it('shows zoom percentage button (replaces old density toggle)', () => {
    const samples = [makeSample(0, 0, 20, null)]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={null} />
    )
    // 默认 100%
    const zoomBtn = screen.getByRole('button', { name: '100%' })
    expect(zoomBtn).toBeInTheDocument()
  })

  it('highlights selected cells via selectedIndices', () => {
    const samples = [makeSample(0, 0, 20, null), makeSample(1, 0, 25, null)]
    render(
      <PreviewXYGrid
        samples={samples}
        taskId={99}
        xDraft={xDraft}
        yDraft={null}
        selectedIndices={[0]}
      />
    )
    const buttons = screen.getAllByRole('img').map((img) => img.closest('button'))
    expect(buttons[0]?.className).toContain('border-accent')
    expect(buttons[1]?.className).not.toContain('border-accent')
  })

  it('navigates fullscreen cells with arrow keys', async () => {
    const user = userEvent.setup()
    const samples: Sample[] = [
      makeSample(0, 0, 20, 3.0),
      makeSample(1, 0, 25, 3.0),
      makeSample(0, 1, 20, 5.0),
      makeSample(1, 1, 25, 5.0),
    ]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={yDraft} />
    )

    await user.dblClick(screen.getAllByRole('img')[0])
    expect(screen.getByText(/步数=20 .* CFG Scale=3/)).toBeInTheDocument()

    await user.keyboard('{ArrowRight}')
    expect(screen.getByText(/步数=25 .* CFG Scale=3/)).toBeInTheDocument()

    await user.keyboard('{ArrowDown}')
    expect(screen.getByText(/步数=25 .* CFG Scale=5/)).toBeInTheDocument()
  })

  it('uses sample.imageUrl when provided (disk 回看路径)', () => {
    const sample: Sample = {
      ...makeSample(0, 0, 20, null),
      imageUrl: '/api/generate/disk/image/2026-06-08/xy/xy%20plot%201/cell%20x0%20y0.png',
    }
    render(
      <PreviewXYGrid samples={[sample]} taskId={-1} xDraft={xDraft} yDraft={null} />
    )
    const img = screen.getAllByRole('img')[0] as HTMLImageElement
    expect(img.src).toContain('/api/generate/disk/image/2026-06-08/xy/')
    expect(img.src).not.toContain('/sample/')  // 不走 generateSampleUrl 兜底
  })

  it('compositeUrl set → 导出 PNG 走 anchor download, 不调 composeXYMatrix', async () => {
    // composeXYMatrix 需要 canvas + fetch，jsdom 都没 → 若 fallback 走它必然抛错。
    // 这里点 export 按钮，断言能拿到 exportDownloaded 文案 = 走了 compositeUrl 直下载分支。
    const user = userEvent.setup()
    const samples = [makeSample(0, 0, 20, null)]
    // 拦截 anchor click（避免真的下载窗口）
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    render(
      <PreviewXYGrid
        samples={samples} taskId={-1} xDraft={xDraft} yDraft={null}
        compositeUrl="/api/generate/disk/image/2026-06-08/xy/xy%20plot%201/xy%20plot.png"
      />
    )
    const exportBtn = screen.getByRole('button', { name: /导出 PNG|Export/ })
    await user.click(exportBtn)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    clickSpy.mockRestore()
  })
})
