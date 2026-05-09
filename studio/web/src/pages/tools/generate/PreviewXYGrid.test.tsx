import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { MonitorState } from '../../../api/client'
import PreviewXYGrid from './PreviewXYGrid'
import type { XYAxisDraft } from './xy'

type Sample = NonNullable<MonitorState['samples']>[number]

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

  it('calls onCellClick with sample index when an image is clicked', async () => {
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
    await user.click(imgs[1])  // 第 2 张图（索引 1）
    expect(onCellClick).toHaveBeenCalledWith(1)
  })

  it('density toggle changes thumbnail size', async () => {
    const user = userEvent.setup()
    const samples = [makeSample(0, 0, 20, null)]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={null} />
    )
    // 默认 standard 选中
    const compactBtn = screen.getByRole('button', { name: '紧凑' })
    expect(compactBtn).toHaveClass('btn-ghost')
    await user.click(compactBtn)
    expect(compactBtn).toHaveClass('btn-primary')
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
})
