import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { MonitorState } from '../../../api/client'
import PreviewCompare from './PreviewCompare'
import type { XYAxisDraft } from './xy'

type Sample = NonNullable<MonitorState['samples']>[number]

const xDraft: XYAxisDraft = { axis: 'steps', raw: '20, 25, 30', loraIndex: null }
const yDraft: XYAxisDraft = { axis: 'cfg_scale', raw: '3.0, 5.0', loraIndex: null }

function makeSample(xi: number, yi: number, xv: number, yv: number | null): Sample {
  return {
    path: `/tmp/anima_gen_99/xy_x${String(xi).padStart(2, '0')}_y${String(yi).padStart(2, '0')}_s42.png`,
    xy: { xi, yi, xv, yv },
  }
}

describe('PreviewCompare', () => {
  function renderCompare(overrides: Partial<{
    samples: Sample[]
    selectedIndices: [number, number]
    yDraft: XYAxisDraft | null
  }> = {}) {
    const onBack = vi.fn()
    const samples = overrides.samples ?? [
      makeSample(0, 0, 20, 3.0),
      makeSample(2, 1, 30, 5.0),
    ]
    const utils = render(
      <PreviewCompare
        samples={samples}
        taskId={99}
        selectedIndices={overrides.selectedIndices ?? [0, 1]}
        xDraft={xDraft}
        yDraft={overrides.yDraft === undefined ? yDraft : overrides.yDraft}
        onBack={onBack}
      />
    )
    return { ...utils, onBack }
  }

  it('renders both selected images side by side', () => {
    renderCompare()
    const imgs = screen.getAllByRole('img')
    expect(imgs.length).toBe(2)
    expect(imgs[0].getAttribute('src')).toContain('xy_x00_y00')
    expect(imgs[1].getAttribute('src')).toContain('xy_x02_y01')
  })

  it('shows axis labels for both samples (2D)', () => {
    renderCompare()
    expect(screen.getByText(/步数=20 · CFG Scale=3/)).toBeInTheDocument()
    expect(screen.getByText(/步数=30 · CFG Scale=5/)).toBeInTheDocument()
  })

  it('shows only X label when yDraft is null (1D)', () => {
    renderCompare({
      samples: [makeSample(0, 0, 20, null), makeSample(1, 0, 25, null)],
      yDraft: null,
    })
    expect(screen.getByText(/步数=20$/)).toBeInTheDocument()
    expect(screen.getByText(/步数=25$/)).toBeInTheDocument()
    // 不应有 CFG Scale 文字
    expect(screen.queryByText(/CFG Scale/)).not.toBeInTheDocument()
  })

  it('renders side badges A and B', () => {
    renderCompare()
    expect(screen.getByText('A')).toBeInTheDocument()
    expect(screen.getByText('B')).toBeInTheDocument()
  })

  it('triggers onBack when 返回网格 clicked', async () => {
    const user = userEvent.setup()
    const { onBack } = renderCompare()
    await user.click(screen.getByText(/返回网格/))
    expect(onBack).toHaveBeenCalled()
  })

  it('shows fallback message when selectedIndices are out of range', () => {
    renderCompare({ selectedIndices: [5, 7] })
    expect(screen.getByText(/所选样本已不可用/)).toBeInTheDocument()
    // 仍渲染返回按钮
    expect(screen.getByText(/返回网格/)).toBeInTheDocument()
  })

  it('triggers onBack from fallback view too', async () => {
    const user = userEvent.setup()
    const { onBack } = renderCompare({ selectedIndices: [5, 7] })
    await user.click(screen.getByText(/返回网格/))
    expect(onBack).toHaveBeenCalled()
  })
})
