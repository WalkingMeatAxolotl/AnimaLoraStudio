import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import SidebarXYAxes from './SidebarXYAxes'
import type { XYAxisDraft } from './xy'

const firstPath = 'G:/checkpoints/final.safetensors'
const secondPath = 'G:/checkpoints/epoch_80.safetensors'

function Harness({
  yEnabled = true,
  yRaw = '1',
  xRaw = `${firstPath}, ${secondPath}`,
  anchorPath = firstPath,
}: {
  yEnabled?: boolean
  yRaw?: string
  xRaw?: string
  anchorPath?: string
}) {
  const [xDraft, setXDraft] = useState<XYAxisDraft>({
    axis: 'lora_ckpt',
    raw: xRaw,
    loraIndex: null,
    checkpointAnchor: { path: anchorPath, scale: 1, project_id: 1, version_id: 2 },
  })
  const [yDraft, setYDraft] = useState<XYAxisDraft>({
    axis: 'lora_scale',
    raw: yRaw,
    loraIndex: null,
  })
  const [activeAxis, setActiveAxis] = useState<'X' | 'Y'>('X')
  const [manualReorders, setManualReorders] = useState(0)
  return (
    <>
      <output data-testid="x-raw">{xDraft.raw}</output>
      <output data-testid="x-anchor">{xDraft.checkpointAnchor?.path ?? ''}</output>
      <output data-testid="y-raw">{yDraft.raw}</output>
      <output data-testid="manual-reorders">{manualReorders}</output>
      <SidebarXYAxes
        xDraft={xDraft}
        yDraft={yDraft}
        yEnabled={yEnabled}
        activeAxis={activeAxis}
        editorOpen={false}
        fp8BaseModel={false}
        onSelectAxis={setActiveAxis}
        onEdit={() => {}}
        onAxisChange={(axis, draft) => axis === 'X' ? setXDraft(draft) : setYDraft(draft)}
        onManualReorder={() => setManualReorders((count) => count + 1)}
        onSwap={vi.fn()}
      />
    </>
  )
}

describe('SidebarXYAxes', () => {
  it('uses X/Y top tabs, shows one compact image count, and has no Y close control', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    expect(screen.getByTestId('xy-image-count')).toHaveTextContent('2张')
    const xTab = screen.getByRole('tab', { name: 'X 轴 · LoRA' })
    expect(xTab).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Y 轴 · 权重' })).toBeInTheDocument()
    expect(screen.queryByText(/固定 LoRA/)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /添加 Y 轴/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /移除 Y 轴/ })).not.toBeInTheDocument()

    expect(screen.getByTestId('xy-axis-sticky-header')).toHaveClass('sticky')
    expect(screen.getAllByText('X 轴 · LoRA')).toHaveLength(1)

    xTab.focus()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('tab', { name: 'Y 轴 · 权重' })).toHaveFocus()
    expect(screen.getByTestId('xy-axis-selected-value')).toHaveTextContent('1')
    expect(screen.getByTestId('xy-axis-selected-values')).toBeInTheDocument()
    expect(screen.getAllByText('Y 轴 · 权重')).toHaveLength(1)
  })

  it('uses each checkpoint card as the drag target and preserves delete', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    const firstCard = screen.getByRole('button', { name: '拖动调整顺序 final' })
    const secondCard = screen.getByRole('button', { name: '拖动调整顺序 epoch_80' })
    const deleteSecond = screen.getByRole('button', { name: '删除 epoch_80' })
    expect(firstCard).toHaveClass('rounded-md', 'cursor-grab', 'bg-overlay', 'p-2.5')
    expect(deleteSecond).toHaveClass('opacity-0', 'pointer-events-none', 'group-hover:opacity-100', 'group-hover:pointer-events-auto', 'group-focus-within:opacity-100')
    expect(secondCard).not.toContainElement(deleteSecond)
    expect(deleteSecond.parentElement).toBe(secondCard.parentElement)
    expect(screen.queryByRole('button', { name: /上移|下移/ })).not.toBeInTheDocument()
    expect(screen.queryByText('⠿')).not.toBeInTheDocument()
    expect(screen.queryByTestId('xy-axis-drop-indicator')).not.toBeInTheDocument()

    await user.click(deleteSecond)
    expect(screen.getByTestId('x-raw')).toHaveTextContent(firstPath)
    expect(screen.getByTestId('manual-reorders')).toHaveTextContent('0')
    expect(screen.getAllByTestId('xy-axis-selected-value')).toHaveLength(1)
  })

  it('lists numeric values as draggable cards with removal', async () => {
    const user = userEvent.setup()
    render(<Harness yRaw="0.5, 0.75, 1" />)

    await user.click(screen.getByRole('tab', { name: 'Y 轴 · 权重' }))
    expect(screen.getAllByTestId('xy-axis-selected-value')).toHaveLength(3)
    expect(screen.getAllByRole('button', { name: /拖动调整顺序/ })).toHaveLength(3)
    expect(screen.queryByRole('button', { name: /上移|下移/ })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '删除 0.75' }))
    expect(screen.getByTestId('y-raw')).toHaveTextContent('0.5, 1')
    expect(screen.getByTestId('manual-reorders')).toHaveTextContent('0')
    expect(screen.getAllByTestId('xy-axis-selected-value')).toHaveLength(2)
  })

  it('does not case-fold POSIX checkpoint anchors when deleting', async () => {
    const user = userEvent.setup()
    render(
      <Harness
        xRaw="/models/Foo.safetensors, /models/foo.safetensors"
        anchorPath="/models/Foo.safetensors"
      />,
    )

    await user.click(screen.getByRole('button', { name: '删除 Foo' }))
    expect(screen.getByTestId('x-raw')).toHaveTextContent('/models/foo.safetensors')
    expect(screen.getByTestId('x-anchor')).toHaveTextContent('/models/foo.safetensors')
  })

  it('treats a virtual one-value Y axis as dimensionless for the count', () => {
    render(<Harness yEnabled={false} />)
    expect(screen.getByTestId('xy-image-count')).toHaveTextContent('2张')
  })
})
