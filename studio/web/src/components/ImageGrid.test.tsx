import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import ImageGrid, { applySelection } from './ImageGrid'

const items = [
  { name: 'a.png', thumbUrl: '/a' },
  { name: 'b.png', thumbUrl: '/b' },
  { name: 'c.png', thumbUrl: '/c' },
]

describe('applySelection (PP3 — checkbox semantics)', () => {
  const e = (mods: Partial<{ shift: boolean }> = {}) =>
    ({
      shiftKey: !!mods.shift,
      ctrlKey: false,
      metaKey: false,
    } as unknown as React.MouseEvent)

  it('plain click on unselected → adds to selection (multi-select default)', () => {
    const r = applySelection(new Set(['a.png']), 'b.png', e(), ['a.png', 'b.png'], 'a.png')
    expect(new Set(r.next)).toEqual(new Set(['a.png', 'b.png']))
    expect(r.anchor).toBe('b.png')
  })

  it('plain click on already-selected → removes (toggle off)', () => {
    const r = applySelection(
      new Set(['a.png', 'b.png']),
      'a.png',
      e(),
      ['a.png', 'b.png'],
      'b.png'
    )
    expect([...r.next]).toEqual(['b.png'])
    expect(r.anchor).toBe('a.png')
  })

  it('shift click → range from anchor adds inclusive', () => {
    const r = applySelection(
      new Set(['a.png']),
      'c.png',
      e({ shift: true }),
      ['a.png', 'b.png', 'c.png'],
      'a.png'
    )
    expect(new Set(r.next)).toEqual(new Set(['a.png', 'b.png', 'c.png']))
  })

  it('shift click without anchor → falls back to toggle', () => {
    const r = applySelection(new Set(), 'b.png', e({ shift: true }), ['a.png', 'b.png'], null)
    expect([...r.next]).toEqual(['b.png'])
  })
})

describe('ImageGrid (PP3)', () => {
  it('renders cells with aria-selected matching set', () => {
    render(
      <ImageGrid
        items={items}
        selected={new Set(['b.png'])}
        onSelect={() => {}}
      />
    )
    const cells = screen.getAllByRole('gridcell')
    expect(cells).toHaveLength(3)
    expect(cells[0]).toHaveAttribute('aria-selected', 'false')
    expect(cells[1]).toHaveAttribute('aria-selected', 'true')
  })

  it('clicking a cell calls onSelect with name', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()
    render(
      <ImageGrid items={items} selected={new Set()} onSelect={onSelect} />
    )
    await user.click(screen.getAllByRole('gridcell')[0])
    expect(onSelect).toHaveBeenCalledWith(
      'a.png',
      expect.objectContaining({ shiftKey: false })
    )
  })

  it('activate mode uses plain cell click for activation', async () => {
    const onSelect = vi.fn()
    const onActivate = vi.fn()
    const user = userEvent.setup()
    render(
      <ImageGrid
        items={items}
        selected={new Set()}
        onSelect={onSelect}
        onActivate={onActivate}
        clickMode="activate"
      />
    )
    await user.click(screen.getAllByRole('gridcell')[0])
    expect(onActivate).toHaveBeenCalledWith('a.png')
    expect(onSelect).not.toHaveBeenCalled()
  })

  it('activate mode keeps checkbox selection separate', async () => {
    const onSelect = vi.fn()
    const onActivate = vi.fn()
    const user = userEvent.setup()
    render(
      <ImageGrid
        items={items}
        selected={new Set()}
        onSelect={onSelect}
        onActivate={onActivate}
        clickMode="activate"
      />
    )
    await user.click(screen.getByRole('button', { name: '选择 a.png' }))
    expect(onSelect).toHaveBeenCalledWith(
      'a.png',
      expect.objectContaining({ shiftKey: false })
    )
    expect(onActivate).not.toHaveBeenCalled()
  })

  it('shows empty hint', () => {
    render(
      <ImageGrid items={[]} selected={new Set()} onSelect={() => {}} emptyHint="空空" />
    )
    expect(screen.getByText('空空')).toBeInTheDocument()
  })
})
