import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import FreeCropEditor, { applyResize, type CropRect } from './FreeCropEditor'

const baseImage = {
  id: 'X.png',
  name: 'X.png',
  w: 1024,
  h: 1536,
  thumbUrl: '/fake-thumb.png',
}

const noop = () => {}

describe('FreeCropEditor', () => {
  it('renders without crops (canvas only)', () => {
    render(
      <FreeCropEditor
        image={baseImage}
        crops={[]}
        selectedId={null}
        arLock={null}
        onSelect={noop}
        onChange={noop}
        onCreate={noop}
        onDelete={noop}
        onDuplicate={noop}
      />,
    )
    // Image filename + dim shown in tools row
    expect(screen.getByText(/X\.png/)).toBeInTheDocument()
    expect(screen.getByText(/1024×1536/)).toBeInTheDocument()
  })

  it('renders rect with label + output pixel size', () => {
    const rects: CropRect[] = [
      { id: 'r1', x: 0.1, y: 0.1, w: 0.5, h: 0.5, label: 'head' },
    ]
    render(
      <FreeCropEditor
        image={baseImage}
        crops={rects}
        selectedId="r1"
        arLock={null}
        onSelect={noop}
        onChange={noop}
        onCreate={noop}
        onDelete={noop}
        onDuplicate={noop}
      />,
    )
    expect(screen.getByText('head')).toBeInTheDocument()
    // 1024*0.5 = 512, 1536*0.5 = 768
    expect(screen.getByText(/512×768/)).toBeInTheDocument()
  })

  it('disables action buttons when nothing is selected', () => {
    const onDelete = vi.fn()
    const onDuplicate = vi.fn()
    render(
      <FreeCropEditor
        image={baseImage}
        crops={[]}
        selectedId={null}
        arLock={null}
        onSelect={noop}
        onChange={noop}
        onCreate={noop}
        onDelete={onDelete}
        onDuplicate={onDuplicate}
      />,
    )
    const dup = screen.getByText('复制选中').closest('button')
    const del = screen.getByText('删除选中').closest('button')
    expect(dup).toBeDisabled()
    expect(del).toBeDisabled()
  })

  /** AR is preserved across any out-of-bounds drag.
   *
   *  Regression for the 0.9.x bug: with arLock=1:1 on a 2:3 image, dragging
   *  the SE handle far past the canvas edge used to clamp `w` and `h`
   *  independently to 1, silently turning 1:1 into 2:3 (= the source image's
   *  AR). Fix scales the locked rect uniformly toward the anchored corner.
   */
  describe('AR lock preserves ratio across out-of-bounds drag', () => {
    const portraitImg = { w: 1024, h: 1536 } // 2:3
    const lock1to1 = { w: 1, h: 1 }
    const tolerance = 0.005

    const pixelAR = (r: { w: number; h: number }, im: { w: number; h: number }) =>
      (r.w * im.w) / (r.h * im.h)

    const startRect: CropRect = {
      id: 'r', label: 'x', x: 0.1, y: 0.1, w: 0.3, h: 0.3 * (1024 / 1536),
    }

    it('SE handle drag past bottom-right keeps 1:1 (does NOT collapse to 2:3)', () => {
      const out = applyResize(
        startRect, 'se',
        5, 5, // way out of bounds
        lock1to1, portraitImg.w, portraitImg.h,
      )
      expect(pixelAR(out, portraitImg)).toBeCloseTo(1, tolerance)
      // and it fits in the canvas
      expect(out.x + out.w).toBeLessThanOrEqual(1 + 1e-9)
      expect(out.y + out.h).toBeLessThanOrEqual(1 + 1e-9)
    })

    it('NW handle drag past top-left keeps 1:1', () => {
      const out = applyResize(
        { ...startRect, x: 0.7, y: 0.7, w: 0.2, h: 0.2 * (1024 / 1536) },
        'nw',
        -5, -5,
        lock1to1, portraitImg.w, portraitImg.h,
      )
      expect(pixelAR(out, portraitImg)).toBeCloseTo(1, tolerance)
      expect(out.x).toBeGreaterThanOrEqual(-1e-9)
      expect(out.y).toBeGreaterThanOrEqual(-1e-9)
    })

    it('E edge handle drag out keeps locked AR (was the easiest repro path)', () => {
      const out = applyResize(
        startRect, 'e',
        5, 0,
        lock1to1, portraitImg.w, portraitImg.h,
      )
      expect(pixelAR(out, portraitImg)).toBeCloseTo(1, tolerance)
    })

    it('custom 16:9 lock on portrait image fits without breaking AR', () => {
      const out = applyResize(
        startRect, 'se',
        5, 5,
        { w: 16, h: 9 }, portraitImg.w, portraitImg.h,
      )
      expect(pixelAR(out, portraitImg)).toBeCloseTo(16 / 9, 0.01)
    })

    it('free mode (no arLock) still clamps to canvas bounds', () => {
      const out = applyResize(
        startRect, 'se',
        5, 5,
        null, portraitImg.w, portraitImg.h,
      )
      expect(out.x + out.w).toBeLessThanOrEqual(1 + 1e-9)
      expect(out.y + out.h).toBeLessThanOrEqual(1 + 1e-9)
    })
  })

  it('renders 8 handles for the selected rect', () => {
    const rects: CropRect[] = [
      { id: 'r1', x: 0.1, y: 0.1, w: 0.5, h: 0.5, label: 'r' },
    ]
    const { container } = render(
      <FreeCropEditor
        image={baseImage}
        crops={rects}
        selectedId="r1"
        arLock={null}
        onSelect={noop}
        onChange={noop}
        onCreate={noop}
        onDelete={noop}
        onDuplicate={noop}
      />,
    )
    const handles = container.querySelectorAll('.handle')
    expect(handles.length).toBe(8)
  })
})
