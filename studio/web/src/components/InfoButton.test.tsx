import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { InfoButton, computeInfoPanelPosition } from './InfoButton'
import Modal from './Modal'

const VIEWPORT = { width: 1280, height: 800 }

function rect(left: number, top: number, size = 18): DOMRect {
  return {
    left, top, right: left + size, bottom: top + size, width: size, height: size,
    x: left, y: top, toJSON: () => ({}),
  } as DOMRect
}

function setViewport(width: number, height: number) {
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: width })
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: height })
}

function openInfo(name = 'Help') {
  const trigger = screen.getByRole('button', { name })
  fireEvent.click(trigger)
  return trigger
}

afterEach(() => {
  vi.restoreAllMocks()
  setViewport(1024, 768)
})

describe('computeInfoPanelPosition', () => {
  it('anchors below the trigger left edge when it fits', () => {
    expect(computeInfoPanelPosition(rect(100, 50), 120, VIEWPORT))
      .toEqual({ top: 76, left: 100, width: 380 })
  })

  it('clamps left so the panel stays 16px inside the right edge', () => {
    // narrow sidebar trigger near the right edge: 1280 - 380 - 16 = 884
    expect(computeInfoPanelPosition(rect(1200, 50), 120, VIEWPORT).left).toBe(884)
  })

  it('clamps left to the 16px gutter on the left edge', () => {
    expect(computeInfoPanelPosition(rect(4, 50), 120, VIEWPORT).left).toBe(16)
  })

  it('shrinks width on phone-width viewports', () => {
    const pos = computeInfoPanelPosition(rect(300, 50), 120, { width: 360, height: 640 })
    expect(pos.width).toBe(328)
    expect(pos.left).toBe(16)
  })

  it('flips above the trigger when there is more room above', () => {
    expect(computeInfoPanelPosition(rect(100, 700), 200, VIEWPORT).top).toBe(700 - 8 - 200)
  })

  it('stays below when neither side fits but below has more room', () => {
    expect(computeInfoPanelPosition(rect(100, 100), 900, VIEWPORT).top).toBe(126)
  })
})

describe('InfoButton', () => {
  it('portals the panel to document.body with viewport-clamped fixed coordinates', () => {
    setViewport(1280, 800)
    const { container } = render(
      <div style={{ overflow: 'auto', width: 240 }}>
        <InfoButton ariaLabel="Help"><p>Explains the field</p></InfoButton>
      </div>,
    )
    const trigger = screen.getByRole('button', { name: 'Help' })
    vi.spyOn(trigger, 'getBoundingClientRect').mockReturnValue(rect(1200, 40))

    fireEvent.click(trigger)

    const panel = screen.getByRole('dialog', { name: 'Help' })
    expect(panel).toHaveTextContent('Explains the field')
    expect(container.contains(panel)).toBe(false)
    expect(panel.parentElement).toBe(document.body)
    expect(trigger).toHaveAttribute('aria-expanded', 'true')
    expect(trigger).toHaveAttribute('aria-controls', panel.id)
    expect(panel.style.left).toBe('884px')
    expect(panel.style.top).toBe('66px')
    expect(panel.style.width).toBe('380px')
    expect(panel.style.visibility).toBe('')
  })

  it('toggles from the trigger and stays open for clicks inside the panel', () => {
    render(<InfoButton ariaLabel="Help"><p>Body</p></InfoButton>)
    const trigger = openInfo()
    fireEvent.mouseDown(screen.getByText('Body'))
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    fireEvent.click(trigger)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(trigger).toHaveAttribute('aria-expanded', 'false')
  })

  it('closes on outside mousedown', () => {
    render(
      <>
        <button type="button">Elsewhere</button>
        <InfoButton ariaLabel="Help"><p>Body</p></InfoButton>
      </>,
    )
    openInfo()
    fireEvent.mouseDown(screen.getByRole('button', { name: 'Elsewhere' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('closes on page scroll and resize but not on scrolling its own content', () => {
    render(<InfoButton ariaLabel="Help"><p>Body</p></InfoButton>)
    openInfo()
    fireEvent.scroll(screen.getByRole('dialog'))
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    fireEvent.scroll(document)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

    openInfo()
    act(() => { window.dispatchEvent(new Event('resize')) })
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('closes on Escape without closing the surrounding Modal', () => {
    const onClose = vi.fn()
    render(
      <Modal title="Settings" onClose={onClose}>
        <InfoButton ariaLabel="Help"><p>Body</p></InfoButton>
      </Modal>,
    )
    openInfo()
    expect(screen.getByRole('dialog', { name: 'Help' })).toBeInTheDocument()

    fireEvent.keyDown(document.activeElement ?? document.body, { key: 'Escape' })
    expect(screen.queryByRole('dialog', { name: 'Help' })).not.toBeInTheDocument()
    expect(onClose).not.toHaveBeenCalled()

    // with the popover closed, Escape reaches the Modal again
    fireEvent.keyDown(document.body, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('does not close the surrounding Modal when clicking inside the portalled panel', () => {
    const onClose = vi.fn()
    render(
      <Modal title="Settings" onClose={onClose}>
        <InfoButton ariaLabel="Help"><p>Body</p></InfoButton>
      </Modal>,
    )
    openInfo()
    fireEvent.mouseDown(screen.getByText('Body'))
    expect(onClose).not.toHaveBeenCalled()
    expect(screen.getByRole('dialog', { name: 'Help' })).toBeInTheDocument()
  })
})
