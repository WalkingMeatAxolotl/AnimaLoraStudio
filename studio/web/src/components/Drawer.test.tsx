import { useState } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Drawer from './Drawer'
import Modal from './Modal'

function endAnimation(element: Element, animationName: string) {
  const event = new Event('animationend', { bubbles: true })
  Object.defineProperty(event, 'animationName', { value: animationName })
  fireEvent(element, event)
}

function DrawerHarness({ onEntered = () => {} }: { onEntered?: () => void }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>Open settings</button>
      <Drawer
        open={open}
        title="Settings"
        onClose={() => setOpen(false)}
        onEntered={onEntered}
        testId="drawer-root"
      >
        <button type="button">First action</button>
        <button type="button">Last action</button>
      </Drawer>
    </>
  )
}

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  document.getElementById('root')?.remove()
})

describe('Drawer', () => {
  it('keeps one shell mounted and completes deterministic enter and exit phases', async () => {
    const onEntered = vi.fn()
    const { rerender } = render(
      <Drawer open={false} title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )

    const root = screen.getByTestId('drawer-root')
    const dialog = screen.getByRole('dialog', { hidden: true })
    expect(root).toHaveAttribute('data-state', 'closed')
    expect(root).toHaveAttribute('aria-hidden', 'true')

    rerender(
      <Drawer open title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'opening'))
    expect(screen.getByRole('dialog', { name: 'Settings' })).toBe(dialog)

    endAnimation(dialog, 'drawer-panel-in')
    expect(root).toHaveAttribute('data-state', 'open')
    expect(onEntered).toHaveBeenCalledTimes(1)

    rerender(
      <Drawer open={false} title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'closing'))
    endAnimation(dialog, 'drawer-panel-out')
    expect(root).toHaveAttribute('data-state', 'closed')
    expect(dialog).toBeInTheDocument()
  })

  it('ignores stale animation events when direction reverses quickly', async () => {
    const { rerender } = render(
      <Drawer open={false} title="Settings" onClose={() => {}} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    const root = screen.getByTestId('drawer-root')
    const dialog = screen.getByRole('dialog', { hidden: true })

    rerender(<Drawer open title="Settings" onClose={() => {}} testId="drawer-root"><p>Content</p></Drawer>)
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'opening'))
    rerender(<Drawer open={false} title="Settings" onClose={() => {}} testId="drawer-root"><p>Content</p></Drawer>)
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'closing'))
    rerender(<Drawer open title="Settings" onClose={() => {}} testId="drawer-root"><p>Content</p></Drawer>)
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'opening'))

    endAnimation(dialog, 'drawer-panel-out')
    expect(root).toHaveAttribute('data-state', 'opening')
    endAnimation(dialog, 'drawer-panel-in')
    expect(root).toHaveAttribute('data-state', 'open')
  })

  it('settles motion when animationend is lost', () => {
    vi.useFakeTimers()
    const onEntered = vi.fn()
    const { rerender } = render(
      <Drawer open={false} title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    const root = screen.getByTestId('drawer-root')

    rerender(
      <Drawer open title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    expect(root).toHaveAttribute('data-state', 'opening')
    act(() => vi.advanceTimersByTime(320))
    expect(root).toHaveAttribute('data-state', 'open')
    expect(onEntered).toHaveBeenCalledTimes(1)

    rerender(
      <Drawer open={false} title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    expect(root).toHaveAttribute('data-state', 'closing')
    act(() => vi.advanceTimersByTime(240))
    expect(root).toHaveAttribute('data-state', 'closed')
  })

  it('skips motion consistently when reduced motion is requested', async () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({
      matches: true,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }))
    const onEntered = vi.fn()
    const { rerender } = render(
      <Drawer open={false} title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    const root = screen.getByTestId('drawer-root')

    rerender(
      <Drawer open title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'open'))
    expect(onEntered).toHaveBeenCalledTimes(1)

    rerender(
      <Drawer open={false} title="Settings" onClose={() => {}} onEntered={onEntered} testId="drawer-root">
        <p>Content</p>
      </Drawer>,
    )
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'closed'))
  })

  it('traps focus, closes from Escape or backdrop, and restores the opener', async () => {
    const user = userEvent.setup()
    render(<DrawerHarness />)

    const opener = screen.getByRole('button', { name: 'Open settings' })
    await user.click(opener)
    const dialog = screen.getByRole('dialog', { name: 'Settings' })
    await waitFor(() => expect(dialog).toHaveFocus())

    await user.tab()
    expect(screen.getByRole('button', { name: 'First action' })).toHaveFocus()
    await user.tab({ shift: true })
    expect(screen.getByRole('button', { name: 'Last action' })).toHaveFocus()

    fireEvent.keyDown(document, { key: 'Escape' })
    await waitFor(() => expect(screen.getByTestId('drawer-root')).toHaveAttribute('data-state', 'closing'))
    expect(opener).not.toHaveFocus()

    endAnimation(dialog, 'drawer-panel-out')
    expect(opener).toHaveFocus()
    await user.click(opener)
    await waitFor(() => expect(screen.getByTestId('drawer-root')).toHaveAttribute('data-state', 'opening'))
    fireEvent.mouseDown(screen.getByTestId('drawer-root').querySelector('.ui-drawer-backdrop')!)
    await waitFor(() => expect(screen.getByTestId('drawer-root')).toHaveAttribute('data-state', 'closing'))
  })

  it('keeps a parent drawer open when Escape belongs to a nested modal', () => {
    const onDrawerClose = vi.fn()
    const onModalClose = vi.fn()
    render(
      <>
        <Drawer open title="Settings" onClose={onDrawerClose}>
          <button type="button">Drawer action</button>
        </Drawer>
        <Modal title="Discard changes?" onClose={onModalClose}>
          <p>Confirmation</p>
        </Modal>
      </>,
    )

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onModalClose).toHaveBeenCalledTimes(1)
    expect(onDrawerClose).not.toHaveBeenCalled()
  })

  it('makes the app root inert without changing body overflow', async () => {
    const appRoot = document.createElement('div')
    appRoot.id = 'root'
    document.body.appendChild(appRoot)
    document.body.style.overflow = 'auto'

    const { rerender, unmount } = render(
      <Drawer open={false} title="Settings" onClose={() => {}} testId="drawer-root"><p>Content</p></Drawer>,
      { container: appRoot },
    )
    rerender(<Drawer open title="Settings" onClose={() => {}} testId="drawer-root"><p>Content</p></Drawer>)

    await waitFor(() => expect(appRoot.inert).toBe(true))
    expect(document.body.style.overflow).toBe('auto')

    rerender(<Drawer open={false} title="Settings" onClose={() => {}} testId="drawer-root"><p>Content</p></Drawer>)
    await waitFor(() => expect(screen.getByTestId('drawer-root')).toHaveAttribute('data-state', 'closing'))
    expect(appRoot.inert).toBe(true)
    endAnimation(screen.getByRole('dialog', { name: 'Settings' }), 'drawer-panel-out')
    await waitFor(() => expect(appRoot.inert).toBe(false))
    expect(document.body.style.overflow).toBe('auto')
    unmount()
    document.body.style.overflow = ''
  })
})
