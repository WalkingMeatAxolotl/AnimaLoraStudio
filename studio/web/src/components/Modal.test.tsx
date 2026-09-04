import { useState } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import Modal from './Modal'

function FocusHarness() {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>Open modal</button>
      {open && (
        <Modal
          title="Switch family"
          description="Review the affected values."
          onClose={() => setOpen(false)}
          footer={(
            <>
              <button type="button" tabIndex={-1}>Programmatic action</button>
              <button type="button" onClick={() => setOpen(false)}>Cancel</button>
              <button type="button">Apply</button>
            </>
          )}
        >
          <p>Changes</p>
        </Modal>
      )}
    </>
  )
}

describe('Modal', () => {
  it('provides an accessible labelled dialog in a body portal', () => {
    render(
      <Modal title="Export bundle" description="Choose what to include." onClose={() => {}}>
        <p>Options</p>
      </Modal>,
    )

    const dialog = screen.getByRole('dialog', { name: 'Export bundle' })
    expect(dialog.parentElement).toBe(document.body.lastElementChild)
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(dialog).toHaveAccessibleDescription('Choose what to include.')
    expect(dialog).toHaveClass('max-w-[560px]')
  })

  it('supports a wide master-detail panel with header actions', () => {
    render(
      <Modal
        title="Announcements"
        size="wide"
        headerActions={<button type="button">Close announcements</button>}
        onClose={() => {}}
      >
        <p>News</p>
      </Modal>,
    )

    const dialog = screen.getByRole('dialog', { name: 'Announcements' })
    expect(dialog).toHaveClass('w-[80vw]', 'max-w-[1440px]')
    expect(dialog).not.toHaveClass('w-full')
    expect(screen.getByRole('button', { name: 'Close announcements' })).toBeInTheDocument()
  })

  it('locks and restores body scrolling for the mounted lifetime', () => {
    document.body.style.overflow = 'auto'
    const { unmount } = render(
      <Modal title="Locked" onClose={() => {}}>
        <p>Content</p>
      </Modal>,
    )

    expect(document.body.style.overflow).toBe('hidden')
    unmount()
    expect(document.body.style.overflow).toBe('auto')
    document.body.style.overflow = ''
  })

  it('traps focus and restores it to the opener after close', async () => {
    const user = userEvent.setup()
    render(<FocusHarness />)

    const opener = screen.getByRole('button', { name: 'Open modal' })
    await user.click(opener)
    const cancel = screen.getByRole('button', { name: 'Cancel' })
    const apply = screen.getByRole('button', { name: 'Apply' })
    await waitFor(() => expect(cancel).toHaveFocus())

    await user.tab({ shift: true })
    expect(apply).toHaveFocus()
    await user.tab()
    expect(cancel).toHaveFocus()

    await user.click(cancel)
    expect(opener).toHaveFocus()
  })

  it('closes on Escape and backdrop mouse-down but not panel mouse-down', () => {
    const onClose = vi.fn()
    render(
      <Modal title="Confirm" onClose={onClose} testId="modal-backdrop">
        <button type="button">Inside</button>
      </Modal>,
    )

    fireEvent.mouseDown(screen.getByRole('dialog'))
    expect(onClose).not.toHaveBeenCalled()

    fireEvent.mouseDown(screen.getByTestId('modal-backdrop'))
    expect(onClose).toHaveBeenCalledTimes(1)

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(2)
  })

  it('supports a form panel and configurable dismissal', () => {
    const onClose = vi.fn()
    const onSubmit = vi.fn((event: React.FormEvent<HTMLFormElement>) => event.preventDefault())
    render(
      <Modal
        as="form"
        title="Create version"
        onClose={onClose}
        onSubmit={onSubmit}
        closeOnBackdrop={false}
        closeOnEscape={false}
        testId="modal-backdrop"
        footer={<button type="submit">Create</button>}
      >
        <input aria-label="Label" />
      </Modal>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Create' }))
    expect(onSubmit).toHaveBeenCalledTimes(1)
    fireEvent.mouseDown(screen.getByTestId('modal-backdrop'))
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).not.toHaveBeenCalled()
  })
})
