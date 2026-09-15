import { useState } from 'react'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Drawer from './Drawer'
import { ToastProvider, useToast } from './Toast'

type Kind = 'info' | 'success' | 'error'

function ToastTrigger({ kind }: { kind: Kind }) {
  const { toast } = useToast()
  return (
    <button type="button" onClick={() => toast(`${kind} message`, kind)}>
      Show {kind}
    </button>
  )
}

function DrawerFeedback() {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>Open settings</button>
      <Drawer open={open} title="Settings" onClose={() => setOpen(false)}>
        <ToastTrigger kind="info" />
        <ToastTrigger kind="error" />
      </Drawer>
    </>
  )
}

function createAppRoot() {
  const root = document.createElement('div')
  root.id = 'root'
  root.inert = false
  document.body.append(root)
  return root
}

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  document.getElementById('root')?.remove()
})

describe('ToastProvider', () => {
  it.each([
    ['info', 'alert-info', 'status'],
    ['success', 'alert-success', 'status'],
    ['error', 'alert-danger', 'alert'],
  ] as const)('maps %s feedback onto the shared Alert contract', (kind, toneClass, role) => {
    vi.useFakeTimers()
    render(
      <ToastProvider>
        <ToastTrigger kind={kind} />
      </ToastProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: `Show ${kind}` }))

    const toast = screen.getByRole(role)
    expect(toast).toHaveClass('alert', toneClass, 'shadow-lg')
    expect(toast).toHaveAttribute('aria-atomic', 'true')
  })

  it.each([
    ['info', 3000],
    ['success', 3000],
    ['error', 6000],
  ] as const)('keeps the %s timeout at %i ms', (kind, timeout) => {
    vi.useFakeTimers()
    render(
      <ToastProvider>
        <ToastTrigger kind={kind} />
      </ToastProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: `Show ${kind}` }))
    act(() => { vi.advanceTimersByTime(timeout - 1) })
    expect(screen.getByText(`${kind} message`)).toBeInTheDocument()

    act(() => { vi.advanceTimersByTime(1) })
    expect(screen.queryByText(`${kind} message`)).not.toBeInTheDocument()
  })

  it('keeps one ordered live output outside the inert app without changing Drawer focus or protection', () => {
    vi.useFakeTimers()
    const appRoot = createAppRoot()
    render(<ToastProvider><DrawerFeedback /></ToastProvider>, { container: appRoot })

    const opener = screen.getByRole('button', { name: 'Open settings' })
    opener.focus()
    fireEvent.click(opener)
    act(() => { vi.advanceTimersByTime(400) })
    expect(appRoot.inert).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Show info' }))
    const errorTrigger = screen.getByRole('button', { name: 'Show error' })
    errorTrigger.focus()
    fireEvent.click(errorTrigger)

    const info = screen.getByRole('status')
    const error = screen.getByRole('alert')
    for (const notice of [info, error]) {
      expect(document.body).toContainElement(notice)
      expect(appRoot).not.toContainElement(notice)
      expect(notice).toHaveAttribute('aria-atomic', 'true')
    }
    expect(info.compareDocumentPosition(error) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(errorTrigger).toHaveFocus()
    expect(appRoot.inert).toBe(true)

    act(() => { vi.advanceTimersByTime(3000) })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(error).toBeInTheDocument()
    expect(appRoot.inert).toBe(true)

    fireEvent.keyDown(document, { key: 'Escape' })
    act(() => { vi.advanceTimersByTime(240) })
    expect(appRoot.inert).toBe(false)
    expect(opener).toHaveFocus()
    expect(error).toBeInTheDocument()

    act(() => { vi.advanceTimersByTime(2760) })
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('removes portalled feedback when the provider unmounts', () => {
    vi.useFakeTimers()
    const appRoot = createAppRoot()
    const { unmount } = render(
      <ToastProvider><ToastTrigger kind="error" /></ToastProvider>,
      { container: appRoot },
    )
    fireEvent.click(screen.getByRole('button', { name: 'Show error' }))
    const error = screen.getByRole('alert')
    const host = error.parentElement
    expect(appRoot).not.toContainElement(error)
    expect(host?.parentElement).toBe(document.body)

    unmount()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(host).not.toBeInTheDocument()
  })
})
