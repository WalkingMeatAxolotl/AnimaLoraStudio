import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
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

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
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

  it('keeps the existing timeout policy for transient feedback', () => {
    vi.useFakeTimers()
    render(
      <ToastProvider>
        <ToastTrigger kind="info" />
      </ToastProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Show info' }))
    expect(screen.getByText('info message')).toBeInTheDocument()

    act(() => { vi.advanceTimersByTime(3000) })
    expect(screen.queryByText('info message')).not.toBeInTheDocument()
  })
})
