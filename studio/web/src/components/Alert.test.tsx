import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Alert, { alertClassName } from './Alert'

describe('Alert', () => {
  it('uses the informational medium treatment by default', () => {
    render(<Alert>Background sync is active.</Alert>)

    const alert = screen.getByText('Background sync is active.').closest('.alert')
    expect(alert).toHaveClass('alert', 'alert-info')
    expect(alert).not.toHaveClass('alert-sm')
    expect(alert?.querySelector('.alert-icon')).toHaveAttribute('aria-hidden', 'true')
  })

  it('supports a title, action, compact danger tone, and explicit live semantics', () => {
    render(
      <Alert
        tone="danger"
        size="sm"
        role="alert"
        title="Could not load"
        action={<button type="button">Retry</button>}
      >
        Check the connection and try again.
      </Alert>,
    )

    const alert = screen.getByRole('alert')
    expect(alert).toHaveClass('alert-danger', 'alert-sm')
    expect(screen.getByText('Could not load')).toHaveClass('alert-title')
    expect(screen.getByRole('button', { name: 'Retry' }).parentElement)
      .toHaveClass('alert-action')
  })

  it('allows the semantic icon to be omitted and merges custom classes', () => {
    render(<Alert icon={false} className="font-mono">Technical details</Alert>)

    const alert = screen.getByText('Technical details').closest('.alert')
    expect(alert).toHaveClass('font-mono')
    expect(alert?.querySelector('.alert-icon')).toBeNull()
  })

  it('builds stable classes for compatibility wrappers', () => {
    expect(alertClassName({ tone: 'success', size: 'sm', className: 'shadow-lg' }))
      .toBe('alert alert-success alert-sm shadow-lg')
  })
})
