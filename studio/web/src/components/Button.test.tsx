import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Button, { buttonClassName } from './Button'

describe('Button', () => {
  it('defaults to a non-submitting secondary button', () => {
    render(<Button>Save</Button>)
    const button = screen.getByRole('button', { name: 'Save' })
    expect(button).toHaveAttribute('type', 'button')
    expect(button).toHaveClass('btn', 'btn-secondary')
  })

  it('maps variants and sizes to the shared primitive classes', () => {
    render(<Button variant="warning" size="xs">Cancel</Button>)
    expect(screen.getByRole('button', { name: 'Cancel' }))
      .toHaveClass('btn-warn', 'btn-xs')
  })

  it('disables activation and exposes busy state while loading', () => {
    render(<Button loading>Saving</Button>)
    const button = screen.getByRole('button', { name: 'Saving' })
    expect(button).toBeDisabled()
    expect(button).toHaveAttribute('aria-busy', 'true')
    expect(button.querySelector('.btn-spinner')).not.toBeNull()
  })

  it('supports an accessible icon-only control', () => {
    render(<Button iconOnly size="sm" aria-label="Close"><span aria-hidden>×</span></Button>)
    expect(screen.getByRole('button', { name: 'Close' }))
      .toHaveClass('btn-icon', 'btn-sm')
  })

  it('builds classes for links without changing link semantics', () => {
    expect(buttonClassName({ variant: 'ghost', size: 'sm', className: 'no-underline' }))
      .toBe('btn btn-ghost btn-sm no-underline')
  })
})
