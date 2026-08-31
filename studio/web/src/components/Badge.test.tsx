import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Badge from './Badge'

describe('Badge', () => {
  it('uses a neutral tone by default', () => {
    render(<Badge>Pending</Badge>)
    expect(screen.getByText('Pending')).toHaveClass('badge', 'badge-neutral')
  })

  it('maps semantic tones to existing visual tokens', () => {
    render(<Badge tone="danger">Failed</Badge>)
    expect(screen.getByText('Failed')).toHaveClass('badge-err')
  })

  it('supports dense metadata sizing', () => {
    render(<Badge size="sm">Release</Badge>)
    expect(screen.getByText('Release')).toHaveClass('badge-sm')
  })

  it('adds the shared active-work indicator', () => {
    const { container } = render(<Badge tone="accent" active>Running</Badge>)
    expect(container.querySelector('.dot.dot-running')).not.toBeNull()
    expect(container.querySelector('.dot')).toHaveAttribute('aria-hidden', 'true')
  })

  it('preserves caller classes and native span attributes', () => {
    render(<Badge className="shrink-0" title="Current state">Ready</Badge>)
    expect(screen.getByTitle('Current state')).toHaveClass('shrink-0')
  })
})
