import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import SaveIndicator from './SaveIndicator'

describe('SaveIndicator', () => {
  it('keeps a stable polite live region while autosave state changes', () => {
    const { rerender } = render(<SaveIndicator status={{ state: 'idle' }} />)
    const status = screen.getByRole('status')

    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).toHaveAttribute('aria-atomic', 'true')
    expect(status).toHaveAttribute('data-state', 'idle')
    expect(status).toBeEmptyDOMElement()

    rerender(<SaveIndicator status={{ state: 'saving' }} />)
    expect(status).toHaveAttribute('data-state', 'saving')
    expect(status).not.toBeEmptyDOMElement()

    rerender(<SaveIndicator status={{ state: 'saved', at: Date.now() }} />)
    expect(status).toHaveAttribute('data-state', 'saved')
    expect(status.querySelector('svg')).toHaveAttribute('aria-hidden', 'true')
  })

  it('exposes the full failure and can suppress a duplicate live announcement', () => {
    const { rerender } = render(
      <SaveIndicator status={{ state: 'error', error: 'Disk is read-only' }} />,
    )

    expect(screen.getByRole('status')).toHaveAttribute('title', 'Disk is read-only')
    expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite')
    expect(screen.getByRole('status')).toHaveClass('text-err')

    rerender(
      <SaveIndicator
        status={{ state: 'error', error: 'Disk is read-only' }}
        announceError={false}
      />,
    )
    expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'off')
  })
})
