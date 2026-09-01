import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ActionGroup from './ActionGroup'

function slotOrder(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll('[data-action-slot]'))
    .map((element) => element.getAttribute('data-action-slot') ?? '')
}

describe('ActionGroup', () => {
  it('keeps status, secondary actions, and the primary action in semantic order', () => {
    const { container } = render(
      <ActionGroup
        aria-label="Caption actions"
        status={<span>2 unsaved</span>}
        secondary={<button type="button">Restore</button>}
        primary={<button type="button">Save</button>}
      />,
    )

    expect(screen.getByRole('group', { name: 'Caption actions' })).toHaveClass(
      'flex-wrap',
      'justify-end',
      'gap-related',
    )
    expect(slotOrder(container)).toEqual(['status', 'secondary', 'primary'])
    expect(screen.getByRole('button', { name: 'Save' }).closest('[data-action-slot]'))
      .toHaveAttribute('data-action-slot', 'primary')
  })

  it('omits empty slots without changing the primary position', () => {
    const { container } = render(
      <ActionGroup primary={<button type="button">Submit</button>} />,
    )

    expect(slotOrder(container)).toEqual(['primary'])
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument()
    expect(container.firstElementChild).not.toHaveAttribute('role')
  })
})
