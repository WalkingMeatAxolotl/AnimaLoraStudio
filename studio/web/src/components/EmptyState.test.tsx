import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import EmptyState from './EmptyState'

describe('EmptyState', () => {
  it('renders a title and supporting description with the shared hierarchy', () => {
    render(<EmptyState title="Nothing here" description="Create an item to get started." />)

    expect(screen.getByText('Nothing here')).toHaveClass('empty-state-title')
    expect(screen.getByText('Create an item to get started.'))
      .toHaveClass('empty-state-description')
  })

  it('supports compact description-only states', () => {
    render(
      <EmptyState
        data-testid="empty"
        size="sm"
        description="No matching results"
      />,
    )

    expect(screen.getByTestId('empty')).toHaveClass('card', 'empty-state', 'empty-state-sm')
    expect(screen.queryByText('Nothing here')).not.toBeInTheDocument()
  })

  it('can be embedded in an existing surface without a nested card', () => {
    render(<EmptyState embedded data-testid="embedded" description="No images" />)
    expect(screen.getByTestId('embedded')).toHaveClass('empty-state')
    expect(screen.getByTestId('embedded')).not.toHaveClass('card')
  })

  it('accepts an action and standard HTML attributes', () => {
    render(
      <EmptyState
        aria-live="polite"
        title="No projects"
        description="Create your first project."
        action={<button type="button">Create project</button>}
      />,
    )

    const state = screen.getByText('No projects').closest('.empty-state')
    expect(state).toHaveAttribute('aria-live', 'polite')
    expect(screen.getByRole('button', { name: 'Create project' }))
      .toBeInTheDocument()
  })
})
