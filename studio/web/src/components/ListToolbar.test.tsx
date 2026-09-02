import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ListToolbar from './ListToolbar'

function slotOrder(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll('[data-list-toolbar-slot]'))
    .map((element) => element.getAttribute('data-list-toolbar-slot') ?? '')
}

describe('ListToolbar', () => {
  it('renders a named region in search, filter, sort order', () => {
    const { container } = render(
      <ListToolbar
        id="projects-list-toolbar"
        ariaLabel="Project filters"
        search={<input aria-label="Search projects" />}
        filters={<select aria-label="Status"><option>All</option></select>}
        sort={<select aria-label="Sort"><option>Updated</option></select>}
      />,
    )

    const toolbar = screen.getByRole('region', { name: 'Project filters' })
    expect(toolbar).toHaveAttribute('id', 'projects-list-toolbar')
    expect(toolbar).toHaveClass('list-toolbar')
    expect(slotOrder(container)).toEqual(['search', 'filters', 'sort'])
    expect(screen.getByRole('textbox', { name: 'Search projects' }))
      .toHaveProperty('tabIndex', 0)
  })

  it('omits empty trailing slots and keeps hidden regions out of layout', () => {
    const { container } = render(
      <ListToolbar
        ariaLabel="Queue filters"
        search={<input aria-label="Search queue" />}
        hidden
      />,
    )

    const toolbar = screen.getByRole('region', { hidden: true })
    expect(toolbar).toHaveAttribute('aria-label', 'Queue filters')
    expect(toolbar).toHaveAttribute('hidden')
    expect(slotOrder(container)).toEqual(['search'])
    expect(container.querySelector('.list-toolbar-controls')).not.toBeInTheDocument()
  })

  it('preserves caller attributes without allowing the region semantics to drift', () => {
    render(
      <ListToolbar
        ariaLabel="Data job filters"
        search={<input aria-label="Search data jobs" />}
        data-testid="jobs-toolbar"
        className="custom-toolbar"
      />,
    )

    expect(screen.getByTestId('jobs-toolbar')).toHaveClass('list-toolbar', 'custom-toolbar')
    expect(screen.getByTestId('jobs-toolbar')).toHaveAttribute('role', 'region')
  })
})
