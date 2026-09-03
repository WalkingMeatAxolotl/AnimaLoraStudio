import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import PageHeader from './PageHeader'

describe('PageHeader', () => {
  it('applies the shared page title, description, and shell rhythm', () => {
    const { container } = render(
      <PageHeader title="Projects" subtitle="Manage training workspaces." />,
    )

    expect(container.firstElementChild).toHaveClass(
      'ui-page-header',
      'px-page',
      'pt-page-start',
      'pb-section',
    )
    expect(screen.getByRole('heading', { level: 1, name: 'Projects' }))
      .toHaveClass('type-page-title')
    expect(screen.getByText('Manage training workspaces.'))
      .toHaveClass('type-page-description', 'mt-related')
  })

  it('lets tabs replace the description without changing the title hierarchy', () => {
    render(
      <PageHeader
        title="Settings"
        subtitle="Hidden description"
        tabs={<nav aria-label="Settings sections">Tabs</nav>}
      />,
    )

    expect(screen.getByRole('heading', { level: 1, name: 'Settings' }))
      .toHaveClass('type-page-title')
    expect(screen.queryByText('Hidden description')).not.toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Settings sections' }))
      .toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Settings sections' }).parentElement)
      .toHaveClass('mt-field')
  })

  it('preserves the action and top-right slots', () => {
    render(
      <PageHeader
        title="Queue"
        actions={<button type="button">Refresh</button>}
        topRight={<span>Phase 2</span>}
        sticky
      />,
    )

    expect(screen.getByRole('button', { name: 'Refresh' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Refresh' }).parentElement)
      .toHaveClass('ui-page-header-actions')
    expect(screen.getByRole('button', { name: 'Refresh' }).parentElement?.parentElement)
      .toHaveClass('ui-page-header-layout')
    expect(screen.getByText('Phase 2')).toBeInTheDocument()
    expect(screen.getByText('Phase 2').parentElement)
      .toHaveClass('ui-page-header-top-right')
    expect(screen.getByRole('heading', { name: 'Queue' }).closest('.sticky')).toBeInTheDocument()
  })
})
