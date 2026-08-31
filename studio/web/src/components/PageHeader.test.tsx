import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import PageHeader from './PageHeader'

describe('PageHeader', () => {
  it('applies the shared page title and description hierarchy', () => {
    render(<PageHeader title="Projects" subtitle="Manage training workspaces." />)

    expect(screen.getByRole('heading', { level: 1, name: 'Projects' }))
      .toHaveClass('type-page-title')
    expect(screen.getByText('Manage training workspaces.'))
      .toHaveClass('type-page-description')
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
    expect(screen.getByRole('navigation', { name: 'Settings sections' })).toBeInTheDocument()
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
    expect(screen.getByText('Phase 2')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Queue' }).closest('.sticky')).toBeInTheDocument()
  })
})
