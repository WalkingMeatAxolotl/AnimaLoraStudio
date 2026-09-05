import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import PreprocessToolsBar from './PreprocessToolsBar'

describe('PreprocessToolsBar', () => {
  it('keeps every tool as a native link, names navigation and identifies the current route', () => {
    render(<MemoryRouter><PreprocessToolsBar current="crop" projectId={3} versionId={8} /></MemoryRouter>)
    const nav = screen.getByRole('navigation', { name: '预处理工具导航' })
    expect(nav).toHaveClass('ui-selection-underline', 'shrink-0')
    const links = within(nav).getAllByRole('link')
    expect(links).toHaveLength(5)
    expect(links.map(link => link.getAttribute('href'))).toEqual([
      '/projects/3/v/8/preprocess',
      '/projects/3/v/8/preprocess?tool=dedupe',
      '/projects/3/v/8/preprocess?tool=upscale',
      '/projects/3/v/8/preprocess?tool=crop',
      '/projects/3/v/8/preprocess?tool=inpaint',
    ])
    expect(links[3]).toHaveAttribute('aria-current', 'page')
    expect(links[3]).toHaveAttribute('data-state', 'active')
    expect(links.filter(link => link.hasAttribute('aria-current'))).toHaveLength(1)
    expect(screen.queryByRole('tablist')).not.toBeInTheDocument()
    for (const link of links) expect(link).toHaveClass('ui-selection-item')
  })
})
