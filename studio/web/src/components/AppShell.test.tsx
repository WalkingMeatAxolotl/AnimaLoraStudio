import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import AppShell, { APP_MAIN_ID } from './AppShell'

describe('AppShell', () => {
  it('provides the workspace landmarks and a keyboard skip target', () => {
    const { container } = render(
      <AppShell
        navigation={<aside aria-label="Primary navigation">Navigation</aside>}
        topbar={<header>Topbar</header>}
        skipLabel="Skip to content"
        overlay={<div data-testid="overlay-slot">Overlay</div>}
      >
        <h1>Workspace</h1>
      </AppShell>,
    )

    const shell = container.firstElementChild as HTMLElement
    const workspace = shell.lastElementChild as HTMLElement
    const main = screen.getByRole('main')
    const overlay = screen.getByTestId('overlay-slot')

    expect(shell).toHaveClass('ui-app-shell')
    expect(screen.getByRole('complementary', { name: 'Primary navigation' })).toBeInTheDocument()
    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(main).toHaveAttribute('id', APP_MAIN_ID)
    expect(main).toHaveAttribute('tabindex', '-1')
    expect(main).toHaveClass('ui-app-shell-main')
    expect(screen.getByRole('link', { name: 'Skip to content' })).toHaveAttribute('href', `#${APP_MAIN_ID}`)
    expect(workspace).toContainElement(screen.getByRole('main'))
    expect(workspace).not.toContainElement(overlay)
    expect(overlay.previousElementSibling).toBe(shell)
  })

  it('supports an explicit main target without changing slot order', () => {
    const { container } = render(
      <AppShell
        navigation={<aside>Navigation</aside>}
        topbar={<header>Topbar</header>}
        skipLabel="Skip"
        mainId="workspace-content"
      >
        Content
      </AppShell>,
    )

    const shell = container.firstElementChild as HTMLElement
    const workspace = shell.lastElementChild as HTMLElement
    expect(screen.getByRole('link', { name: 'Skip' })).toHaveAttribute('href', '#workspace-content')
    expect(workspace.children[0]).toBe(screen.getByRole('banner'))
    expect(workspace.children[1]).toBe(screen.getByRole('main'))
  })
})
