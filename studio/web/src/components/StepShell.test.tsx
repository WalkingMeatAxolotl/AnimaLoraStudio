import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import StepShell from './StepShell'

describe('StepShell', () => {
  it('uses the shared page inset while keeping the toolbar outside workspace content', () => {
    render(
      <StepShell
        title="Queue"
        belowHeader={<div data-testid="toolbar">Filters</div>}
      >
        <div data-testid="content">Content</div>
      </StepShell>,
    )

    const shell = screen.getByTestId('content').closest('[data-step-shell]')
    const content = screen.getByTestId('content').parentElement

    expect(shell).toHaveClass('h-full', 'min-h-0', 'relative')
    expect(content).toHaveClass('p-page', 'overflow-hidden')
    expect(content).toHaveAttribute('data-inset', 'page')
    expect(screen.getByTestId('toolbar').nextElementSibling).toBe(content)
    expect(screen.getByRole('heading', { level: 1 }).closest('.ui-page-header'))
      .not.toHaveClass('sticky')
  })

  it('allows specialized workspaces to own edge geometry explicitly', () => {
    render(
      <StepShell title="Canvas" inset="none">
        <div data-testid="canvas">Canvas</div>
      </StepShell>,
    )

    const content = screen.getByTestId('canvas').parentElement
    expect(content).toHaveAttribute('data-inset', 'none')
    expect(content).not.toHaveClass('p-page')
    expect(content).toHaveClass('overflow-hidden')
  })

  it('mounts the task log footer after the bounded workspace content', () => {
    render(
      <StepShell
        title="Train"
        logSources={[{
          key: 'train',
          label: 'Train',
          status: 'done',
          lines: ['Complete'],
        }]}
      >
        <div data-testid="workspace">Workspace</div>
      </StepShell>,
    )

    const content = screen.getByTestId('workspace').parentElement
    const spacer = content?.nextElementSibling
    const drawer = spacer?.nextElementSibling

    expect(spacer).toHaveAttribute('aria-hidden', 'true')
    expect(drawer).toContainElement(screen.getByRole('button', { name: /Train/ }))
  })
})
