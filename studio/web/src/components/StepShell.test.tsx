import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import StepShell from './StepShell'

describe('StepShell', () => {
  it('uses the shared page inset while keeping the toolbar outside the scroll content', () => {
    render(
      <StepShell
        idx={1}
        title="Queue"
        belowHeader={<div data-testid="toolbar">Filters</div>}
      >
        <div data-testid="content">Content</div>
      </StepShell>,
    )

    expect(screen.getByTestId('content').parentElement).toHaveClass('p-page')
    expect(screen.getByTestId('toolbar').nextElementSibling)
      .toBe(screen.getByTestId('content').parentElement)
  })
})
