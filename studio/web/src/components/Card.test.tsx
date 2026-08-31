import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Card, { cardClassName } from './Card'

describe('Card', () => {
  it('renders the standard surface without implicit padding', () => {
    render(<Card data-testid="card">Content</Card>)
    expect(screen.getByTestId('card')).toHaveClass('card')
    expect(screen.getByTestId('card')).not.toHaveClass('card-pad-md')
  })

  it('maps tone, radius, padding, and interaction to shared classes', () => {
    render(
      <Card
        data-testid="card"
        tone="sunken"
        radius="compact"
        padding="md"
        interactive
      >
        Content
      </Card>,
    )
    expect(screen.getByTestId('card')).toHaveClass(
      'card',
      'card-sunken',
      'card-compact',
      'card-pad-md',
      'card-hover',
    )
  })

  it('supports semantic section and article elements', () => {
    const { rerender } = render(<Card as="section" data-testid="card">Section</Card>)
    expect(screen.getByTestId('card').tagName).toBe('SECTION')

    rerender(<Card as="article" data-testid="card">Article</Card>)
    expect(screen.getByTestId('card').tagName).toBe('ARTICLE')
  })

  it('builds classes for interactive elements that retain their own semantics', () => {
    expect(cardClassName({
      interactive: true,
      radius: 'compact',
      className: 'cursor-pointer',
    })).toBe('card card-compact card-hover cursor-pointer')
  })
})
