import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ProgressBar from './ProgressBar'

describe('ProgressBar', () => {
  it('exposes a labelled determinate progressbar and clamps its value', () => {
    const { rerender } = render(
      <ProgressBar label="Upload progress" value={35} valueText="35%, 7 MB of 20 MB" />,
    )

    const progress = screen.getByRole('progressbar', { name: 'Upload progress' })
    expect(progress).toHaveAttribute('aria-valuemin', '0')
    expect(progress).toHaveAttribute('aria-valuemax', '100')
    expect(progress).toHaveAttribute('aria-valuenow', '35')
    expect(progress).toHaveAttribute('aria-valuetext', '35%, 7 MB of 20 MB')
    expect(progress).toHaveAttribute('data-state', 'determinate')
    expect(progress.querySelector('.ui-progress-fill')).toHaveStyle({ transform: 'scaleX(0.35)' })

    rerender(<ProgressBar label="Upload progress" value={140} />)
    expect(progress).toHaveAttribute('aria-valuenow', '100')
    expect(progress.querySelector('.ui-progress-fill')).toHaveStyle({ transform: 'scaleX(1)' })
  })

  it('omits aria-valuenow when progress is indeterminate', () => {
    render(<ProgressBar label="Preparing generation" value={null} size="xs" />)

    const progress = screen.getByRole('progressbar', { name: 'Preparing generation' })
    expect(progress).not.toHaveAttribute('aria-valuenow')
    expect(progress).toHaveAttribute('data-state', 'indeterminate')
    expect(progress).toHaveClass('ui-progress', 'ui-progress-xs', 'ui-progress-accent')
    expect(progress.querySelector('.ui-progress-fill')).not.toHaveAttribute('style')
  })

  it('supports a custom max, success tone, size, and class name', () => {
    render(
      <ProgressBar
        label="Copy files"
        value={3}
        max={4}
        size="md"
        tone="success"
        className="mt-related"
      />,
    )

    const progress = screen.getByRole('progressbar', { name: 'Copy files' })
    expect(progress).toHaveAttribute('aria-valuemax', '4')
    expect(progress).toHaveAttribute('aria-valuenow', '3')
    expect(progress).toHaveClass('ui-progress-md', 'ui-progress-success', 'mt-related')
    expect(progress.querySelector('.ui-progress-fill')).toHaveStyle({ transform: 'scaleX(0.75)' })
  })
})
