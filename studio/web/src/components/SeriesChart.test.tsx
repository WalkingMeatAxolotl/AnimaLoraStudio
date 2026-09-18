import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SeriesChart } from './SeriesChart'

describe('SeriesChart accessibility', () => {
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      width: 320,
      height: 160,
      x: 0,
      y: 0,
      top: 0,
      right: 320,
      bottom: 160,
      left: 0,
      toJSON: () => ({}),
    })
  })

  afterEach(() => { vi.restoreAllMocks() })

  it('exposes a concise text alternative while keeping the SVG decorative', () => {
    render(
      <SeriesChart
        data={[{ step: 1, value: 0.8 }, { step: 2, value: 0.6 }]}
        rawColor="gray"
        smoothColor="blue"
        emaAlpha={1}
        yFormat={(value) => value.toFixed(2)}
        height={160}
        ariaLabel="Loss trend"
        summary="2 data points, steps 1 to 2; current 0.60, range 0.60–0.80"
      />,
    )

    const chart = screen.getByRole('img', { name: /Loss trend.*2 data points/ })
    expect(chart.querySelector('svg')).toHaveAttribute('aria-hidden', 'true')
  })

  it('uses caller-provided empty copy', () => {
    render(
      <SeriesChart
        data={[]}
        rawColor="gray"
        smoothColor="blue"
        emaAlpha={1}
        yFormat={String}
        height={160}
        ariaLabel="Loss trend"
        summary="No data points yet"
        emptyLabel="Waiting for metric data…"
      />,
    )

    expect(screen.getByRole('status')).toHaveTextContent('Waiting for metric data…')
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })
})
