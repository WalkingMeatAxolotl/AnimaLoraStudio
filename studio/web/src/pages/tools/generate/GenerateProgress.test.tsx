import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import GenerateProgressBar, { type GenerateProgress } from './GenerateProgress'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, number>) => {
      if (key === 'generate.pipelineProgress') return 'Pipeline progress'
      if (key === 'generate.progressPreparing') return 'Preparing…'
      if (key === 'generate.phaseLoad') return 'Loading model…'
      if (key === 'generate.phaseClip') return 'Encoding prompt…'
      if (key === 'generate.phaseVae') return 'Decoding image…'
      if (key === 'generate.phaseSample') return `Sampling ${values?.step}/${values?.total}`
      if (key === 'generate.phaseSampling') return 'Sampling'
      if (key === 'generate.totalProgress') return `Total ${values?.pct}%`
      return key
    },
  }),
}))

const EMPTY_PROGRESS: GenerateProgress = {
  phase: null,
  batchIdx: null,
  batchTotal: null,
  currentStep: null,
  totalSteps: null,
}

describe('GenerateProgressBar', () => {
  it('does not render while the pipeline is idle', () => {
    const { container } = render(
      <GenerateProgressBar busy={false} progress={EMPTY_PROGRESS} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('uses indeterminate progress while busy but before the first phase event', () => {
    render(<GenerateProgressBar busy progress={EMPTY_PROGRESS} />)

    const progress = screen.getByRole('progressbar', { name: 'Pipeline progress' })
    expect(progress).not.toHaveAttribute('aria-valuenow')
    expect(progress).toHaveAttribute('aria-valuetext', 'Preparing…')
    expect(screen.queryByText(/Total \d+%/)).not.toBeInTheDocument()
  })

  it('uses the stable sampling announcement for legacy step events without a phase', () => {
    render(
      <GenerateProgressBar
        busy
        progress={{
          phase: null,
          batchIdx: null,
          batchTotal: null,
          currentStep: 2,
          totalSteps: 10,
        }}
      />,
    )

    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow')
    expect(screen.getByRole('status')).toHaveTextContent('Sampling')
    expect(screen.getByRole('status')).not.toHaveTextContent('2/10')
  })

  it('maps sampling steps into determinate pipeline progress', () => {
    render(
      <GenerateProgressBar
        busy
        progress={{
          phase: 'sample',
          batchIdx: 0,
          batchTotal: 1,
          currentStep: 10,
          totalSteps: 20,
        }}
      />,
    )

    const progress = screen.getByRole('progressbar', { name: 'Pipeline progress' })
    expect(progress).toHaveAttribute('aria-valuenow', '56')
    expect(progress).toHaveAttribute('aria-valuetext', 'Sampling 10/20 · Total 56%')
    expect(screen.getByRole('status')).toHaveTextContent('Sampling')
    expect(screen.getByText('Total 56%')).toBeInTheDocument()
  })

  it('folds the current phase into multi-image batch progress', () => {
    render(
      <GenerateProgressBar
        busy
        progress={{
          phase: 'sample',
          batchIdx: 1,
          batchTotal: 4,
          currentStep: 10,
          totalSteps: 20,
        }}
      />,
    )

    expect(screen.getByRole('progressbar'))
      .toHaveAttribute('aria-valuenow', '39')
    expect(screen.getByRole('progressbar'))
      .toHaveAttribute('aria-valuetext', '2/4 · Sampling 10/20 · Total 39%')
  })
})
