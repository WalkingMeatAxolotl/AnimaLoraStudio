import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { UploadProgressState } from '../lib/useUploadProgress'
import UploadProgressBar from './UploadProgressBar'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => ({
      'upload.progressLabel': 'Upload progress',
      'upload.processing': 'Server processing…',
      'upload.complete': 'Upload complete',
      'upload.failed': 'Failed',
      'upload.etaPrefix': 'ETA',
    })[key] ?? key,
  }),
}))

const makeState = (overrides: Partial<UploadProgressState>): UploadProgressState => ({
  phase: 'uploading',
  loaded: 0,
  total: 100,
  speedBps: 0,
  etaSec: null,
  error: null,
  ...overrides,
})

describe('UploadProgressBar', () => {
  it('renders upload bytes as determinate progress without a live region per tick', () => {
    render(
      <UploadProgressBar
        state={makeState({ loaded: 50, total: 100, speedBps: 25, etaSec: 2 })}
      />,
    )

    const progress = screen.getByRole('progressbar', { name: 'Upload progress' })
    expect(progress).toHaveAttribute('aria-valuenow', '50')
    expect(progress).toHaveAttribute('aria-valuetext', expect.stringContaining('50%'))
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.getByText(/ETA 2s/)).toBeInTheDocument()
  })

  it('keeps an unknown-length upload indeterminate without fabricating 0%', () => {
    render(
      <UploadProgressBar state={makeState({ loaded: 512, total: 0 })} />,
    )

    const progress = screen.getByRole('progressbar', { name: 'Upload progress' })
    expect(progress).not.toHaveAttribute('aria-valuenow')
    expect(progress).toHaveAttribute('data-state', 'indeterminate')
    expect(screen.getByText('512 B')).toBeInTheDocument()
    expect(screen.queryByText('0%')).not.toBeInTheDocument()
  })

  it('switches server processing to indeterminate progress and announces the phase once', () => {
    render(
      <UploadProgressBar state={makeState({ phase: 'processing', loaded: 100, total: 100 })} />,
    )

    const progress = screen.getByRole('progressbar', { name: 'Server processing…' })
    expect(progress).not.toHaveAttribute('aria-valuenow')
    expect(progress).toHaveAttribute('data-state', 'indeterminate')
    expect(screen.getByRole('status')).toHaveTextContent('Server processing…')
  })

  it('uses an alert instead of presenting a failed operation as active progress', () => {
    render(
      <UploadProgressBar state={makeState({ phase: 'error', error: 'network down' })} />,
    )

    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Failed: network down')
  })

  it('marks the completed state with success progress and a phase announcement', () => {
    render(
      <UploadProgressBar state={makeState({ phase: 'done', loaded: 100, total: 100 })} />,
    )

    expect(screen.getByRole('progressbar', { name: 'Upload complete' }))
      .toHaveAttribute('aria-valuenow', '100')
    expect(screen.getByRole('progressbar')).toHaveClass('ui-progress-success')
    expect(screen.getByRole('status')).toHaveTextContent('Upload complete')
  })
})
