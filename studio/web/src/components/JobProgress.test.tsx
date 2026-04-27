import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { Job } from '../api/client'
import JobProgress from './JobProgress'

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: 7,
    project_id: 1,
    version_id: null,
    kind: 'download',
    params: '{}',
    status: 'running',
    started_at: 1700000000,
    finished_at: null,
    pid: 1234,
    log_path: '/tmp/7.log',
    error_msg: null,
    ...overrides,
  }
}

describe('JobProgress (PP2)', () => {
  it('renders status badge + cancel button when running', async () => {
    const onCancel = vi.fn()
    const user = userEvent.setup()
    render(
      <JobProgress
        job={makeJob({ status: 'running' })}
        logs={['line a', 'line b']}
        onCancel={onCancel}
      />
    )
    expect(screen.getByText('running')).toBeInTheDocument()
    expect(screen.getByText(/job #7/)).toBeInTheDocument()
    expect(screen.getByText(/line a\s+line b/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '取消' }))
    expect(onCancel).toHaveBeenCalledOnce()
  })

  it('hides cancel button on terminal states', () => {
    render(
      <JobProgress
        job={makeJob({ status: 'done', finished_at: 1700000010 })}
        logs={['done line']}
        onCancel={() => {}}
      />
    )
    expect(screen.queryByRole('button', { name: '取消' })).toBeNull()
    expect(screen.getByText('done')).toBeInTheDocument()
  })

  it('shows "等待日志..." when no logs yet', () => {
    render(
      <JobProgress job={makeJob()} logs={[]} onCancel={() => {}} />
    )
    expect(screen.getByText(/等待日志/)).toBeInTheDocument()
  })
})
