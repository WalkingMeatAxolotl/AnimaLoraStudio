import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type Task } from '../api/client'
import i18n from '../i18n'
import Topbar from './Topbar'

const mocks = vi.hoisted(() => ({
  openCenter: vi.fn(),
}))

vi.mock('../lib/Announcements', () => ({
  useAnnouncements: () => ({
    unreadCount: 2,
    updateInfo: { has_update: false },
    open: false,
    openCenter: mocks.openCenter,
  }),
}))

vi.mock('../lib/useEventStream', () => ({
  useEventStream: () => undefined,
}))

vi.mock('../lib/useMonitorProgress', () => ({
  useMonitorProgress: () => ({ state: null }),
}))

vi.mock('./SystemStats', () => ({
  default: () => <div data-testid="system-stats" />,
}))

vi.mock('./CommandPalette', () => ({
  default: ({ open }: { open: boolean }) => open
    ? <div role="dialog" aria-label="command palette" />
    : null,
}))

const RUNNING_TASK = {
  id: 41,
  name: 'portrait-v2',
  config_name: 'portrait-v2',
  status: 'running',
  priority: 0,
  created_at: 1,
  started_at: null,
  finished_at: null,
  pid: 123,
  exit_code: null,
  output_dir: null,
  error_msg: null,
  project_id: null,
  version_id: null,
} as Task

function renderTopbar() {
  return render(
    <MemoryRouter
      initialEntries={['/queue']}
      future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
    >
      <Topbar />
    </MemoryRouter>,
  )
}

describe('Topbar', () => {
  beforeEach(async () => {
    mocks.openCenter.mockReset()
    await i18n.changeLanguage('zh')
    vi.spyOn(api, 'listQueue').mockImplementation(async (status) => (
      status === 'pending' ? [RUNNING_TASK, RUNNING_TASK] : []
    ))
  })

  it('uses navigation links for queue status and shared buttons for global actions', async () => {
    renderTopbar()

    const pending = await screen.findByRole('link', { name: '2 排队中' })
    expect(pending).toHaveAttribute('href', '/queue')
    expect(pending).toHaveClass('btn', 'btn-secondary', 'btn-sm')

    const announcements = screen.getByRole('button', { name: '2 条未读公告' })
    expect(announcements).toHaveClass('btn', 'btn-secondary', 'btn-sm', 'btn-icon')
    expect(announcements).toHaveAttribute('aria-haspopup', 'dialog')
    expect(announcements).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(announcements)
    expect(mocks.openCenter).toHaveBeenCalledTimes(1)

    const search = screen.getByRole('button', { name: '搜索' })
    expect(search).toHaveClass('btn', 'btn-secondary', 'btn-sm', 'btn-icon')
    expect(search).toHaveAttribute('aria-haspopup', 'dialog')
    expect(search).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(search)
    expect(search).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('dialog', { name: 'command palette' })).toBeInTheDocument()
  })

  it('opens the command palette from the documented keyboard shortcut', async () => {
    renderTopbar()
    await waitFor(() => expect(api.listQueue).toHaveBeenCalled())

    fireEvent.keyDown(window, { key: 'k', ctrlKey: true })

    expect(screen.getByRole('button', { name: '搜索' })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('dialog', { name: 'command palette' })).toBeInTheDocument()
  })

  it('links a running task directly to its queue detail with an accessible status name', async () => {
    vi.mocked(api.listQueue).mockImplementation(async (status) => (
      status === 'running' ? [RUNNING_TASK] : []
    ))
    renderTopbar()

    const taskLink = await screen.findByRole('link', {
      name: '任务 #41：训练中 · portrait-v2',
    })
    expect(taskLink).toHaveAttribute('href', '/queue/41')
    expect(taskLink).toHaveClass('btn', 'btn-secondary', 'btn-sm')
    expect(taskLink.querySelector('.dot-running')).toHaveAttribute('aria-hidden', 'true')
  })
})
