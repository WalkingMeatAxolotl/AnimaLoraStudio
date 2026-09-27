import { useState, type ReactNode } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import i18n from '../i18n'
import { SettingsDrawerProvider, useSettingsDrawer } from '../lib/SettingsDrawer'
import { DialogProvider } from './Dialog'
import Drawer from './Drawer'
import Modal from './Modal'
import Topbar from './Topbar'

vi.mock('../lib/Announcements', () => ({
  useAnnouncements: () => ({ unreadCount: 0, updateInfo: null, open: false, openCenter: vi.fn() }),
}))
vi.mock('../lib/useEventStream', () => ({ useEventStream: () => undefined }))
vi.mock('../lib/useMonitorProgress', () => ({ useMonitorProgress: () => ({ state: null }) }))
vi.mock('./SystemStats', () => ({ default: () => <span>Test stats</span> }))

type DialogRole = 'dialog' | 'alertdialog'
type Shortcut = 'ctrl' | 'meta'

function TaskSurfaces({ role }: { role: DialogRole }) {
  const [modalOpen, setModalOpen] = useState(false)
  const drawer = useSettingsDrawer()
  return (
    <>
      <button type="button" onClick={() => setModalOpen(true)}>Open task</button>
      <button type="button" onClick={() => drawer.open()}>Open settings</button>
      {modalOpen && (
        <Modal title="Protected task" role={role} onClose={() => setModalOpen(false)}>
          <input aria-label="Task field" />
          <button type="button">Task action</button>
        </Modal>
      )}
      <Drawer open={drawer.isOpen} title="Settings" onClose={() => drawer.close()} testId="search-test-drawer">
        <input aria-label="Settings field" />
        <button type="button" onClick={() => setModalOpen(true)}>Open nested task</button>
      </Drawer>
    </>
  )
}

function renderWorkspace(role: DialogRole = 'dialog', extra?: ReactNode) {
  const root = document.createElement('div')
  root.id = 'root'
  root.inert = false
  document.body.append(root)
  return render(
    <MemoryRouter future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
      <DialogProvider>
        <SettingsDrawerProvider>
          <Topbar />
          <TaskSurfaces role={role} />
          {extra}
        </SettingsDrawerProvider>
      </DialogProvider>
    </MemoryRouter>,
    { container: root },
  )
}

function shortcut(target: Element | Window, modifier: Shortcut) {
  fireEvent.keyDown(target, { key: 'k', ctrlKey: modifier === 'ctrl', metaKey: modifier === 'meta' })
}

beforeEach(async () => {
  await i18n.changeLanguage('zh')
  vi.spyOn(api, 'listQueue').mockResolvedValue([])
  vi.spyOn(api, 'listProjects').mockResolvedValue([])
  vi.spyOn(api, 'listPresets').mockResolvedValue([])
})
afterEach(() => {
  vi.restoreAllMocks()
  document.getElementById('root')?.remove()
})

describe('Topbar command entry with real overlay components', () => {
  it.each([
    ['dialog', 'ctrl'], ['dialog', 'meta'], ['alertdialog', 'ctrl'], ['alertdialog', 'meta'],
  ] as const)('does not open search over an active %s from %s+K or the search button', async (role, modifier) => {
    const user = userEvent.setup()
    renderWorkspace(role)
    const opener = screen.getByRole('button', { name: 'Open task' })
    await user.click(opener)
    const field = screen.getByRole('textbox', { name: 'Task field' })
    await waitFor(() => expect(field).toHaveFocus())

    shortcut(field, modifier)
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(field).toHaveFocus()
    const search = screen.getByRole('button', { name: '搜索' })
    fireEvent.click(search)
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(search).toHaveAttribute('aria-expanded', 'false')
    expect(field).toHaveFocus()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole(role, { name: 'Protected task' })).not.toBeInTheDocument()
    expect(opener).toHaveFocus()
    shortcut(opener, modifier)
    await waitFor(() => expect(screen.getByRole('combobox')).toHaveFocus())
  })

  it.each(['ctrl', 'meta'] as const)('blocks %s+K for an open Drawer but ignores its closed keep-alive shell', async (modifier) => {
    const user = userEvent.setup()
    renderWorkspace()
    const opener = screen.getByRole('button', { name: 'Open settings' })
    await user.click(opener)
    const drawer = screen.getByRole('dialog', { name: 'Settings' })
    await waitFor(() => expect(drawer).toHaveFocus())
    shortcut(drawer, modifier)
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(drawer).toHaveFocus()
    expect(document.getElementById('root')?.inert).toBe(true)

    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.getByTestId('search-test-drawer')).toHaveAttribute('data-state', 'closed'))
    expect(opener).toHaveFocus()
    expect(document.getElementById('root')?.inert).toBe(false)
    shortcut(opener, modifier)
    const input = await screen.findByRole('combobox')
    await waitFor(() => expect(input).toHaveFocus())
    shortcut(input, modifier)
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '搜索' })).toHaveFocus()
  })

  it('keeps the underlying Drawer open when Escape closes its nested task, without adding a palette', async () => {
    const user = userEvent.setup()
    renderWorkspace()
    await user.click(screen.getByRole('button', { name: 'Open settings' }))
    const opener = screen.getByRole('button', { name: 'Open nested task' })
    await user.click(opener)
    const field = screen.getByRole('textbox', { name: 'Task field' })
    await waitFor(() => expect(field).toHaveFocus())
    shortcut(field, 'ctrl')
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: 'Protected task' })).not.toBeInTheDocument()
    expect(screen.getByRole('dialog', { name: 'Settings' })).toBeInTheDocument()
    expect(document.getElementById('root')?.inert).toBe(true)
    expect(opener).toHaveFocus()
  })

  it.each(['hidden', 'aria-hidden', 'inert'] as const)('ignores dialogs in a %s cached subtree', async (attribute) => {
    const attributes = attribute === 'hidden' ? { hidden: true }
      : attribute === 'aria-hidden' ? { 'aria-hidden': true as const } : { inert: '' }
    renderWorkspace('dialog', <div {...attributes}><div role="dialog" aria-modal="true">Cached task</div></div>)
    shortcut(window, 'ctrl')
    const input = await screen.findByRole('combobox')
    await waitFor(() => expect(input).toHaveFocus())
    shortcut(input, 'ctrl')
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
  })

  it('does not mistake non-modal panels or live feedback for blocking task dialogs', async () => {
    renderWorkspace('dialog', <><div role="dialog" aria-modal="false">Inline panel</div><div role="alert">Feedback</div></>)
    fireEvent.click(screen.getByRole('button', { name: '搜索' }))
    await waitFor(() => expect(screen.getByRole('combobox')).toHaveFocus())
  })

  it('still opens Settings from a command and protects that task until it closes', async () => {
    const user = userEvent.setup()
    renderWorkspace()
    await user.click(screen.getByRole('button', { name: '搜索' }))
    const input = screen.getByRole('combobox')
    await waitFor(() => expect(input).toHaveFocus())
    fireEvent.keyDown(input, { key: 'End' })
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() => expect(screen.getByRole('dialog', { name: 'Settings' })).toHaveFocus())
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    shortcut(screen.getByRole('dialog', { name: 'Settings' }), 'ctrl')
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
  })
})
