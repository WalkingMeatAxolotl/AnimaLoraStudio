import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import i18n from '../i18n'
import CommandPalette from './CommandPalette'

const mocks = vi.hoisted(() => ({
  close: vi.fn(),
  openSettings: vi.fn(),
}))

vi.mock('../lib/SettingsDrawer', () => ({
  useSettingsDrawer: () => ({ open: mocks.openSettings }),
}))

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname}</output>
}

function renderPalette(open = true, anchorEl?: HTMLElement) {
  return render(
    <MemoryRouter
      initialEntries={['/']}
      future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
    >
      <CommandPalette open={open} onClose={mocks.close} anchorEl={anchorEl} />
      <LocationProbe />
    </MemoryRouter>,
  )
}

describe('CommandPalette', () => {
  beforeEach(async () => {
    mocks.close.mockReset()
    mocks.openSettings.mockReset()
    await i18n.changeLanguage('zh')
    vi.spyOn(api, 'listProjects').mockResolvedValue([])
    vi.spyOn(api, 'listPresets').mockResolvedValue([])
  })

  afterEach(() => {
    document.querySelectorAll('[data-command-palette-test-anchor]').forEach((element) => element.remove())
  })

  it('exposes a dialog, combobox, listbox, and active option relationship', async () => {
    renderPalette()

    expect(screen.getByRole('dialog', { name: '命令面板' })).toHaveAttribute('aria-modal', 'true')
    const input = screen.getByRole('combobox', { name: '搜索命令' })
    const listbox = screen.getByRole('listbox', { name: '命令搜索结果' })
    expect(input).toHaveAttribute('aria-controls', listbox.id)
    expect(input).toHaveAttribute('aria-haspopup', 'listbox')
    expect(input).toHaveAttribute('aria-expanded', 'true')
    await waitFor(() => expect(input).toHaveFocus())

    const firstOption = screen.getByRole('option', { name: /项目列表/ })
    expect(firstOption).toHaveAttribute('aria-selected', 'true')
    expect(input).toHaveAttribute('aria-activedescendant', firstOption.id)
  })

  it('supports wrap-around arrows and Home/End while keeping focus in the combobox', async () => {
    renderPalette()
    const input = screen.getByRole('combobox', { name: '搜索命令' })
    await waitFor(() => expect(input).toHaveFocus())

    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(screen.getByRole('option', { name: /设置/ })).toHaveAttribute('aria-selected', 'true')

    fireEvent.keyDown(input, { key: 'Home' })
    expect(screen.getByRole('option', { name: /项目列表/ })).toHaveAttribute('aria-selected', 'true')

    fireEvent.keyDown(input, { key: 'End' })
    expect(screen.getByRole('option', { name: /设置/ })).toHaveAttribute('aria-selected', 'true')
    expect(input).toHaveFocus()
  })

  it('recaptures unexpected panel focus so command keys stay on the query', async () => {
    renderPalette()
    const input = screen.getByRole('combobox', { name: '搜索命令' })
    const dialog = screen.getByRole('dialog', { name: '命令面板' })
    await waitFor(() => expect(input).toHaveFocus())

    dialog.focus()
    fireEvent.keyDown(document, { key: 'Tab', shiftKey: true })
    expect(input).toHaveFocus()

    dialog.focus()
    await waitFor(() => expect(input).toHaveFocus())
  })

  it('navigates the selected command with ArrowDown and Enter', async () => {
    renderPalette()
    const input = screen.getByRole('combobox', { name: '搜索命令' })
    await waitFor(() => expect(input).toHaveFocus())

    fireEvent.keyDown(input, { key: 'ArrowDown' })
    expect(screen.getByRole('option', { name: /队列/ })).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(screen.getByTestId('location')).toHaveTextContent('/queue')
    expect(mocks.close).toHaveBeenCalledTimes(1)
  })

  it('reports an empty search result without leaving a stale active descendant', async () => {
    renderPalette()
    const input = screen.getByRole('combobox', { name: '搜索命令' })
    await waitFor(() => expect(input).toHaveFocus())

    fireEvent.change(input, { target: { value: 'no-such-command' } })

    expect(screen.getByText('无匹配结果')).toHaveAttribute('role', 'status')
    expect(input).not.toHaveAttribute('aria-activedescendant')
  })

  it('closes on Escape or backdrop click and restores the trigger focus after closing', async () => {
    const anchor = document.createElement('button')
    anchor.dataset.commandPaletteTestAnchor = 'true'
    document.body.appendChild(anchor)
    anchor.focus()

    const view = renderPalette(true, anchor)
    await waitFor(() => expect(screen.getByRole('combobox', { name: '搜索命令' })).toHaveFocus())

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(mocks.close).toHaveBeenCalledTimes(1)
    view.rerender(
      <MemoryRouter
        initialEntries={['/']}
        future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
      >
        <CommandPalette open={false} onClose={mocks.close} anchorEl={anchor} />
        <LocationProbe />
      </MemoryRouter>,
    )
    expect(anchor).toHaveFocus()

    mocks.close.mockReset()
    view.rerender(
      <MemoryRouter
        initialEntries={['/']}
        future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
      >
        <CommandPalette open onClose={mocks.close} anchorEl={anchor} />
        <LocationProbe />
      </MemoryRouter>,
    )
    await waitFor(() => expect(screen.getByRole('combobox', { name: '搜索命令' })).toHaveFocus())
    await screen.findByRole('option', { name: /项目列表/ })
    fireEvent.click(screen.getByTestId('command-palette-backdrop'))
    expect(mocks.close).toHaveBeenCalledTimes(1)
  })
})
