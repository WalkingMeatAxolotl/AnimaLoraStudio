import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { DialogProvider } from './Dialog'
import SettingsDrawer from './SettingsDrawer'
import { SettingsDrawerProvider, useSettingsDrawer } from '../lib/SettingsDrawer'

vi.mock('../pages/tools/Settings', () => ({
  default: () => <div data-testid="settings-content">Settings content</div>,
}))

function endAnimation(element: Element, animationName: string) {
  const event = new Event('animationend', { bubbles: true })
  Object.defineProperty(event, 'animationName', { value: animationName })
  fireEvent(element, event)
}

function Harness() {
  const drawer = useSettingsDrawer()
  return (
    <>
      <button type="button" onClick={() => drawer.open()}>Open settings</button>
      <output data-testid="drawer-ready">{String(drawer.isReady)}</output>
      <SettingsDrawer />
    </>
  )
}

function renderHarness() {
  return render(
    <DialogProvider>
      <SettingsDrawerProvider>
        <Harness />
      </SettingsDrawerProvider>
    </DialogProvider>,
  )
}

describe('SettingsDrawer', () => {
  it('mounts static content with the shell and keeps one instance across opens', async () => {
    const user = userEvent.setup()
    renderHarness()

    expect(screen.getByTestId('settings-drawer')).toHaveAttribute('data-state', 'closed')
    expect(screen.queryByTestId('settings-content')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Open settings' }))
    const root = screen.getByTestId('settings-drawer')
    const dialog = screen.getByRole('dialog', { name: /设置|Settings/ })
    const content = await screen.findByTestId('settings-content')
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'opening'))
    expect(content).toBeVisible()
    expect(screen.queryByText(/加载中|Loading/)).not.toBeInTheDocument()
    expect(screen.getByTestId('drawer-ready')).toHaveTextContent('false')

    endAnimation(dialog, 'drawer-panel-in')
    expect(root).toHaveAttribute('data-state', 'open')
    expect(content).toBeVisible()
    expect(screen.getByTestId('drawer-ready')).toHaveTextContent('true')

    fireEvent.keyDown(document, { key: 'Escape' })
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'closing'))
    expect(content).toBeVisible()
    expect(screen.queryByText(/加载中|Loading/)).not.toBeInTheDocument()
    expect(screen.getByTestId('drawer-ready')).toHaveTextContent('false')
    endAnimation(dialog, 'drawer-panel-out')
    expect(root).toHaveAttribute('data-state', 'closed')

    await user.click(screen.getByRole('button', { name: 'Open settings' }))
    await waitFor(() => expect(root).toHaveAttribute('data-state', 'opening'))
    expect(screen.getByTestId('settings-content')).toBe(content)
    expect(content).toBeVisible()
    expect(screen.queryByText(/加载中|Loading/)).not.toBeInTheDocument()
  })
})
