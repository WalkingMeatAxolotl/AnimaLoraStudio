import { createContext, useContext, useState, type ReactNode } from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { LLMPreset } from '../api/client'
import i18n from '../i18n'
import { MASK } from '../pages/tools/settings/constants'
import Drawer from './Drawer'
import { DialogProvider } from './Dialog'
import { ToastProvider } from './Toast'
import LLMPresetEditorModal from './LLMPresetEditorModal'

const api = vi.hoisted(() => ({
  patchLLMPreset: vi.fn(),
  deleteLLMPreset: vi.fn(),
  resetLLMPreset: vi.fn(),
  setDefaultLLMPreset: vi.fn(),
  duplicateLLMPreset: vi.fn(),
  llmPresetExportUrl: vi.fn(),
  listCredentials: vi.fn(),
  createCredential: vi.fn(),
  replaceCredentialSecret: vi.fn(),
  refreshLLMModels: vi.fn(),
  testLLMConnection: vi.fn(),
  updateSecrets: vi.fn(),
}))
vi.mock('../api/client', async (importOriginal) => ({
  ...await importOriginal<typeof import('../api/client')>(),
  api,
}))

// Only the SettingsData projection consumed by this editor is mocked. The actual
// editor fields, message editor, Modal, Drawer, Dialog and Toast stay in the tree.
interface EditorSettings {
  secrets: { llm_tagger: { presets: LLMPreset[]; current_preset: string } }
  reloadSecrets: () => Promise<void>
  runSave: (operation: () => Promise<void>) => Promise<void>
}
const SettingsFixture = createContext<EditorSettings | null>(null)
vi.mock('../lib/SettingsData', () => ({
  useSettingsData: () => useContext(SettingsFixture),
}))

function preset(overrides: Partial<LLMPreset> = {}): LLMPreset {
  return {
    id: 'custom', label: 'Custom fixture', builtin: false, etag: 'etag-1',
    base_url: 'https://fixture.invalid/v1', api_key: '', credential_ref: '',
    model: 'fixture-model', model_ids: [], endpoint: 'chat_completions',
    messages: [{ type: 'text', role: 'system', content: 'Fixture prompt' }],
    output_format: 'json', assist_tagger: '', temperature: 0.2, max_tokens: 700,
    max_side: 1280, jpeg_quality: 85, max_image_mb: 5, timeout: 60,
    max_retries: 3, concurrency: 1, requests_per_second: 0, max_requests_per_minute: 0,
    ...overrides,
  }
}

let serverPresets: LLMPreset[]
let serverDefault: string
let revision: number
const trace: string[] = []
const closed = vi.fn()

function DataFixture({ children }: { children: ReactNode }) {
  const [llm, setLlm] = useState(() => ({ presets: structuredClone(serverPresets), current_preset: serverDefault }))
  return (
    <SettingsFixture.Provider value={{
      secrets: { llm_tagger: llm },
      reloadSecrets: async () => {
        setLlm({ presets: structuredClone(serverPresets), current_preset: serverDefault })
      },
      runSave: async (operation) => { await operation() },
    }}>
      <ToastProvider><DialogProvider>{children}</DialogProvider></ToastProvider>
    </SettingsFixture.Provider>
  )
}

function Harness({ inDrawer = false, removeMissingOpener = false }: { inDrawer?: boolean; removeMissingOpener?: boolean }) {
  const [editorOpen, setEditorOpen] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(true)
  const settings = useContext(SettingsFixture)
  const entry = !removeMissingOpener || settings?.secrets.llm_tagger.presets.some((item) => item.id === 'custom')
    ? <button type="button" onClick={() => setEditorOpen(true)}>Edit fixture preset</button>
    : null
  return (
    <>
      <button type="button">Background action</button>
      {inDrawer ? (
        <Drawer open={drawerOpen} onClose={() => setDrawerOpen(false)} title="Settings fixture">
          {entry}
        </Drawer>
      ) : entry}
      {editorOpen && (
        <LLMPresetEditorModal presetId="custom" onClose={() => { closed(); setEditorOpen(false) }} />
      )}
    </>
  )
}

const titleMatches = (name: string) => name.startsWith(i18n.t('llmPreset.title'))
function editor() { return screen.getByRole('dialog', { name: titleMatches }) }
function footerAction(name: string) {
  const actions = within(editor()).getAllByRole('button', { name })
  return actions[actions.length - 1]
}
async function openEditor(inDrawer = false, removeMissingOpener = false) {
  const user = userEvent.setup()
  const root = document.createElement('div')
  root.id = 'root'
  document.body.appendChild(root)
  render(<DataFixture><Harness inDrawer={inDrawer} removeMissingOpener={removeMissingOpener} /></DataFixture>, { container: root })
  const opener = screen.getByRole('button', { name: 'Edit fixture preset' })
  await user.click(opener)
  return { user, opener, dialog: editor() }
}

beforeEach(() => {
  vi.resetAllMocks()
  serverPresets = [preset(), preset({ id: 'backup', label: 'Backup fixture', etag: 'backup-1' })]
  serverDefault = 'custom'
  revision = 1
  trace.length = 0
  api.patchLLMPreset.mockImplementation(async (id: string, patch: Partial<LLMPreset>, etag: string) => {
    trace.push(`patch:${etag}`)
    const index = serverPresets.findIndex((item) => item.id === id)
    if (index < 0) throw new Error('Unknown fixture preset')
    const updated = { ...serverPresets[index], ...patch, etag: `etag-${++revision}` }
    serverPresets[index] = updated
    return updated
  })
  api.setDefaultLLMPreset.mockImplementation(async (id: string) => {
    trace.push(`default:${id}`)
    serverDefault = id
  })
  api.deleteLLMPreset.mockImplementation(async (id: string, etag: string) => {
    trace.push(`delete:${etag}`)
    serverPresets = serverPresets.filter((item) => item.id !== id)
  })
  api.resetLLMPreset.mockResolvedValue(undefined)
  api.duplicateLLMPreset.mockImplementation(async (_id: string, label: string) => {
    const copy = preset({ id: 'copy', label, etag: 'copy-1' })
    serverPresets.push(copy)
    return copy
  })
  api.llmPresetExportUrl.mockReturnValue('https://fixture.invalid/export.json')
  api.listCredentials.mockResolvedValue([{ id: 'fixture-credential', etag: 'credential-1' }])
  api.createCredential.mockResolvedValue({ id: 'fixture-credential', etag: 'credential-1' })
  api.replaceCredentialSecret.mockResolvedValue(undefined)
  api.refreshLLMModels.mockResolvedValue({ items: ['fixture-model'] })
  api.testLLMConnection.mockResolvedValue({ ok: true, elapsed_ms: 1, status_code: 200 })
})
afterEach(() => {
  cleanup()
  document.getElementById('root')?.remove()
  document.body.style.overflow = ''
  vi.restoreAllMocks()
})

describe('LLMPresetEditorModal keyboard lifecycle', () => {
  it('uses a named viewport-bounded Modal, enters focus and locks background scroll', async () => {
    const { dialog } = await openEditor()
    await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true))
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(dialog).toHaveClass('w-[80vw]', 'max-w-[1440px]')
    expect(dialog.parentElement?.parentElement).toBe(document.body)
    expect(document.body.style.overflow).toBe('hidden')
  })

  it('cycles Tab at the header/footer boundaries and returns focus on Escape', async () => {
    const { user, opener, dialog } = await openEditor()
    const close = within(dialog).getByRole('button', { name: i18n.t('common.close') })
    const done = footerAction(i18n.t('llmPreset.done'))
    close.focus()
    await user.tab({ shift: true })
    expect(done).toHaveFocus()
    await user.tab()
    expect(close).toHaveFocus()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: titleMatches })).not.toBeInTheDocument()
    expect(opener).toHaveFocus()
    expect(document.body.style.overflow).toBe('')
  })

  it('closes only the editor above a real Drawer, leaving its opener and inert protection', async () => {
    const { user, opener, dialog } = await openEditor(true)
    const settings = screen.getByRole('dialog', { name: 'Settings fixture' })
    await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true))
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog', { name: titleMatches })).not.toBeInTheDocument()
    expect(settings).toBeInTheDocument()
    expect(opener).toHaveFocus()
    // jsdom does not reflect the inert property to its HTML attribute.
    expect(document.getElementById('root')?.inert).toBe(true)
    expect(settings.parentElement?.getAttribute('data-state')).toMatch(/^(open|opening)$/)
  })

  it('returns to the Drawer when deletion removes its original edit action', async () => {
    const { user, opener } = await openEditor(true, true)
    const settings = screen.getByRole('dialog', { name: 'Settings fixture' })
    await user.click(footerAction(i18n.t('common.delete')))
    await user.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: i18n.t('common.confirm') }))
    await waitFor(() => expect(closed).toHaveBeenCalledTimes(1))
    expect(opener).not.toBeInTheDocument()
    expect(settings).toHaveFocus()
    expect(document.getElementById('root')?.inert).toBe(true)
  })

  it.each(['Escape', 'Done', 'backdrop'] as const)('commits the active text buffer once before %s closes', async (method) => {
    const { user, dialog } = await openEditor()
    const input = within(dialog).getByDisplayValue('Custom fixture')
    await user.clear(input)
    await user.type(input, 'Renamed fixture')
    expect(api.patchLLMPreset).not.toHaveBeenCalled()
    if (method === 'Escape') await user.keyboard('{Escape}')
    else if (method === 'Done') await user.click(footerAction(i18n.t('llmPreset.done')))
    else await user.click(screen.getByTestId('llm-preset-editor-modal'))
    await waitFor(() => expect(closed).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(api.patchLLMPreset).toHaveBeenCalledTimes(1))
    expect(api.patchLLMPreset).toHaveBeenCalledWith('custom', { label: 'Renamed fixture' }, 'etag-1')
    expect(api.updateSecrets).not.toHaveBeenCalled()
  })

  it('keeps Enter as a message newline and commits the message on Escape', async () => {
    const { user, dialog } = await openEditor()
    const message = within(dialog).getByDisplayValue('Fixture prompt')
    await user.click(message)
    await user.keyboard('{End}{Enter}more')
    expect(message).toHaveValue('Fixture prompt\nmore')
    expect(closed).not.toHaveBeenCalled()
    expect(api.patchLLMPreset).not.toHaveBeenCalled()
    await user.keyboard('{Escape}')
    await waitFor(() => expect(api.patchLLMPreset).toHaveBeenCalledWith('custom', {
      messages: [{ type: 'text', role: 'system', content: 'Fixture prompt\nmore' }],
    }, 'etag-1'))
  })

  it('lets the sortable cancel its keyboard drag before Escape exits the editor', async () => {
    const { user, dialog } = await openEditor()
    const handle = within(dialog).getByRole('button', { name: i18n.t('llmMessages.dragHandle') })
    handle.focus()
    await user.keyboard(' ')
    await waitFor(() => expect(handle).toHaveAttribute('aria-pressed', 'true'))
    await user.keyboard('{Escape}')
    await waitFor(() => expect(handle).not.toHaveAttribute('aria-pressed', 'true'))
    expect(editor()).toBe(dialog)
    expect(api.patchLLMPreset).not.toHaveBeenCalled()
    await user.keyboard('{Escape}')
    expect(closed).toHaveBeenCalledTimes(1)
  })

  it('keeps numeric clamping and the write-only credential route when Escape exits', async () => {
    serverPresets[0] = preset({ api_key: MASK, credential_ref: 'fixture-credential' })
    const { user, dialog } = await openEditor()
    const number = within(dialog).getByDisplayValue('0.2')
    await user.clear(number)
    await user.type(number, '9')
    await user.tab()
    await waitFor(() => expect(api.patchLLMPreset).toHaveBeenCalledWith('custom', { temperature: 2 }, 'etag-1'))
    const key = dialog.querySelector<HTMLInputElement>('input[type="password"]')!
    await user.type(key, 'not-a-real-fixture-key')
    await user.keyboard('{Escape}')
    await waitFor(() => expect(api.replaceCredentialSecret).toHaveBeenCalledWith(
      'fixture-credential', 'not-a-real-fixture-key', 'credential-1',
    ))
    expect(api.createCredential).not.toHaveBeenCalled()
    expect(api.patchLLMPreset.mock.calls.every(([, patch]) => !('api_key' in patch))).toBe(true)
    expect(api.updateSecrets).not.toHaveBeenCalled()
  })

  it.each(['cancel', 'Escape'] as const)('uses one confirmation surface and preserves input nodes on %s', async (method) => {
    const { user, dialog } = await openEditor()
    const message = within(dialog).getByDisplayValue('Fixture prompt')
    const remove = footerAction(i18n.t('common.delete'))
    await user.click(remove)
    const confirmation = screen.getByRole('alertdialog')
    expect(confirmation).toBe(dialog)
    expect(screen.queryByRole('dialog', { name: titleMatches })).not.toBeInTheDocument()
    expect(screen.queryByDisplayValue('Fixture prompt')).not.toBeVisible()
    const cancel = within(confirmation).getByRole('button', { name: i18n.t('common.cancel') })
    await waitFor(() => expect(cancel).toHaveFocus())
    if (method === 'Escape') await user.keyboard('{Escape}')
    else await user.click(cancel)
    expect(editor()).toBe(dialog)
    expect(within(dialog).getByDisplayValue('Fixture prompt')).toBe(message)
    await waitFor(() => expect(remove).toHaveFocus())
    expect(api.deleteLLMPreset).not.toHaveBeenCalled()
    expect(api.setDefaultLLMPreset).not.toHaveBeenCalled()
    expect(closed).not.toHaveBeenCalled()
  })

  it('cancels only confirmation on backdrop and leaves Settings behind the editor', async () => {
    const { user } = await openEditor(true)
    await user.click(footerAction(i18n.t('common.delete')))
    expect(screen.getByRole('alertdialog')).toBeInTheDocument()
    await user.click(screen.getByTestId('llm-preset-editor-modal'))
    expect(editor()).toBeInTheDocument()
    expect(screen.getByRole('dialog', { name: 'Settings fixture' })).toBeInTheDocument()
    expect(closed).not.toHaveBeenCalled()
    expect(api.deleteLLMPreset).not.toHaveBeenCalled()
  })

  it('waits for the existing mutation queue and uses its latest ETag before confirmed deletion', async () => {
    let finishPatch!: () => void
    api.patchLLMPreset.mockImplementationOnce((_id: string, patch: Partial<LLMPreset>, etag: string) => {
      trace.push(`patch:${etag}`)
      return new Promise<LLMPreset>((resolve) => {
        finishPatch = () => {
          serverPresets[0] = { ...serverPresets[0], ...patch, etag: 'etag-after-patch' }
          resolve(serverPresets[0])
        }
      })
    })
    const { user, dialog } = await openEditor()
    const input = within(dialog).getByDisplayValue('Custom fixture')
    await user.clear(input)
    await user.type(input, 'Queued fixture')
    await user.click(footerAction(i18n.t('common.delete')))
    const confirmation = screen.getByRole('alertdialog')
    await user.click(within(confirmation).getByRole('button', { name: i18n.t('common.confirm') }))
    expect(api.deleteLLMPreset).not.toHaveBeenCalled()
    await act(async () => { finishPatch() })
    await waitFor(() => expect(api.deleteLLMPreset).toHaveBeenCalledWith('custom', 'etag-after-patch'))
    expect(trace).toEqual(['patch:etag-1', 'default:backup', 'delete:etag-after-patch'])
    await waitFor(() => expect(closed).toHaveBeenCalledTimes(1))
  })

  it('keeps the confirmed request pending without duplicate submission or premature Escape', async () => {
    let finishDelete!: () => void
    api.deleteLLMPreset.mockImplementationOnce(() => new Promise<void>((resolve) => { finishDelete = resolve }))
    const { user, dialog } = await openEditor()
    await user.click(footerAction(i18n.t('common.delete')))
    const confirm = within(screen.getByRole('alertdialog')).getByRole('button', { name: i18n.t('common.confirm') })
    await user.click(confirm)
    await waitFor(() => expect(api.deleteLLMPreset).toHaveBeenCalledTimes(1))
    expect(confirm).toBeDisabled()
    await user.click(confirm)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.getByRole('alertdialog')).toBe(dialog)
    expect(closed).not.toHaveBeenCalled()
    expect(api.deleteLLMPreset).toHaveBeenCalledTimes(1)
    await act(async () => { finishDelete() })
    await waitFor(() => expect(closed).toHaveBeenCalledTimes(1))
  })

  it('keeps a failed destructive operation in the same confirmation with a local retry path', async () => {
    api.deleteLLMPreset.mockRejectedValueOnce(new Error('Synthetic delete failure'))
    const { user, dialog } = await openEditor()
    await user.click(footerAction(i18n.t('common.delete')))
    await user.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: i18n.t('common.confirm') }))
    await waitFor(() => expect(within(screen.getByRole('alertdialog')).getByRole('alert')).toHaveTextContent('Synthetic delete failure'))
    expect(screen.getByRole('alertdialog')).toBe(dialog)
    expect(closed).not.toHaveBeenCalled()
    await user.click(within(dialog).getByRole('button', { name: i18n.t('common.confirm') }))
    await waitFor(() => expect(closed).toHaveBeenCalledTimes(1))
    expect(api.deleteLLMPreset).toHaveBeenCalledTimes(2)
  })

  it('confirms reset in the same surface and keeps builtin and last-preset deletion guards', async () => {
    serverPresets = [preset({ builtin: true })]
    const { user, dialog } = await openEditor()
    expect(within(dialog).queryByRole('button', { name: i18n.t('common.delete') })).not.toBeInTheDocument()
    await user.click(footerAction(i18n.t('llmPreset.resetBuiltin')))
    expect(screen.getByRole('alertdialog')).toBe(dialog)
    await user.click(within(dialog).getByRole('button', { name: i18n.t('common.confirm') }))
    await waitFor(() => expect(api.resetLLMPreset).toHaveBeenCalledWith('custom', 'etag-1'))
    expect(api.deleteLLMPreset).not.toHaveBeenCalled()
    await waitFor(() => expect(closed).toHaveBeenCalledTimes(1))
  })

  it('restores each existing pane scroll offset after canceling confirmation', async () => {
    const { user, dialog } = await openEditor()
    const panes = Array.from(dialog.querySelectorAll<HTMLElement>('[data-llm-editor-scroll]'))
    expect(panes).toHaveLength(2)
    panes[0].scrollTop = 42
    panes[1].scrollTop = 120
    await user.click(footerAction(i18n.t('common.delete')))
    // Model a browser resetting hidden scroll geometry; the retained nodes remain authoritative.
    panes.forEach((pane) => { pane.scrollTop = 0 })
    await user.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: i18n.t('common.cancel') }))
    await waitFor(() => {
      expect(panes[0].scrollTop).toBe(42)
      expect(panes[1].scrollTop).toBe(120)
    })
  })

  it.each(['cancel', 'Escape', 'unmount'] as const)(
    'retains offsets past the first revealed-frame resize and cancels return on %s',
    async (method) => {
      const { user, dialog, opener } = await openEditor()
      const panes = Array.from(dialog.querySelectorAll<HTMLElement>('[data-llm-editor-scroll]'))
      const remove = footerAction(i18n.t('common.delete'))
      const message = within(dialog).getByDisplayValue('Fixture prompt')
      panes[0].scrollTop = 100
      panes[1].scrollTop = 21
      await user.click(remove)
      const cancel = within(screen.getByRole('alertdialog')).getByRole('button', { name: i18n.t('common.cancel') })
      await waitFor(() => expect(cancel).toHaveFocus())

      // The browser delivers the auto-growing textarea's ResizeObserver after
      // the first reveal frame. jsdom has no layout; control that ordering here.
      let frameId = 0
      const frames = new Map<number, FrameRequestCallback>()
      vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation((callback) => {
        frames.set(++frameId, callback)
        return frameId
      })
      vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation((id) => { frames.delete(id) })
      const nextFrame = () => {
        const entry = frames.entries().next().value as [number, FrameRequestCallback] | undefined
        expect(entry).toBeDefined()
        if (!entry) throw new Error('Expected a scheduled return frame')
        frames.delete(entry[0])
        act(() => { entry[1](0) })
      }
      if (method === 'Escape') await user.keyboard('{Escape}')
      else await user.click(cancel)
      nextFrame()
      // Fitting the textarea temporarily shrinks the message pane's scroll range.
      panes[1].scrollTop = 0
      if (method === 'unmount') {
        cleanup()
        expect(frames.size).toBe(0)
        expect(opener).not.toBeInTheDocument()
        expect(panes[1].scrollTop).toBe(0)
      } else {
        nextFrame()
        expect(editor()).toBe(dialog)
        expect(within(dialog).getByDisplayValue('Fixture prompt')).toBe(message)
        expect(panes[0].scrollTop).toBe(100)
        expect(panes[1].scrollTop).toBe(21)
        expect(remove).toHaveFocus()
        expect(frames.size).toBe(0)
        expect(api.patchLLMPreset).not.toHaveBeenCalled()
        expect(api.deleteLLMPreset).not.toHaveBeenCalled()
      }
    },
  )

  it('does not offer deletion of the only remaining custom preset', async () => {
    serverPresets = [preset()]
    const { dialog } = await openEditor()
    expect(within(dialog).queryByRole('button', { name: i18n.t('common.delete') })).not.toBeInTheDocument()
    expect(api.deleteLLMPreset).not.toHaveBeenCalled()
  })

  it('never retargets a queued confirmation to a newly created copy', async () => {
    let finishCopy!: () => void
    api.duplicateLLMPreset.mockImplementationOnce(() => new Promise<LLMPreset>((resolve) => {
      finishCopy = () => {
        const copy = preset({ id: 'copy', label: 'Copy fixture', etag: 'copy-1' })
        serverPresets.push(copy)
        resolve(copy)
      }
    }))
    const { user } = await openEditor()
    await user.click(footerAction(i18n.t('llmPreset.saveAsCopy')))
    await user.click(footerAction(i18n.t('common.delete')))
    await user.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: i18n.t('common.confirm') }))
    expect(api.deleteLLMPreset).not.toHaveBeenCalled()
    await act(async () => { finishCopy() })
    expect(api.deleteLLMPreset.mock.calls.every(([id]) => id === 'custom')).toBe(true)
    expect(serverPresets.some((item) => item.id === 'copy')).toBe(true)
  })

  it('keeps export on the dedicated resource URL without sending preset fields', async () => {
    let downloaded: { href: string; filename: string } | undefined
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      downloaded = { href: this.href, filename: this.download }
    })
    const { user } = await openEditor()
    await user.click(footerAction(i18n.t('llmPreset.export')))
    expect(api.llmPresetExportUrl).toHaveBeenCalledWith('custom')
    expect(downloaded).toEqual({ href: 'https://fixture.invalid/export.json', filename: 'llm-preset-custom.json' })
    expect(api.patchLLMPreset).not.toHaveBeenCalled()
    expect(api.updateSecrets).not.toHaveBeenCalled()
  })

  it('keeps copy, connection and model discovery on the preset resource APIs', async () => {
    const { user } = await openEditor()
    await user.click(footerAction(i18n.t('llmPreset.testConnection')))
    await waitFor(() => expect(api.testLLMConnection).toHaveBeenCalledWith('custom', 60))
    await user.click(footerAction(i18n.t('llmPreset.fetchModels')))
    await waitFor(() => expect(api.refreshLLMModels).toHaveBeenCalledWith('custom', 60))
    await user.click(footerAction(i18n.t('llmPreset.saveAsCopy')))
    await waitFor(() => expect(api.duplicateLLMPreset).toHaveBeenCalledWith('custom', 'Custom fixture - Copy'))
    await waitFor(() => expect(within(editor()).getByDisplayValue('Custom fixture - Copy')).toBeInTheDocument())
    expect(api.updateSecrets).not.toHaveBeenCalled()
  })
})
