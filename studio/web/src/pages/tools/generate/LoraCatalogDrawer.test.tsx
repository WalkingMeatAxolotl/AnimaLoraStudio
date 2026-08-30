import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { LoraCatalogItem, LoraEntry } from '../../../api/client'
import LoraCatalogDrawer from './LoraCatalogDrawer'
import type { LoraUiState } from './loraSelection'

const fetchMock = vi.fn()
const source = {
  source_type: 'external',
  source_id: 'external:0',
  source_label: 'ComfyUI',
  path: 'D:/ComfyUI/models/loras',
  item_count: 1,
  error: null,
  project_archived: false,
} as const
const projectSource = {
  source_type: 'project',
  source_id: 'project:1',
  source_label: 'Alpha project',
  path: 'G:/AnimaLoraStudio/studio_data/projects/alpha',
  item_count: 2,
  error: null,
  project_archived: false,
} as const
const item = {
  path: 'D:/ComfyUI/models/loras/styles/ink.safetensors',
  name: 'ink.safetensors',
  relative_path: 'styles/ink.safetensors',
  size: 1024,
  mtime: 1_700_000_000,
  source_type: 'external',
  source_id: 'external:0',
  source_label: 'ComfyUI',
  project_id: null,
  version_id: null,
  project_title: null,
  version_label: null,
  project_archived: false,
  kind: 'other',
} as const

function response(items: LoraCatalogItem[], nextCursor: number | null = null) {
  return new Response(JSON.stringify({
    items,
    sources: [source, projectSource],
    total: items.length,
    cursor: 0,
    next_cursor: nextCursor,
    generated_at: 1,
    cached: false,
    cache_ttl_seconds: 20,
  }), { status: 200, headers: { 'content-type': 'application/json' } })
}

function Harness() {
  const [open, setOpen] = useState(true)
  const [loras, setLoras] = useState<LoraEntry[]>([])
  const [ui, setUi] = useState<LoraUiState[]>([])
  return (
    <>
      <div data-testid="count">{loras.length}</div>
      {!open && <button type="button" onClick={() => setOpen(true)}>Open catalog</button>}
      <LoraCatalogDrawer
        open={open}
        onClose={() => setOpen(false)}
        loras={loras}
        ui={ui}
        onChange={(nextLoras, nextUi) => { setLoras(nextLoras); setUi(nextUi) }}
      />
    </>
  )
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((input: string | URL | Request) => {
    const url = String(input)
    return Promise.resolve(response(url.includes('source=external%3A0') ? [item] : []))
  })
})

afterEach(() => vi.unstubAllGlobals())

describe('LoraCatalogDrawer', () => {
  it('keeps the top-right close button visible and closes the drawer', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    const drawer = screen.getByTestId('lora-catalog-drawer')
    const closeButton = screen.getByRole('button', { name: '关闭' })
    expect(closeButton).not.toHaveClass('xl:hidden')
    await screen.findByRole('button', { name: /^loras / })
    const requestsBeforeClose = fetchMock.mock.calls.length
    await user.click(closeButton)

    expect(screen.getByTestId('lora-catalog-drawer')).not.toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Open catalog' }))
    expect(screen.getByTestId('lora-catalog-drawer')).toBe(drawer)
    expect(fetchMock).toHaveBeenCalledTimes(requestsBeforeClose)
  })

  it('reuses a pending catalog request across close and reopen', async () => {
    let resolveRequest!: (value: Response) => void
    fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => { resolveRequest = resolve }))
    const user = userEvent.setup()
    render(<Harness />)

    await user.click(screen.getByRole('button', { name: '关闭' }))
    await user.click(screen.getByRole('button', { name: 'Open catalog' }))
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await act(async () => { resolveRequest(response([])) })
    expect(await screen.findByRole('button', { name: /^loras / })).toBeInTheDocument()
  })

  it('ignores a stale load-more response after switching sources', async () => {
    const staleItem: LoraCatalogItem = {
      ...item,
      path: 'D:/ComfyUI/models/loras/styles/stale.safetensors',
      name: 'stale.safetensors',
      relative_path: 'styles/stale.safetensors',
    }
    const projectItem: LoraCatalogItem = {
      ...item,
      path: 'G:/AnimaLoraStudio/studio_data/projects/alpha/final.safetensors',
      name: 'final.safetensors',
      relative_path: 'final.safetensors',
      source_type: 'project',
      source_id: 'project:1',
      source_label: 'Alpha project',
      project_id: 1,
      version_id: 1,
      project_title: 'Alpha project',
      version_label: 'v1',
      kind: 'final',
    }
    let resolveLoadMore!: (value: Response) => void
    fetchMock.mockImplementation((input: string | URL | Request) => {
      const url = String(input)
      if (url.includes('cursor=1')) {
        return new Promise<Response>((resolve) => { resolveLoadMore = resolve })
      }
      if (url.includes('source=external%3A0')) return Promise.resolve(response([item], 1))
      if (url.includes('source=project%3A1')) return Promise.resolve(response([projectItem], 2))
      return Promise.resolve(response([]))
    })
    const user = userEvent.setup()
    render(<Harness />)

    await user.click(await screen.findByRole('button', { name: /^loras / }))
    await screen.findByText('ink')
    await user.click(screen.getByRole('button', { name: '加载更多' }))
    await user.click(screen.getByRole('button', { name: '返回项目和来源' }))
    await user.click(await screen.findByRole('button', { name: /^Alpha project/ }))
    expect(await screen.findByText('final')).toBeInTheDocument()

    await act(async () => { resolveLoadMore(response([staleItem])) })
    expect(screen.queryByText('stale')).not.toBeInTheDocument()
    expect(screen.getByText('final')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '加载更多' })).toBeEnabled()
  })

  it('renders sources first and only requests items after entering a source', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    await screen.findByRole('button', { name: /^loras / })
    expect(screen.queryByText('ink')).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(String(fetchMock.mock.calls[0][0])).not.toContain('source=')
    expect(screen.queryByText(/显示归档项目/)).not.toBeInTheDocument()

    expect(screen.queryByText('ComfyUI')).not.toBeInTheDocument()
    expect(screen.getByText('Alpha project')).toBeInTheDocument()
    const sourceCards = screen.getAllByTestId('lora-catalog-source')
    expect(sourceCards[0]).toHaveTextContent('Alpha project')
    expect(sourceCards[1]).toHaveTextContent('loras')

    const filter = screen.getByRole('combobox', { name: '筛选' })
    await user.selectOptions(filter, 'project')
    expect(screen.queryByRole('button', { name: /^loras / })).not.toBeInTheDocument()
    await user.selectOptions(filter, 'non_project')

    await user.click(screen.getByRole('button', { name: /^loras / }))
    expect(await screen.findByText('ink')).toBeInTheDocument()
    expect(String(fetchMock.mock.calls[fetchMock.mock.calls.length - 1]?.[0])).toContain('source=external%3A0')
    expect(screen.queryByText('ink.safetensors')).not.toBeInTheDocument()
    expect(screen.getByText('styles')).toBeInTheDocument()
  })

  it('toggles a LoRA with an ordinary row click and has no detail or add controls', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await user.click(await screen.findByRole('button', { name: /^loras / }))
    const row = (await screen.findByText('ink')).closest('[data-testid="lora-catalog-item"]')!

    await user.click(row)
    expect(screen.getByTestId('count')).toHaveTextContent('1')
    expect(row).toHaveAttribute('aria-pressed', 'true')
    expect(screen.queryByText(item.path)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /添加 LoRA/ })).not.toBeInTheDocument()

    fireEvent.click(row, { ctrlKey: true })
    await waitFor(() => expect(screen.getByTestId('count')).toHaveTextContent('0'))

    fireEvent.keyDown(window, { key: 'Escape' })
    await waitFor(() => expect(screen.getByTestId('lora-catalog-drawer')).not.toBeVisible())
  })
})
