import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { LoraCatalogItem, LoraCatalogSource, LoraEntry } from '../../../api/client'
import XYAxisEditorDrawer from './XYAxisEditorDrawer'
import type { XYAxisDraft } from './xy'

const fetchMock = vi.fn()

const source: LoraCatalogSource = {
  source_type: 'project',
  source_id: 'project:1',
  source_label: 'Alpha project',
  path: 'G:/AnimaLoraStudio/studio_data/projects/alpha',
  item_count: 3,
  error: null,
  project_archived: false,
}

function item(name: string, kind: LoraCatalogItem['kind']): LoraCatalogItem {
  return {
    path: `G:/checkpoints/${name}.safetensors`,
    name: `${name}.safetensors`,
    relative_path: `v1/${name}.safetensors`,
    size: 1024,
    mtime: 1_700_000_000,
    source_type: 'project',
    source_id: source.source_id,
    source_label: source.source_label,
    project_id: 1,
    version_id: 2,
    project_title: 'Alpha project',
    version_label: 'v1',
    project_archived: false,
    kind,
  }
}

const finalItem = item('final', 'final')
const epoch40Item = item('epoch_40', 'epoch')
const epoch80Item = item('epoch_80', 'epoch')

function catalogResponse(items: LoraCatalogItem[], nextCursor: number | null = null) {
  return new Response(JSON.stringify({
    items,
    sources: [source],
    total: items.length + (nextCursor == null ? 0 : 1),
    cursor: 0,
    next_cursor: nextCursor,
    generated_at: 1,
    cached: false,
    cache_ttl_seconds: 20,
  }), { status: 200, headers: { 'content-type': 'application/json' } })
}

function Harness({
  initial,
  fixedLoras = [],
}: {
  initial: XYAxisDraft
  fixedLoras?: LoraEntry[]
}) {
  const [draft, setDraft] = useState(initial)
  const [open, setOpen] = useState(true)
  return (
    <>
      <output data-testid="draft-raw">{draft.raw}</output>
      <output data-testid="draft-anchor">{draft.checkpointAnchor?.path ?? ''}</output>
      {!open && <button type="button" onClick={() => setOpen(true)}>Open editor</button>}
      <XYAxisEditorDrawer
        open={open}
        label="X"
        draft={draft}
        otherAxis={null}
        fixedLoras={fixedLoras}
        onChange={setDraft}
        onClose={() => setOpen(false)}
      />
    </>
  )
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((input: string | URL | Request) => {
    const url = String(input)
    return Promise.resolve(catalogResponse(
      url.includes('source=project%3A1') ? [epoch40Item, finalItem, epoch80Item] : [],
    ))
  })
})

afterEach(() => vi.unstubAllGlobals())

describe('XYAxisEditorDrawer', () => {
  it('keeps the top-right close button visible and closes the drawer', async () => {
    const user = userEvent.setup()
    render(<Harness initial={{ axis: 'steps', raw: '20, 25', loraIndex: null }} />)

    const closeButton = screen.getByRole('button', { name: '关闭' })
    expect(closeButton).not.toHaveClass('xl:hidden')
    await user.click(closeButton)

    expect(screen.queryByTestId('xy-axis-editor-drawer')).not.toBeInTheDocument()
  })

  it('edits a numeric axis through one direct input and generates an integer range', async () => {
    const user = userEvent.setup()
    render(<Harness initial={{ axis: 'steps', raw: '20, 25', loraIndex: null }} />)

    expect(screen.getByTestId('xy-axis-editor-drawer')).toHaveClass('generate-attached-drawer')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByLabelText('轴类型')).toBeInTheDocument()

    const inputs = screen.getAllByLabelText('输入值，用逗号分隔')
    expect(inputs).toHaveLength(1)
    expect(screen.queryByPlaceholderText('搜索项目或来源…')).not.toBeInTheDocument()

    const start = screen.getByLabelText('起始')
    const end = screen.getByLabelText('结束')
    const step = screen.getByLabelText('步长')
    const addRange = screen.getByRole('button', { name: '添加范围' })

    await user.clear(start)
    await user.clear(end)
    await user.click(addRange)
    expect(screen.getByText('范围无效')).toBeInTheDocument()
    expect(screen.getByTestId('draft-raw')).toHaveTextContent('20, 25')

    await user.type(start, '20')
    await user.type(end, '22')
    await user.clear(step)
    await user.type(step, '1')
    await user.click(addRange)

    expect(screen.getByTestId('draft-raw')).toHaveTextContent('20, 21, 22')
  })

  it('generates a stable decimal range without duplicate values', async () => {
    const user = userEvent.setup()
    render(<Harness initial={{ axis: 'cfg_scale', raw: '1', loraIndex: null }} />)

    const rangeInputs = screen.getAllByRole('spinbutton')
    await user.type(rangeInputs[0], '0.1')
    await user.type(rangeInputs[1], '0.3')
    await user.clear(rangeInputs[2])
    await user.type(rangeInputs[2], '0.1')
    expect(rangeInputs[0]).toHaveValue(0.1)
    expect(rangeInputs[1]).toHaveValue(0.3)
    expect(rangeInputs[2]).toHaveValue(0.1)
    await user.click(screen.getByRole('button', { name: '添加范围' }))

    expect(screen.getByTestId('draft-raw')).toHaveTextContent('0.1, 0.2, 0.3')
  })

  it('does not repeat selected checkpoint management inside the catalog drawer', async () => {
    const user = userEvent.setup()
    render(<Harness initial={{
      axis: 'lora_ckpt',
      raw: `${finalItem.path}, ${epoch40Item.path}`,
      loraIndex: null,
      checkpointAnchor: {
        path: finalItem.path,
        scale: 1,
        project_id: finalItem.project_id,
        version_id: finalItem.version_id,
      },
    }} />)

    expect(screen.queryByTestId('xy-axis-selected-values')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /上移/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /删除/ })).not.toBeInTheDocument()
    await user.click(await screen.findByRole('button', { name: /^Alpha project/ }))
    expect(screen.queryByTestId('xy-axis-selected-values')).not.toBeInTheDocument()
  })

  it('replaces snapshot basenames and restores canonical checkpoint order', async () => {
    const user = userEvent.setup()
    render(<Harness initial={{ axis: 'lora_ckpt', raw: 'old.safetensors', loraIndex: null }} />)

    await user.click(await screen.findByRole('button', { name: /^Alpha project/ }))
    const rows = await screen.findAllByTestId('xy-axis-checkpoint')
    const row = (name: string) => rows.find((candidate) => candidate.textContent?.includes(name))!

    await user.click(row('epoch_40'))
    expect(screen.getByTestId('draft-raw')).not.toHaveTextContent('old.safetensors')
    await user.click(row('final'))
    await user.click(row('epoch_80'))

    expect(screen.getByTestId('draft-raw')).toHaveTextContent(
      `${finalItem.path}, ${epoch80Item.path}, ${epoch40Item.path}`,
    )
    expect(screen.getByTestId('draft-anchor')).toHaveTextContent(epoch40Item.path)

    await user.keyboard('{Escape}')
    await user.click(screen.getByRole('button', { name: 'Open editor' }))
    const reopenedRows = await screen.findAllByTestId('xy-axis-checkpoint')
    expect(reopenedRows.find((candidate) => candidate.textContent?.includes('final'))).toHaveAttribute('aria-pressed', 'true')
    expect(reopenedRows.find((candidate) => candidate.textContent?.includes('epoch_80'))).toHaveAttribute('aria-pressed', 'true')
    expect(reopenedRows.find((candidate) => candidate.textContent?.includes('epoch_40'))).toHaveAttribute('aria-pressed', 'true')
  })

  it('blocks checkpoints already used by an enabled fixed LoRA', async () => {
    const user = userEvent.setup()
    render(
      <Harness
        initial={{ axis: 'lora_ckpt', raw: '', loraIndex: null }}
        fixedLoras={[{ path: finalItem.path, scale: 1 }]}
      />,
    )

    await user.click(await screen.findByRole('button', { name: /^Alpha project/ }))
    const rows = await screen.findAllByTestId('xy-axis-checkpoint')
    const finalRow = rows.find((candidate) => candidate.textContent?.includes('final'))!
    expect(finalRow).toHaveAttribute('aria-disabled', 'true')
    expect(finalRow).toBeDisabled()

    await user.click(finalRow)
    await waitFor(() => expect(screen.getByTestId('draft-raw')).toHaveTextContent(/^$/))
  })
})
