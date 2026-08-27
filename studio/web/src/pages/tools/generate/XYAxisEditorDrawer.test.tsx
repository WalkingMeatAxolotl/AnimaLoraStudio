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
  it('edits a numeric axis through one direct input and generates an integer range', async () => {
    const user = userEvent.setup()
    render(<Harness initial={{ axis: 'steps', raw: '20, 25', loraIndex: null }} />)

    expect(screen.getByRole('dialog')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByLabelText('轴类型')).toHaveFocus())

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

  it('keeps an absolute checkpoint anchor when removing it before catalog rows load', async () => {
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

    await user.click(screen.getAllByRole('button', { name: '删除' })[0])
    expect(screen.getByTestId('draft-raw')).toHaveTextContent(epoch40Item.path)
    expect(screen.getByTestId('draft-anchor')).toHaveTextContent(epoch40Item.path)
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

    // Once the user reorders the list manually, later additions append rather
    // than silently restoring canonical order.
    const moveUpButtons = screen.getAllByRole('button', { name: '上移' })
    await user.click(moveUpButtons[moveUpButtons.length - 1])
    await user.click(row('final'))
    await user.click(row('final'))
    expect(screen.getByTestId('draft-raw')).toHaveTextContent(
      `${epoch40Item.path}, ${epoch80Item.path}, ${finalItem.path}`,
    )

    await user.click(screen.getByRole('button', { name: '关闭' }))
    await user.click(screen.getByRole('button', { name: 'Open editor' }))
    await user.click((await screen.findAllByTestId('xy-axis-checkpoint')).find(
      (candidate) => candidate.textContent?.includes('final'),
    )!)
    await user.click((await screen.findAllByTestId('xy-axis-checkpoint')).find(
      (candidate) => candidate.textContent?.includes('final'),
    )!)
    expect(screen.getByTestId('draft-raw')).toHaveTextContent(
      `${epoch40Item.path}, ${epoch80Item.path}, ${finalItem.path}`,
    )
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
