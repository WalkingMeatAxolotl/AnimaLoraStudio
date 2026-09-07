import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type ProjectDetail, type Version } from '../../../api/client'
import TagEditPage from './TagEdit'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  confirm: vi.fn(async () => true),
  reload: vi.fn(async () => undefined),
  setVersionSwitchGuard: vi.fn(),
  onEvent: undefined as undefined | ((event: Record<string, unknown>) => void),
}))

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useBlocker: () => ({ state: 'unblocked' as const }),
  }
})
vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../components/Dialog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Dialog')>()
  return { ...actual, useDialog: () => ({ confirm: mocks.confirm }) }
})
vi.mock('../../../lib/useEventStream', () => ({
  useEventStream: (callback: (event: Record<string, unknown>) => void) => {
    mocks.onEvent = callback
  },
}))
vi.mock('../../../components/PaneResizer', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/PaneResizer')>()
  return {
    ...actual,
    default: ({ ariaLabel }: { ariaLabel: string }) => <div role="separator" aria-label={ariaLabel} />,
  }
})
vi.mock('../../../components/ZoomableImage', () => ({
  default: ({ alt }: { alt: string }) => <div data-testid="preview-image">{alt}</div>,
}))
vi.mock('../../../components/ImageGrid', () => ({
  applySelection: (selected: Set<string>, name: string) => {
    const next = new Set(selected)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    return { next, anchor: name }
  },
  default: ({
    items,
    selected,
    onSelect,
    onActivate,
    clickMode,
    ariaLabel,
    emptyHint,
  }: {
    items: Array<{ name: string; label?: string; badge?: string }>
    selected: Set<string>
    onSelect: (name: string, event: React.MouseEvent) => void
    onActivate?: (name: string) => void
    clickMode?: 'select' | 'activate'
    ariaLabel?: string
    emptyHint?: string
  }) => (
    <div role="grid" aria-label={ariaLabel}>
      {items.length === 0 ? <span>{emptyHint}</span> : items.map((item) => {
        const label = item.label ?? item.name
        return (
          <div key={item.name}>
            <button
              type="button"
              onClick={(event) => onSelect(item.name, event)}
              aria-pressed={selected.has(item.name)}
            >
              选择 {label}
            </button>
            <button
              type="button"
              onClick={(event) => {
                if (clickMode === 'select') onSelect(item.name, event)
                else onActivate?.(item.name)
              }}
            >
              打开 {label}
            </button>
            {item.badge && <span data-testid={`image-badge-${item.name}`}>{item.badge}</span>}
          </div>
        )
      })}
    </div>
  ),
}))
vi.mock('../../../components/TagEditor', () => ({
  default: ({ tags, onChange }: { tags: string[]; onChange: (tags: string[]) => void }) => (
    <div>
      <span>当前标签 {tags.join(',')}</span>
      <input
        aria-label="以文本编辑标签"
        value={tags.join(', ')}
        onChange={(event) => onChange(event.target.value.split(',').map((tag) => tag.trim()).filter(Boolean))}
      />
      <button type="button" onClick={() => onChange([...tags, 'edited'])}>修改标签</button>
    </div>
  ),
}))
vi.mock('../../../components/BulkActionBar', () => ({
  default: ({ selectedKeys }: { selectedKeys: string[] }) => (
    <div data-testid="bulk-selection">{selectedKeys.length}</div>
  ),
}))
vi.mock('../../../components/TagStatsPanel', () => ({
  default: ({ selectedKeys, onPickTag }: { selectedKeys: string[]; onPickTag: (tag: string) => void }) => (
    <div>
      <span data-testid="stats-selection">{selectedKeys.join('|')}</span>
      <button type="button" onClick={() => onPickTag('cat')}>选择 cat 图片</button>
    </div>
  ),
}))

const captions = {
  folder: null,
  items: [
    { folder: '人物 A', name: 'a1.png', tags: ['cat'], format: 'json' as const, tag_count: 1, tags_preview: ['cat'], has_caption: true },
    { folder: '人物 A', name: 'a2.png', tags: ['dog'], format: 'txt' as const, tag_count: 1, tags_preview: ['dog'], has_caption: true },
    { folder: '人物 B', name: 'b1.png', tags: ['cat'], format: 'txt' as const, tag_count: 1, tags_preview: ['cat'], has_caption: true },
  ],
}

function renderPage() {
  const project = { id: 7 } as ProjectDetail
  const activeVersion = {
    id: 11,
    trigger_word: 'sks',
    stats: { train_image_count: 3, tagged_image_count: 3 },
  } as Version
  return render(
    <MemoryRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      initialEntries={['/tag-edit']}
    >
      <Routes>
        <Route element={<Outlet context={{ project, activeVersion, reload: mocks.reload, setVersionSwitchGuard: mocks.setVersionSwitchGuard }} />}>
          <Route path="/tag-edit" element={<TagEditPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

async function ready() {
  return screen.findByRole('grid', { name: '训练图片标签编辑列表' })
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  localStorage.clear()
  mocks.onEvent = undefined
  vi.spyOn(api, 'listCaptionsFull').mockResolvedValue(captions)
  vi.spyOn(api, 'commitCaptions').mockResolvedValue({
    written: 1,
    skipped: [],
    snapshot: { id: 'snap-1', created_at: 1, size: 1, file_count: 1 },
  })
})

describe('TagEdit workspace', () => {
  it('separates first-load failure from an empty dataset and retries', async () => {
    vi.mocked(api.listCaptionsFull)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(captions)
    const user = userEvent.setup()
    renderPage()

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Caption 加载失败')
    expect(screen.queryByRole('grid')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '重试' }))
    expect(await ready()).toBeInTheDocument()
  })

  it('renders a genuine empty dataset separately from loading and errors', async () => {
    vi.mocked(api.listCaptionsFull).mockResolvedValue({ folder: null, items: [] })
    renderPage()

    expect(await screen.findByText('还没有可编辑的训练图片')).toBeInTheDocument()
    expect(screen.queryByRole('grid')).not.toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('keeps the loaded workspace mounted when a background refresh fails', async () => {
    renderPage()
    await ready()
    vi.mocked(api.listCaptionsFull).mockRejectedValueOnce(new Error('offline'))

    act(() => mocks.onEvent?.({
      type: 'version_state_changed',
      project_id: 7,
      version_id: 11,
    }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Caption 刷新失败')
    expect(screen.getByRole('grid', { name: '训练图片标签编辑列表' })).toBeInTheDocument()

    await userEvent.setup().click(screen.getByRole('button', { name: '重试' }))
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
    expect(screen.getByRole('grid', { name: '训练图片标签编辑列表' })).toBeInTheDocument()
  })

  it('never overwrites dirty captions when a relevant external update arrives', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    expect(screen.getByRole('button', { name: '保存（1）' })).toBeEnabled()

    act(() => mocks.onEvent?.({
      type: 'job_state_changed',
      kind: 'tag',
      status: 'done',
      project_id: 7,
      version_id: 11,
    }))

    expect(await screen.findByText('检测到外部 caption 更新')).toBeInTheDocument()
    expect(api.listCaptionsFull).toHaveBeenCalledTimes(1)
    expect(screen.getByText('当前标签 cat,edited')).toBeInTheDocument()
  })

  it('does not overwrite an edit made while a background refresh is in flight', async () => {
    const refreshed = {
      ...captions,
      items: captions.items.map((item) => (
        item.folder === '人物 A' && item.name === 'a1.png'
          ? { ...item, tags: ['external'] }
          : item
      )),
    }
    let resolveRefresh!: (value: typeof captions) => void
    const refreshResponse = new Promise<typeof captions>((resolve) => {
      resolveRefresh = resolve
    })
    vi.mocked(api.listCaptionsFull)
      .mockResolvedValueOnce(captions)
      .mockReturnValueOnce(refreshResponse)
    const user = userEvent.setup()
    renderPage()
    await ready()

    act(() => mocks.onEvent?.({
      type: 'version_state_changed',
      project_id: 7,
      version_id: 11,
    }))
    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledTimes(2))
    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    await act(async () => resolveRefresh(refreshed))

    expect(await screen.findByText('检测到外部 caption 更新')).toBeInTheDocument()
    expect(screen.getByText('当前标签 cat,edited')).toBeInTheDocument()
  })

  it('offers explicit save or discard paths for a dirty external update', async () => {
    const refreshed = {
      ...captions,
      items: captions.items.map((item) => (
        item.folder === '人物 A' && item.name === 'a1.png'
          ? { ...item, tags: ['external'] }
          : item
      )),
    }
    vi.mocked(api.listCaptionsFull)
      .mockResolvedValueOnce(captions)
      .mockResolvedValueOnce(refreshed)
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    act(() => mocks.onEvent?.({
      type: 'version_state_changed',
      project_id: 7,
      version_id: 11,
    }))

    expect(await screen.findByRole('button', { name: '保存并刷新' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '放弃并刷新' }))

    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledTimes(2))
    expect(api.commitCaptions).not.toHaveBeenCalled()
    expect(screen.queryByText('检测到外部 caption 更新')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    expect(screen.getByText('当前标签 external')).toBeInTheDocument()
  })

  it('saves local edits before merging an external update', async () => {
    vi.mocked(api.listCaptionsFull)
      .mockResolvedValueOnce(captions)
      .mockResolvedValueOnce({
        ...captions,
        items: captions.items.map((item) => (
          item.folder === '人物 A' && item.name === 'a1.png'
            ? { ...item, tags: ['cat', 'edited'] }
            : item
        )),
      })
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    act(() => mocks.onEvent?.({
      type: 'job_state_changed',
      kind: 'tag',
      status: 'done',
      project_id: 7,
      version_id: 11,
    }))
    await user.click(await screen.findByRole('button', { name: '保存并刷新' }))

    await waitFor(() => expect(api.commitCaptions).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledTimes(2))
    expect(screen.getByRole('button', { name: '已保存' })).toBeDisabled()
    expect(screen.queryByText('检测到外部 caption 更新')).not.toBeInTheDocument()
  })

  it('preserves edits made to another image while save-and-refresh is in flight', async () => {
    type CommitResponse = Awaited<ReturnType<typeof api.commitCaptions>>
    let resolveCommit!: (value: CommitResponse) => void
    const commitResponse = new Promise<CommitResponse>((resolve) => {
      resolveCommit = resolve
    })
    vi.mocked(api.commitCaptions).mockReturnValueOnce(commitResponse)
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    act(() => mocks.onEvent?.({
      type: 'version_state_changed',
      project_id: 7,
      version_id: 11,
    }))
    await user.click(await screen.findByRole('button', { name: '保存并刷新' }))
    await waitFor(() => expect(api.commitCaptions).toHaveBeenCalledTimes(1))

    await user.click(screen.getByRole('button', { name: '打开 人物 B/b1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    await act(async () => resolveCommit({
      written: 1,
      skipped: [],
      snapshot: { id: 'snap-save', created_at: 3, size: 1, file_count: 1 },
    }))

    await waitFor(() => expect(mocks.toast).toHaveBeenCalledWith('已保存 1 张，还原点 snap-save', 'success'))
    expect(api.listCaptionsFull).toHaveBeenCalledTimes(1)
    expect(screen.getByText('检测到外部 caption 更新')).toBeInTheDocument()
    expect(screen.getByText('当前标签 cat,edited')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存（1）' })).toBeEnabled()
  })

  it('clears a pending external update after a failed post-save refresh is retried', async () => {
    vi.mocked(api.listCaptionsFull)
      .mockResolvedValueOnce(captions)
      .mockRejectedValueOnce(new Error('refresh offline'))
      .mockResolvedValueOnce(captions)
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    act(() => mocks.onEvent?.({
      type: 'version_state_changed',
      project_id: 7,
      version_id: 11,
    }))
    await user.click(await screen.findByRole('button', { name: '保存并刷新' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Caption 刷新失败')
    expect(screen.getByText('检测到外部 caption 更新')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '重试' }))

    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledTimes(3))
    await waitFor(() => expect(screen.queryByText('检测到外部 caption 更新')).not.toBeInTheDocument())
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('limits tag picks to the active folder and closes the old editing context on folder change', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    expect(screen.getByTestId('preview-image')).toHaveTextContent('a1.png')

    await user.click(screen.getByRole('radio', { name: /人物 B/ }))
    expect(screen.queryByTestId('preview-image')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '选择 cat 图片' }))

    expect(screen.getByTestId('stats-selection')).toHaveTextContent('人物 B/b1.png')
    expect(screen.queryByRole('button', { name: '选择 人物 A/a1.png' })).not.toBeInTheDocument()
  })

  it('selects matching images across folders while the All scope is active', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '选择 cat 图片' }))

    expect(screen.getByTestId('stats-selection')).toHaveTextContent('人物 A/a1.png|人物 B/b1.png')
    expect(screen.getByTestId('bulk-selection')).toHaveTextContent('2')
  })

  it('keeps bulk selection visible instead of opening a single-image editor', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '选择 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '打开 人物 A/a2.png' }))

    expect(screen.queryByTestId('preview-image')).not.toBeInTheDocument()
    expect(screen.getByTestId('bulk-selection')).toHaveTextContent('2')
    expect(screen.getByRole('button', { name: '选择 人物 A/a2.png' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('navigates within the selection when the open image belongs to it', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '选择 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '选择 人物 A/a2.png' }))

    expect(screen.getByText('1/2')).toHaveAttribute('title', '在当前选中的图片中切换')
    await user.click(screen.getByRole('button', { name: '下一张' }))
    expect(screen.getByTestId('preview-image')).toHaveTextContent('a2.png')
  })

  it('navigates the active folder when the open image is outside the selection', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    expect(screen.getByText('1/3')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '选择 人物 A/a2.png' }))
    expect(screen.getByText('1/3')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '下一张' }))
    expect(screen.getByTestId('preview-image')).toHaveTextContent('a2.png')
  })

  it('groups the single-image editor controls in a visible panel header', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))

    expect(screen.getByRole('heading', { level: 2, name: '标签编辑' })).toBeInTheDocument()
    expect(screen.getByText('1 个标签')).toBeInTheDocument()
    const closeButton = screen.getByRole('button', { name: '关闭编辑' })
    expect(closeButton.querySelector('svg')).toHaveAttribute('stroke', 'currentColor')
    await user.click(closeButton)
    expect(screen.queryByTestId('preview-image')).not.toBeInTheDocument()
  })

  it('saves the latest text edit with one dataset-level save action', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    fireEvent.change(screen.getByRole('textbox', { name: '以文本编辑标签' }), {
      target: { value: 'cat, latest' },
    })
    await user.click(screen.getByRole('button', { name: '保存（1）' }))

    await waitFor(() => expect(api.commitCaptions).toHaveBeenCalledWith(7, 11, [{
      folder: '人物 A',
      name: 'a1.png',
      tags: ['cat', 'latest'],
    }]))
  })

  it('marks the dataset and active image as unsaved, then clears both after save', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))

    const saveButton = screen.getByRole('button', { name: '保存（1）' })
    expect(saveButton).toHaveClass('btn-danger')
    expect(screen.getByTestId('image-badge-人物 A/a1.png')).toHaveTextContent('未保存')
    const editorHeader = screen.getByRole('heading', { level: 2, name: '标签编辑' }).closest('header')
    expect(editorHeader).not.toBeNull()
    expect(within(editorHeader as HTMLElement).getByText('未保存')).toBeInTheDocument()

    await user.click(saveButton)

    await waitFor(() => expect(screen.getByRole('button', { name: '已保存' })).toBeDisabled())
    expect(screen.queryByTestId('image-badge-人物 A/a1.png')).not.toBeInTheDocument()
    expect(within(editorHeader as HTMLElement).queryByText('未保存')).not.toBeInTheDocument()
  })

  it('registers the existing version-switch guard while edits are dirty', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    await waitFor(() => {
      const calls = mocks.setVersionSwitchGuard.mock.calls
      expect(calls[calls.length - 1]?.[0]).toEqual(expect.any(Function))
    })

    const calls = mocks.setVersionSwitchGuard.mock.calls
    const guard = calls[calls.length - 1][0] as () => Promise<boolean>
    await expect(guard()).resolves.toBe(true)
    expect(mocks.confirm).toHaveBeenCalledWith(
      '你还有 1 张图的标签未保存。切换页面会丢失这些修改，确定要离开吗？',
      expect.objectContaining({ title: '有未保存的标签修改' }),
    )
  })

  it('keeps skipped caption writes dirty and preserves the commit payload contract', async () => {
    vi.mocked(api.commitCaptions).mockResolvedValue({
      written: 0,
      skipped: ['人物 A/a1.png'],
      snapshot: { id: 'snap-2', created_at: 2, size: 1, file_count: 0 },
    })
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '打开 人物 A/a1.png' }))
    await user.click(screen.getByRole('button', { name: '修改标签' }))
    await user.click(screen.getByRole('button', { name: '保存（1）' }))

    await waitFor(() => expect(api.commitCaptions).toHaveBeenCalledWith(7, 11, [{
      folder: '人物 A',
      name: 'a1.png',
      tags: ['cat', 'edited'],
    }]))
    expect(screen.getByRole('button', { name: '保存（1）' })).toBeEnabled()
    expect(mocks.toast).toHaveBeenCalledWith('已保存 0 张；1 张未写入，仍保留为待保存', 'error')
  })
})
