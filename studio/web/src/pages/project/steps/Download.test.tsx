import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  api,
  type DownloadFile,
  type Job,
  type ProjectDetail,
} from '../../../api/client'
import DownloadPage from './Download'

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  confirm: vi.fn(async () => true),
  reload: vi.fn(async () => undefined),
  onEvent: undefined as undefined | ((event: unknown) => void),
}))

vi.mock('../../../components/Toast', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Toast')>()
  return { ...actual, useToast: () => ({ toast: mocks.toast }) }
})
vi.mock('../../../components/Dialog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../components/Dialog')>()
  return { ...actual, useDialog: () => ({ confirm: mocks.confirm }) }
})
vi.mock('../../../lib/useEventStream', () => ({
  useEventStream: (callback: (event: unknown) => void) => { mocks.onEvent = callback },
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
    ariaLabel,
    className,
    contentClassName,
    emptyHint,
  }: {
    items: { name: string }[]
    selected: Set<string>
    onSelect: (name: string, event: React.MouseEvent) => void
    onActivate?: (name: string) => void
    ariaLabel?: string
    className?: string
    contentClassName?: string
    emptyHint?: string
  }) => (
    <div role="grid" aria-label={ariaLabel} className={className} data-content-class={contentClassName}>
      {items.length === 0 ? <span>{emptyHint}</span> : items.map((item) => (
        <div key={item.name}>
          <button type="button" onClick={(event) => onSelect(item.name, event)} aria-pressed={selected.has(item.name)}>
            选择 {item.name}
          </button>
          <button type="button" onClick={() => onActivate?.(item.name)}>
            预览 {item.name}
          </button>
        </div>
      ))}
    </div>
  ),
}))
vi.mock('../../../components/ImagePreviewModal', () => ({
  default: ({ caption, onClose }: { caption?: string; onClose: () => void }) => (
    <div role="dialog" aria-label="图片预览">
      <span>{caption}</span>
      <button type="button" onClick={onClose}>关闭预览</button>
    </div>
  ),
}))
vi.mock('../../../components/PathPicker', () => ({
  default: ({ onPick, onClose }: { onPick: (path: string) => void; onClose: () => void }) => (
    <div role="dialog" aria-label="服务器文件选择器">
      <button type="button" onClick={() => onPick('/srv/images.zip')}>选择测试压缩包</button>
      <button type="button" onClick={onClose}>关闭路径选择器</button>
    </div>
  ),
}))

const files: DownloadFile[] = [
  { name: 'a.png', size: 1024, has_meta: true },
  { name: 'b.jpg', size: 2048, has_meta: false },
]
const runningJob: Job = {
  id: 11,
  project_id: 7,
  version_id: null,
  kind: 'download',
  params: '{}',
  status: 'running',
  started_at: null,
  finished_at: null,
  pid: null,
  log_path: null,
  error_msg: null,
}

function renderPage() {
  const project = { id: 7, download_image_count: 2 } as ProjectDetail
  return render(
    <MemoryRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      initialEntries={['/download']}
    >
      <Routes>
        <Route element={<Outlet context={{ project, activeVersion: null, reload: mocks.reload }} />}>
          <Route path="/download" element={<DownloadPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

async function ready() {
  await screen.findByRole('grid', { name: '已加入项目的原始素材' })
  await screen.findByText('2 张 · 3.0 KB')
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  mocks.onEvent = undefined
  vi.spyOn(api, 'listFiles').mockResolvedValue({ items: files, count: files.length })
  vi.spyOn(api, 'getDownloadStatus').mockResolvedValue({ job: null, log_tail: '' })
  vi.spyOn(api, 'estimateDownload').mockResolvedValue({
    tag: 'character_x',
    api_source: 'gelbooru',
    exclude_tags: ['rating:explicit'],
    effective_query: 'character_x -rating:explicit',
    count: 42,
  })
  vi.spyOn(api, 'startDownload').mockResolvedValue(runningJob)
  vi.spyOn(api, 'uploadProjectFiles').mockResolvedValue({ added: ['local.png'], skipped: [] })
  vi.spyOn(api, 'uploadProjectFileFromPath').mockResolvedValue({ added: ['server.png'], skipped: [] })
  vi.spyOn(api, 'deleteProjectFiles').mockResolvedValue({ deleted: ['a.png'], missing: [] })
  vi.spyOn(api, 'cancelJob').mockResolvedValue({ task_id: runningJob.id, canceled: true })
})

describe('Download workspace', () => {
  it('keeps estimate and download as two explicit Booru steps with the existing payloads', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    expect(screen.queryByRole('tablist')).not.toBeInTheDocument()
    const acquisitionGrid = document.querySelector('[data-download-acquisition-grid]')
    expect(acquisitionGrid).toHaveClass('xl:grid-cols-2')
    expect(screen.getByRole('region', { name: 'Booru 抓取' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: '文件导入' })).toBeInTheDocument()
    expect(screen.getByText('尚未查询')).toBeInTheDocument()
    expect(screen.queryByText('—')).not.toBeInTheDocument()
    expect(screen.queryByRole('spinbutton', { name: '本次下载' })).not.toBeInTheDocument()
    const initialStart = screen.getByRole('button', { name: '开始下载' })
    expect(initialStart).toBeDisabled()
    expect(initialStart.closest('[role="status"]')).toBeNull()
    await user.type(screen.getByRole('textbox', { name: '查询标签' }), 'character_x')
    await user.click(screen.getByRole('button', { name: '查询' }))

    expect(api.estimateDownload).toHaveBeenCalledWith(7, {
      tag: 'character_x',
      api_source: 'gelbooru',
    })
    const count = await screen.findByRole('spinbutton', { name: '本次下载' })
    expect(count).toHaveValue(42)
    expect(screen.getAllByText('42')).toHaveLength(1)
    expect(screen.queryByRole('button', { name: '全部 42' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始下载' })).toBeInTheDocument()
    const exclusionHint = screen.getByText('· 已应用 1 个排除标签')
    expect(exclusionHint).toHaveAttribute('title', 'character_x -rating:explicit')
    await user.clear(count)
    await user.type(count, '12{Enter}')
    await waitFor(() => expect(api.startDownload).toHaveBeenCalledWith(7, {
      tag: 'character_x',
      count: 12,
      api_source: 'gelbooru',
    }))
  })

  it('keeps estimate and start loading states on their owning buttons', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.type(screen.getByRole('textbox', { name: '查询标签' }), 'character_x')
    await user.click(screen.getByRole('button', { name: '查询' }))
    const startButton = screen.getByRole('button', { name: '开始下载' })

    let resolveEstimate: ((value: {
      tag: string
      api_source: 'gelbooru' | 'danbooru'
      exclude_tags: string[]
      effective_query: string
      count: number
    }) => void) | undefined
    vi.mocked(api.estimateDownload).mockImplementationOnce(() => new Promise((resolve) => {
      resolveEstimate = resolve
    }))
    const queryButton = screen.getByRole('button', { name: '查询' })
    await user.click(queryButton)

    expect(queryButton).toHaveAttribute('aria-busy', 'true')
    expect(startButton).not.toHaveAttribute('aria-busy')
    await act(async () => {
      resolveEstimate?.({
        tag: 'character_x',
        api_source: 'gelbooru',
        exclude_tags: [],
        effective_query: 'character_x',
        count: 42,
      })
    })
    expect(queryButton).not.toHaveAttribute('aria-busy')
    expect(screen.queryByText(/已应用 .* 个排除标签/)).not.toBeInTheDocument()
  })

  it('handles zero and unknown estimate counts without skipping confirmation', async () => {
    const user = userEvent.setup()
    vi.mocked(api.estimateDownload)
      .mockResolvedValueOnce({
        tag: 'none',
        api_source: 'gelbooru',
        exclude_tags: [],
        effective_query: 'none',
        count: 0,
      })
      .mockResolvedValueOnce({
        tag: 'unknown',
        api_source: 'gelbooru',
        exclude_tags: [],
        effective_query: 'unknown',
        count: -1,
      })
    renderPage()
    await ready()

    const tag = screen.getByRole('textbox', { name: '查询标签' })
    await user.type(tag, 'none')
    await user.click(screen.getByRole('button', { name: '查询' }))
    expect(screen.queryByRole('spinbutton', { name: '本次下载' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始下载' })).toBeDisabled()

    await user.clear(tag)
    await user.type(tag, 'unknown')
    await user.click(screen.getByRole('button', { name: '查询' }))
    const count = await screen.findByRole('spinbutton', { name: '本次下载' })
    expect(count).toHaveValue(20)
    await user.type(count, '{Enter}')
    await waitFor(() => expect(api.startDownload).toHaveBeenCalledWith(7, {
      tag: 'unknown',
      count: 20,
      api_source: 'gelbooru',
    }))
  })

  it('keeps both acquisition cards visible and imports local files through the shared confirmation', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    const chooseLocalButton = screen.getByRole('button', { name: '选择当前设备文件' })
    expect(chooseLocalButton).toHaveAttribute('title', 'png / jpg / webp / bmp / gif / zip')
    expect(screen.queryByText('png / jpg / webp / bmp / gif / zip')).not.toBeInTheDocument()
    const file = new File(['image'], 'local.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('当前设备中的图片或 zip 文件'), file)
    const localSummary = document.querySelector('[data-download-import-summary]')
    expect(localSummary).toHaveTextContent('当前设备 · 1 个文件 · 0.0 MB')
    expect(localSummary).toHaveAttribute('title', expect.stringContaining('local.png'))
    expect(screen.getByRole('textbox', { name: '查询标签' })).toHaveValue('')

    vi.mocked(api.uploadProjectFiles).mockResolvedValueOnce({
      added: ['local.png'],
      skipped: [{ name: 'archive-entry.txt', reason: 'unsupported' }],
    })
    await user.click(screen.getByRole('button', { name: '从当前设备导入 1 个文件' }))
    await waitFor(() => expect(api.uploadProjectFiles).toHaveBeenCalledWith(
      7,
      [file],
      expect.any(Function),
    ))
    expect(await screen.findByText('导入完成：添加 1 张，跳过 1 张')).toBeInTheDocument()
    await user.click(screen.getByText('查看 1 个跳过项'))
    expect(screen.getByText(/archive-entry\.txt/)).toBeInTheDocument()
    expect(mocks.toast).not.toHaveBeenCalled()
    expect(mocks.reload).toHaveBeenCalled()
  })

  it('keeps selected local files after an upload transport failure', async () => {
    const user = userEvent.setup()
    vi.mocked(api.uploadProjectFiles).mockRejectedValueOnce(new Error('network unavailable'))
    renderPage()
    await ready()

    const file = new File(['image'], 'retry.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('当前设备中的图片或 zip 文件'), file)
    await user.click(screen.getByRole('button', { name: '从当前设备导入 1 个文件' }))

    await waitFor(() => expect(mocks.toast).toHaveBeenCalledWith('Error: network unavailable', 'error'))
    const retrySummary = document.querySelector('[data-download-import-summary]')
    expect(retrySummary).toHaveAttribute('title', expect.stringContaining('retry.png'))
  })

  it('replaces a pending local selection with a server path and waits for shared confirmation', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    const local = new File(['image'], 'replace-me.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('当前设备中的图片或 zip 文件'), local)
    const pendingSummary = document.querySelector('[data-download-import-summary]')
    expect(pendingSummary).toHaveAttribute('title', expect.stringContaining('replace-me.png'))

    await user.click(screen.getByRole('button', { name: '浏览运行服务器' }))
    expect(screen.getByRole('dialog', { name: '服务器文件选择器' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '选择测试压缩包' }))

    expect(api.uploadProjectFileFromPath).not.toHaveBeenCalled()
    expect(pendingSummary).not.toHaveAttribute('title', expect.stringContaining('replace-me.png'))
    expect(pendingSummary).toHaveTextContent('运行服务器 · /srv/images.zip')
    await user.click(screen.getByRole('button', {
      name: '从运行服务器导入 /srv/images.zip',
    }))
    await waitFor(() => expect(api.uploadProjectFileFromPath).toHaveBeenCalledWith(7, '/srv/images.zip'))
    expect(await screen.findByText('导入完成：添加 1 张，跳过 0 张')).toBeInTheDocument()
  })

  it('replaces a pending server path with files from the current device', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '浏览运行服务器' }))
    await user.click(screen.getByRole('button', { name: '选择测试压缩包' }))
    const serverSummary = document.querySelector('[data-download-import-summary]')
    expect(serverSummary).toHaveTextContent('运行服务器 · /srv/images.zip')

    const local = new File(['image'], 'local-wins.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('当前设备中的图片或 zip 文件'), local)

    expect(serverSummary).not.toHaveTextContent('/srv/images.zip')
    expect(serverSummary).toHaveTextContent('当前设备 · 1 个文件 · 0.0 MB')
    expect(serverSummary).toHaveAttribute('title', expect.stringContaining('local-wins.png'))
    const importButton = screen.getByRole('button', { name: '从当前设备导入 1 个文件' })
    expect(importButton.closest('[role="status"]')).toBeNull()
    await user.click(importButton)
    await waitFor(() => expect(api.uploadProjectFiles).toHaveBeenCalledWith(
      7,
      [local],
      expect.any(Function),
    ))
    expect(api.uploadProjectFileFromPath).not.toHaveBeenCalled()
  })

  it('keeps the image workspace primary and permanently deletes only after confirmation', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    expect(screen.queryByText('格式分布')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '选择 a.png' }))
    mocks.confirm.mockResolvedValueOnce(false)
    await user.click(screen.getByRole('button', { name: '删除 1' }))
    expect(api.deleteProjectFiles).not.toHaveBeenCalled()

    mocks.confirm.mockResolvedValueOnce(true)
    await user.click(screen.getByRole('button', { name: '删除 1' }))
    expect(mocks.confirm).toHaveBeenLastCalledWith(
      expect.stringContaining('操作不可恢复'),
      { tone: 'danger', okText: '删除' },
    )
    expect(api.deleteProjectFiles).toHaveBeenCalledWith(7, ['a.png'])
  })

  it('opens the existing preview flow from an image cell', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('button', { name: '预览 b.jpg' }))
    expect(screen.getByRole('dialog', { name: '图片预览' })).toHaveTextContent('b.jpg')
  })

  it('retains loaded images and offers an in-place retry after refresh failure', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()
    vi.mocked(api.listFiles).mockRejectedValueOnce(new Error('offline'))

    act(() => mocks.onEvent?.({ type: 'project_state_changed', project_id: 7 }))
    expect(await screen.findByText('图片列表刷新失败')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '预览 a.png' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '重试' }))
    await waitFor(() => expect(screen.queryByText('图片列表刷新失败')).not.toBeInTheDocument())
  })

  it('shows a loading state before the first file inventory resolves', async () => {
    let resolveFiles: ((value: { items: DownloadFile[]; count: number }) => void) | undefined
    vi.mocked(api.listFiles).mockImplementationOnce(() => new Promise((resolve) => {
      resolveFiles = resolve
    }))
    renderPage()

    expect(await screen.findByText('加载中...')).toBeInTheDocument()
    await act(async () => {
      resolveFiles?.({ items: files, count: files.length })
    })
    await ready()
  })

  it('distinguishes an initial load failure from an empty project', async () => {
    vi.mocked(api.listFiles).mockRejectedValueOnce(new Error('offline'))
    const user = userEvent.setup()
    renderPage()

    expect(await screen.findByText('图片列表加载失败')).toBeInTheDocument()
    expect(screen.getByText('图片列表暂时不可用。重试后再检查已有素材。')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '重试' }))
    await ready()
  })
})
