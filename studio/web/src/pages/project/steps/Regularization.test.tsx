import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type ProjectDetail, type RegStatus, type Version } from '../../../api/client'
import RegularizationPage, { ExcludeTags, SourceSegmented } from './Regularization'

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
vi.mock('../../../components/BaseModelSelect', () => ({
  default: ({ value, onChange, ariaLabel }: {
    value: string | null
    onChange: (value: string) => void
    ariaLabel?: string
  }) => (
    <select aria-label={ariaLabel} value={value ?? ''} onChange={(event) => onChange(event.target.value)}>
      <option value="">默认底模</option>
      <option value="test-model">测试底模</option>
    </select>
  ),
}))
vi.mock('../../../components/ImageGrid', () => ({
  applySelection: (selected: Set<string>, name: string) => {
    const next = new Set(selected)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    return { next, anchor: name }
  },
  default: ({ items, selected, onSelect, ariaLabel }: {
    items: { name: string }[]
    selected: Set<string>
    onSelect: (name: string, event: React.MouseEvent) => void
    ariaLabel?: string
  }) => (
    <div role="grid" aria-label={ariaLabel}>
      {items.map((item) => (
        <button
          type="button"
          key={item.name}
          aria-pressed={selected.has(item.name)}
          onClick={(event) => onSelect(item.name, event)}
        >
          选择 {item.name}
        </button>
      ))}
    </div>
  ),
}))
vi.mock('../../../components/ImagePreviewModal', () => ({ default: () => null }))

const emptyReg: RegStatus = {
  exists: false,
  meta: null,
  image_count: 0,
  files: [],
}
const existingReg: RegStatus = {
  exists: true,
  image_count: 3,
  files: ['1_data/a.png', '1_data/b.png', '1_data/c.png'],
  meta: {
    generated_at: 1_700_000_000,
    based_on_version: 'v1',
    api_source: 'gelbooru',
    target_count: 3,
    actual_count: 3,
    source_tags: [],
    excluded_tags: [],
    blacklist_tags: [],
    failed_tags: [],
    train_tag_distribution: {},
    auto_tagged: true,
    auto_tag_kind: 'wd14',
    build_mode: 'flat',
    incremental_runs: 0,
    postprocessed_at: null,
    postprocess_clusters: null,
    postprocess_method: null,
    postprocess_max_crop_ratio: null,
    generation_method: 'scrape',
  },
}

function renderPage() {
  const project = { id: 7 } as ProjectDetail
  const activeVersion = {
    id: 11,
    stats: { train_image_count: 4 },
  } as Version
  return render(
    <MemoryRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      initialEntries={['/regularization']}
    >
      <Routes>
        <Route element={<Outlet context={{ project, activeVersion, reload: mocks.reload }} />}>
          <Route path="/regularization" element={<RegularizationPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

async function ready() {
  await screen.findByText('本次生成计划')
  await waitFor(() => expect(api.getRegStatus).toHaveBeenCalled())
}

beforeEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
  localStorage.clear()
  mocks.onEvent = undefined
  mocks.confirm.mockResolvedValue(true)
  vi.spyOn(api, 'getRegStatus').mockResolvedValue(emptyReg)
  vi.spyOn(api, 'previewRegTags').mockResolvedValue([{ tag: '1girl', count: 4 }])
  vi.spyOn(api, 'getSecrets').mockResolvedValue({ reg: { default_excluded_tags: [] } } as never)
  vi.spyOn(api, 'getVersionConfig').mockResolvedValue({ has_config: false, config: null } as never)
  vi.spyOn(api, 'getLatestVersionJob').mockResolvedValue({ job: null, log: '' })
  vi.spyOn(api, 'getLatestRegPriorTask').mockResolvedValue({ task: null, log: '' })
  vi.spyOn(api, 'enqueueRegPrior').mockResolvedValue({ id: 31, status: 'pending', version_id: 11 } as never)
  vi.spyOn(api, 'startRegBuild').mockResolvedValue({ id: 41, status: 'pending', version_id: 11 } as never)
  vi.spyOn(api, 'deleteReg').mockResolvedValue({ deleted: true })
  vi.spyOn(api, 'deleteRegFiles').mockResolvedValue({ deleted: ['1_data/a.png'], count: 1 })
  vi.spyOn(api, 'dedupPurgeReg').mockResolvedValue({ scanned: 3, groups: 1, deleted: ['1_data/b.png'], count: 1 })
  vi.spyOn(api, 'getJob').mockResolvedValue({ id: 41, status: 'done', version_id: 11 } as never)
  vi.spyOn(api, 'getRegPriorTask').mockResolvedValue({ id: 77, status: 'done', version_id: 11 } as never)
  vi.spyOn(api, 'cancelTask').mockResolvedValue({ canceled: true } as never)
  vi.spyOn(api, 'cancelJob').mockResolvedValue({ canceled: true } as never)
})

describe('regularization source selector', () => {
  it('keeps the original content-sized pill radios and switches source with the keyboard', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<SourceSegmented source="ai" onChange={onChange} />)

    const group = screen.getByRole('radiogroup', { name: '来源' })
    const ai = within(group).getByRole('radio', { name: /AI 先验/ })
    const booru = within(group).getByRole('radio', { name: /Booru 抓取/ })
    expect(group).not.toHaveClass('ui-selection-segmented')
    expect(ai).toHaveClass('pill-radio', 'pill-radio-content')
    expect(ai).toHaveAttribute('data-state', 'active')
    expect(ai).toHaveAttribute('tabindex', '0')
    expect(booru).toHaveAttribute('tabindex', '-1')

    ai.focus()
    await user.keyboard('{ArrowRight}')
    expect(onChange).toHaveBeenCalledWith('booru')
  })
})

describe('regularization exclusion chips', () => {
  it('keeps a long natural-language tag inside a single-height chip', () => {
    const longTag = 'a single girl with very long pale lavender hair faces toward the viewer while the background extends across the entire frame'
    render(
      <ExcludeTags
        trainTags={[{ tag: longTag, count: 1 }]}
        loading={false}
        excluded={new Set()}
        onToggle={vi.fn()}
      />,
    )

    const text = screen.getByText(longTag)
    const textContainer = text.parentElement
    const chip = text.closest('button')
    expect(chip).toHaveClass('h-6', 'max-w-full', 'overflow-hidden', 'whitespace-nowrap')
    expect(chip).toHaveAttribute('aria-pressed', 'false')
    expect(textContainer).toHaveClass('min-w-0', 'truncate', 'text-left')
    expect(textContainer).toHaveAttribute('title', longTag)
  })

  it('distinguishes a loading tag list from a real empty list', () => {
    const { rerender } = render(
      <ExcludeTags trainTags={[]} loading excluded={new Set()} onToggle={vi.fn()} />,
    )
    expect(screen.getByText('加载中...')).toBeInTheDocument()
    expect(screen.queryByText(/train 还没有 tag 分布/)).not.toBeInTheDocument()

    rerender(<ExcludeTags trainTags={[]} loading={false} excluded={new Set()} onToggle={vi.fn()} />)
    expect(screen.getByText(/train 还没有 tag 分布/)).toBeInTheDocument()
  })
})

describe('Regularization workspace', () => {
  it('keeps the page action stable and submits the default AI prior payload', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    expect(screen.getByRole('tab', { name: '生成' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tablist', { name: '正则集工作阶段' })).toHaveClass('w-full', 'px-page')
    expect(screen.getByRole('tabpanel')).toHaveClass(
      'xl:grid-cols-[minmax(0,3fr)_minmax(20rem,2fr)]',
    )
    expect(screen.getByRole('group', { name: '正则集任务操作' })).not.toHaveTextContent(
      'AI 先验 · 增量补足 · 4 张',
    )
    expect(screen.getByText('本次生成计划')).toBeInTheDocument()
    expect(screen.getByText('当前正则集')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '生成 AI 先验' }))
    await waitFor(() => expect(api.enqueueRegPrior).toHaveBeenCalledWith(7, 11, expect.objectContaining({
      incremental: true,
      excluded_tags: [],
      width: 1024,
      height: 1024,
    })))
    expect(mocks.confirm).not.toHaveBeenCalled()
  })

  it('requires confirmation before a full rebuild and does not enqueue when cancelled', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getRegStatus).mockResolvedValue(existingReg)
    mocks.confirm.mockResolvedValueOnce(false)
    renderPage()
    await ready()

    await user.selectOptions(screen.getByRole('combobox', { name: '生成范围' }), 'full')
    await user.click(screen.getByRole('button', { name: '生成 AI 先验' }))

    expect(mocks.confirm).toHaveBeenCalledWith(
      expect.stringContaining('将删除当前正则集的 3 张图片'),
      expect.objectContaining({ tone: 'danger', okText: '确认重建' }),
    )
    expect(api.enqueueRegPrior).not.toHaveBeenCalled()
  })

  it('switches to Booru without moving the primary action and preserves the build payload', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('radio', { name: /Booru 抓取/ }))
    expect(screen.getByRole('button', { name: '从 Booru 抓取' })).toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox', { name: '结构' }), 'mirror')
    await user.click(screen.getByRole('button', { name: '从 Booru 抓取' }))

    await waitFor(() => expect(api.startRegBuild).toHaveBeenCalledWith(7, 11, expect.objectContaining({
      api_source: 'gelbooru',
      incremental: true,
      build_mode: 'mirror',
      target_count: null,
      auto_tag: true,
      auto_tag_kind: 'wd14',
    })))
  })

  it('labels editable controls as next-run settings while a task is running', async () => {
    vi.mocked(api.getLatestRegPriorTask).mockResolvedValue({
      task: { id: 77, status: 'running', version_id: 11 },
      log: 'running',
    } as never)
    renderPage()

    expect(await screen.findByText('当前任务：AI 先验 #77')).toBeInTheDocument()
    expect(screen.getByText('下一轮生成计划')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '生成范围' })).toBeEnabled()
    expect(screen.getByRole('button', { name: '生成中…' })).toBeDisabled()
  })

  it('keeps train-tag fetch failure distinct from a real empty distribution', async () => {
    vi.mocked(api.previewRegTags).mockRejectedValueOnce(new Error('tag service offline'))
    renderPage()

    expect(await screen.findByText('训练标签加载失败')).toBeInTheDocument()
    expect(screen.getByText('训练标签列表暂不可用；可继续使用已有或自定义排除标签。')).toBeInTheDocument()
    expect(screen.queryByText(/train 还没有 tag 分布/)).not.toBeInTheDocument()
  })

  it('keeps conflicting actions disabled while preserving an editable next-run draft', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getRegStatus).mockResolvedValue(existingReg)
    vi.mocked(api.getLatestRegPriorTask).mockResolvedValue({
      task: { id: 77, status: 'running', version_id: 11 },
      log: 'running',
    } as never)
    renderPage()

    expect(await screen.findByText('当前任务：AI 先验 #77')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '清空正则集' })).toBeDisabled()
    expect(screen.getByRole('combobox', { name: '生成范围' })).toBeEnabled()

    await user.click(screen.getByRole('tab', { name: '图片（3）' }))
    expect(screen.getByRole('button', { name: '自动去重' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '选择 1_data/a.png' }))
    expect(screen.getByRole('button', { name: '删除选中 (1)' })).toBeDisabled()
  })

  it('refreshes project statistics when an AI task reaches a terminal state', async () => {
    vi.mocked(api.getLatestRegPriorTask).mockResolvedValue({
      task: { id: 77, status: 'running', version_id: 11 },
      log: 'running',
    } as never)
    renderPage()

    expect(await screen.findByText('当前任务：AI 先验 #77')).toBeInTheDocument()
    await act(async () => {
      mocks.onEvent?.({ type: 'task_state_changed', task_id: 77, status: 'done' })
    })
    await waitFor(() => expect(mocks.reload).toHaveBeenCalled())
  })

  it('shows an actionable initial load error and recovers in place', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getRegStatus)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(emptyReg)
    renderPage()

    expect(await screen.findByText('正则集状态加载失败')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: '重试' }).length).toBeGreaterThan(0)
    await user.click(screen.getAllByRole('button', { name: '重试' })[0])
    await waitFor(() => expect(api.getRegStatus).toHaveBeenCalledTimes(2))
    expect(await screen.findByText('当前版本 reg 集：不存在')).toBeInTheDocument()
  })

  it('unlocks the existing folder structure only after choosing a full rebuild', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getRegStatus).mockResolvedValue(existingReg)
    renderPage()
    await ready()

    await user.click(screen.getByRole('radio', { name: /Booru 抓取/ }))
    const structure = screen.getByRole('combobox', { name: '结构' })
    expect(structure).toBeDisabled()
    await user.selectOptions(screen.getByRole('combobox', { name: '生成范围' }), 'full')
    expect(structure).toBeEnabled()
  })

  it('keeps a loaded image workspace visible when a background refresh fails', async () => {
    vi.mocked(api.getRegStatus)
      .mockResolvedValueOnce(existingReg)
      .mockRejectedValueOnce(new Error('refresh offline'))
    vi.mocked(api.getLatestVersionJob).mockResolvedValue({
      job: { id: 41, status: 'running', version_id: 11 },
      log: 'running',
    } as never)
    renderPage()

    expect(await screen.findByText('当前任务：Booru 抓取 #41')).toBeInTheDocument()
    await act(async () => {
      mocks.onEvent?.({ type: 'job_state_changed', job_id: 41, status: 'done' })
    })

    expect(await screen.findByRole('grid', { name: '正则集图片' })).toBeInTheDocument()
    expect(screen.getByText('正则集状态刷新失败')).toBeInTheDocument()
    expect(screen.getByText('选择 1_data/a.png')).toBeInTheDocument()
  })

  it('keeps image filtering and destructive batch actions in the image workspace', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getRegStatus).mockResolvedValue(existingReg)
    renderPage()
    await ready()

    await user.click(screen.getByRole('tab', { name: '图片（3）' }))
    const folderFilter = screen.getByRole('radiogroup', { name: '正则集文件夹筛选' })
    expect(folderFilter).not.toHaveClass('ui-selection-segmented', 'flex-1')
    expect(within(folderFilter).getByRole('radio', { name: '全部 3' })).toHaveClass('rounded-full')
    await user.click(screen.getByRole('button', { name: '选择 1_data/a.png' }))
    await user.click(screen.getByRole('button', { name: '删除选中 (1)' }))

    expect(mocks.confirm).toHaveBeenCalledWith(
      expect.stringContaining('删除选中 1 张图'),
      expect.objectContaining({ tone: 'danger' }),
    )
    expect(api.deleteRegFiles).toHaveBeenCalledWith(7, 11, ['1_data/a.png'])
  })

  it('exposes the image stage as a real tab and routes an empty state back to generation', async () => {
    const user = userEvent.setup()
    renderPage()
    await ready()

    await user.click(screen.getByRole('tab', { name: '图片' }))
    expect(screen.getByRole('tabpanel', { name: '图片' })).toBeInTheDocument()
    expect(screen.getByText('还没有正则集')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '前往生成' }))
    expect(screen.getByRole('tab', { name: '生成' })).toHaveAttribute('aria-selected', 'true')
  })
})
