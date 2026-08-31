import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type CaptionEntry, type ProjectDetail, type ProjectSummary } from '../../../api/client'
import PromptFromDatasetPicker from './PromptFromDatasetPicker'

// 测试范围：缩略图 URL（versionThumbUrl(pid, vid, …, c.name, c.folder)）拼的是
// 实时 pid/vid + caption 行名；两者只要不同步，整列 thumb 就指向别的 project 的
// 文件 → 404 黑图（线上现象：切到没切过的 project 时集体失效，刷新自愈）。根因是
// captions effect 没 stale-response 守卫，旧 (pid,vid) 的迟到响应覆盖当前 captions。

type CaptionsResult = { folder: null; items: CaptionEntry[] }

function deferred<T>() {
  let resolve!: (v: T) => void
  const promise = new Promise<T>((r) => { resolve = r })
  return { promise, resolve }
}

function cap(name: string, tag: string): CaptionEntry {
  return {
    name, folder: '2_data', tag_count: 1, tags_preview: [tag],
    has_caption: true, tags: [tag], format: 'txt',
  }
}

const projects = [{ id: 1, slug: 'a', title: 'projA' }] as unknown as ProjectSummary[]
// 组件只读 p.versions 的 id/label
const projectDetail = {
  id: 1, slug: 'a', title: 'projA',
  versions: [{ id: 11, label: 'v1' }, { id: 12, label: 'v2' }],
} as unknown as ProjectDetail

function rowThumb() {
  // 行内缩略图 alt="" → 隐式 role=presentation，getByRole('img') 取不到；直接选元素。
  // 底部大预览仅在 hover / 已选 value 时渲染，本测试都没有，故首个 img 即行缩略图。
  const img = document.querySelector('img')
  if (!img) throw new Error('no row thumbnail rendered')
  return img as HTMLImageElement
}

describe('PromptFromDatasetPicker — thumbnail / pid·vid 同步', () => {
  beforeEach(() => { localStorage.clear() })
  afterEach(() => { vi.restoreAllMocks(); localStorage.clear() })

  async function selectProjectAndV1() {
    const user = userEvent.setup()
    render(<PromptFromDatasetPicker value={null} onChange={vi.fn()} onClose={vi.fn()} />)
    await screen.findByRole('option', { name: 'projA' })
    await user.selectOptions(screen.getByLabelText('选择项目'), '1')
    // getProject(1) → vid 落到 v1(11) → captions effect 拉 (1, 11)
    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledWith(1, 11))
    return user
  }

  it('retries the project list after an initial keep-alive load failure', async () => {
    const listProjects = vi.spyOn(api, 'listProjects')
      .mockRejectedValueOnce(new Error('project list unavailable'))
      .mockResolvedValueOnce(projects)
    const user = userEvent.setup()
    render(<PromptFromDatasetPicker variant="drawer" open value={null} onChange={vi.fn()} onClose={vi.fn()} />)

    expect(await screen.findByText(/project list unavailable/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '重试' }))

    expect(await screen.findByRole('option', { name: 'projA' })).toBeInTheDocument()
    expect(listProjects).toHaveBeenCalledTimes(2)
    expect(screen.queryByText(/project list unavailable/)).not.toBeInTheDocument()
  })

  it('does not let a late project-list response clear a newer controlled project', async () => {
    localStorage.setItem('studio:generate:promptDataset:projectId', JSON.stringify(999))
    const pendingProjects = deferred<ProjectSummary[]>()
    vi.spyOn(api, 'listProjects').mockReturnValue(pendingProjects.promise)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)
    vi.spyOn(api, 'listCaptionsFull').mockResolvedValue({ folder: null, items: [] })

    const props = { onChange: vi.fn(), onClose: vi.fn() }
    const { rerender } = render(
      <PromptFromDatasetPicker value={null} {...props} />,
    )
    rerender(
      <PromptFromDatasetPicker
        value={{ projectId: 1, versionId: 11, name: 'selected.png', tags: [] }}
        {...props}
      />,
    )
    await waitFor(() => expect(api.getProject).toHaveBeenCalledWith(1))

    await act(async () => { pendingProjects.resolve(projects) })

    const projectSelect = await screen.findByLabelText('选择项目')
    await waitFor(() => expect(projectSelect).toHaveValue('1'))
  })

  it('行缩略图 URL 锚定当前 (pid, vid) + 行文件名', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)
    vi.spyOn(api, 'listCaptionsFull').mockResolvedValue({ folder: null, items: [cap('only.png', 'x')] })

    await selectProjectAndV1()

    await waitFor(() => expect(screen.getByText('only.png')).toBeInTheDocument())
    const src = rowThumb().getAttribute('src') ?? ''
    expect(src).toContain('/projects/1/versions/11/thumb')
    expect(src).toContain('name=only.png')
    expect(src).toContain('folder=2_data')
  })

  it('迟到的旧响应不覆盖当前 captions（regression：thumb 集体 404 黑图）', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)

    // v1(11) 的 captions 故意慢，模拟切到 v2 后才迟到返回
    const d11 = deferred<CaptionsResult>()
    const d12 = deferred<CaptionsResult>()
    vi.spyOn(api, 'listCaptionsFull').mockImplementation((_pid, vid) =>
      vid === 11 ? d11.promise : d12.promise
    )

    const user = await selectProjectAndV1()

    // 切到 v2(12)：触发新 captions 请求，旧 (1,11) effect cleanup 应置 cancelled
    await user.selectOptions(screen.getByLabelText('选择版本'), '12')
    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledWith(1, 12))

    // 新请求先回：列表 = v2 文件
    await act(async () => { d12.resolve({ folder: null, items: [cap('b_v12.png', 'y')] }) })
    await waitFor(() => expect(screen.getByText('b_v12.png')).toBeInTheDocument())

    // 旧请求迟到：守卫住的话应被忽略，不得把列表覆盖回 v1 文件
    await act(async () => { d11.resolve({ folder: null, items: [cap('a_v11.png', 'x')] }) })

    expect(screen.getByText('b_v12.png')).toBeInTheDocument()
    expect(screen.queryByText('a_v11.png')).not.toBeInTheDocument()
    // 缩略图也必须仍锚定当前 v2 + v2 文件名（错配就是 404 黑图的来源）
    const src = rowThumb().getAttribute('src') ?? ''
    expect(src).toContain('/versions/12/thumb')
    expect(src).toContain('name=b_v12.png')
    expect(src).not.toContain('name=a_v11.png')
  })

  it('新版本 captions 加载失败时旧行缩略图仍锚定旧 (pid,vid)（不套 live vid 出 404）', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)
    // v1(11) 正常；切到 v2(12) 的 captions 失败 → 旧行继续显示，live vid 已是 12 但
    // captions / loaded 仍是 v1(11)。缩略图绑 loaded 才对；用 live vid 就 404 黑图。
    vi.spyOn(api, 'listCaptionsFull').mockImplementation((_pid, vid) =>
      vid === 11
        ? Promise.resolve({ folder: null, items: [cap('only.png', 'x')] })
        : Promise.reject(new Error('boom'))
    )

    const user = await selectProjectAndV1()
    await waitFor(() => expect(screen.getByText('only.png')).toBeInTheDocument())

    await user.selectOptions(screen.getByLabelText('选择版本'), '12')
    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledWith(1, 12))

    // v2 captions 失败、旧行未清空仍可见；缩略图必须仍指 v1(11)，不能跟 live vid 漂到 12
    expect(screen.getByText('only.png')).toBeInTheDocument()
    const src = rowThumb().getAttribute('src') ?? ''
    expect(src).toContain('/versions/11/thumb')
    expect(src).not.toContain('/versions/12/thumb')
  })

  it('选择行返回来源，当前行再点反选；drawer 无只读框且列表/预览平分剩余高度', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)
    vi.spyOn(api, 'listCaptionsFull').mockResolvedValue({
      folder: null,
      items: [cap('first.png', 'tag one'), cap('second.png', 'tag two')],
    })
    const onChange = vi.fn()
    const user = userEvent.setup()
    const view = render(
      <PromptFromDatasetPicker variant="drawer" value={null} onChange={onChange} onClose={vi.fn()} />,
    )
    await screen.findByRole('option', { name: 'projA' })
    await user.selectOptions(screen.getByLabelText('选择项目'), '1')
    await screen.findByText('first.png')

    await user.click(screen.getByText('first.png'))
    const firstPick = onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]
    expect(firstPick).toMatchObject({ projectId: 1, versionId: 11, name: 'first.png', tags: ['tag one'] })

    view.rerender(
      <PromptFromDatasetPicker variant="drawer" value={firstPick} onChange={onChange} onClose={vi.fn()} />,
    )
    await user.click(screen.getByText('first.png'))
    expect(onChange).toHaveBeenLastCalledWith(null)

    const picker = screen.getByTestId('prompt-dataset-picker')
    expect(picker).toHaveClass('overflow-hidden')
    expect(screen.getByTestId('dataset-caption-list')).toHaveClass('flex-1', 'min-h-0', 'overflow-y-auto')
    expect(screen.getByTestId('dataset-image-preview')).toHaveClass('flex-1', 'min-h-0', 'overflow-hidden')
    expect(screen.queryByLabelText(/已选 caption 的 tags/)).not.toBeInTheDocument()
  })

  it('点击底部大图预览放大成全屏 modal（复用 ImagePreviewModal，请求 1600 大图）', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)
    vi.spyOn(api, 'listCaptionsFull').mockResolvedValue({ folder: null, items: [cap('shot.png', 'x')] })

    const user = await selectProjectAndV1()
    await waitFor(() => expect(screen.getByText('shot.png')).toBeInTheDocument())

    // 悬停一行 → 底部大图预览出现，其可点击放大按钮 aria-label="点击放大"
    await user.hover(screen.getByText('shot.png'))
    const zoomBtn = await screen.findByRole('button', { name: '点击放大' })

    // 点击放大按钮 → 全屏 ImagePreviewModal，主图 alt=文件名、URL 请求 1600 大图 + 当前 (pid,vid)。
    // 用 fireEvent.click 而非 user.click：userEvent 点击前会把指针移出再移入目标，途中触发 picker
    // 根节点 onMouseLeave 清掉 hoveredKey，使放大按钮在点击前卸载（真实浏览器里指针从列表行到大图
    // 连续移动、始终在 picker 内，根节点 mouseleave 不会触发，故属模拟 artifact）。这里只验证
    // 「点击放大按钮」本身，直接派发 click 更贴切。
    fireEvent.click(zoomBtn)
    const modalImg = await screen.findByAltText('shot.png')
    const src = modalImg.getAttribute('src') ?? ''
    expect(src).toContain('/projects/1/versions/11/thumb')
    expect(src).toContain('name=shot.png')
    expect(src).toContain('size=1600')

    // Esc 关闭（ImagePreviewModal 的键盘监听；picker 自身不监听 Esc，不冲突）
    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.queryByAltText('shot.png')).not.toBeInTheDocument())
  })
})
