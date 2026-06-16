import { act, render, screen, waitFor } from '@testing-library/react'
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
})
