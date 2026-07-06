/** ModelsRootMigrateModal：confirm 启动 / 409 target_conflict 三选流程（issue #351）。
 *  SSE 在 jsdom 下不连（useEventStream 内部 EventSource guard），相位推进
 *  只测到 running —— done/error 由 SSE 事件驱动，后端测试覆盖事件发布。 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'

import ModelsRootMigrateModal from './ModelsRootMigrateModal'

const mockApi = {
  getModelsRootInfo: vi.fn(),
  startModelsRootMigrate: vi.fn(),
  getModelsRootMigrateStatus: vi.fn(),
}
vi.mock('../api/client', () => ({
  get api() { return mockApi },
}))

const INFO = {
  current: 'G:\\AnimaLoraStudio\\models',
  default: 'G:\\AnimaLoraStudio\\models',
  is_custom: false,
  scan: {
    total_files: 7,
    total_bytes: 3 * 1024 * 1024,
    entries: [
      { name: 'diffusion_models', is_dir: true, files: 2, bytes: 2 * 1024 * 1024 },
      { name: 'vae', is_dir: true, files: 5, bytes: 1024 * 1024 },
    ],
  },
}

/** 后端 409 models_root.target_conflict 的 ApiError 形状（makeApiError 产物）。 */
const conflictError = () =>
  Object.assign(new Error('Target already contains a non-empty models directory'), {
    status: 409,
    code: 'models_root.target_conflict',
    detail: {
      target: 'D:\\newroot\\models',
      existing_files: 12,
      existing_bytes: 2 * 1024 * 1024,
      same_name_files: 3,
    },
  })

describe('ModelsRootMigrateModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getModelsRootInfo.mockResolvedValue(INFO)
    mockApi.startModelsRootMigrate.mockResolvedValue({ ok: true })
  })

  it('confirm 态点开始迁移 → 调 startModelsRootMigrate(target, undefined) 并进入 running', async () => {
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={() => {}} onDone={() => {}} />)
    await screen.findByText('开始迁移')
    await userEvent.click(screen.getByText('开始迁移'))
    expect(mockApi.startModelsRootMigrate).toHaveBeenCalledWith('D:\\newroot', undefined)
    await screen.findByText('正在复制…')
  })

  it('409 target_conflict → conflict 态展示统计，跳过已有文件 → 带 skip 重发进 running', async () => {
    mockApi.startModelsRootMigrate
      .mockRejectedValueOnce(conflictError())
      .mockResolvedValueOnce({ ok: true })
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={() => {}} onDone={() => {}} />)
    await screen.findByText('开始迁移')
    await userEvent.click(screen.getByText('开始迁移'))

    await screen.findByText('目标目录已包含 models 数据')
    // 统计插值：目标路径 + 已有文件数/大小 + 同名数
    expect(screen.getByText(/已有 12 个文件（2\.0 MB），其中 3 个/)).toBeInTheDocument()

    await userEvent.click(screen.getByText('跳过已有文件'))
    await waitFor(() => {
      expect(mockApi.startModelsRootMigrate).toHaveBeenLastCalledWith('D:\\newroot', 'skip')
    })
    await screen.findByText('正在复制…')
  })

  it('conflict 态覆盖已有文件 → 带 overwrite 重发', async () => {
    mockApi.startModelsRootMigrate
      .mockRejectedValueOnce(conflictError())
      .mockResolvedValueOnce({ ok: true })
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={() => {}} onDone={() => {}} />)
    await screen.findByText('开始迁移')
    await userEvent.click(screen.getByText('开始迁移'))
    await screen.findByText('覆盖已有文件')

    await userEvent.click(screen.getByText('覆盖已有文件'))
    await waitFor(() => {
      expect(mockApi.startModelsRootMigrate).toHaveBeenLastCalledWith('D:\\newroot', 'overwrite')
    })
  })

  it('conflict 态取消 → 回到 confirm（不关 modal，不再发请求）', async () => {
    mockApi.startModelsRootMigrate.mockRejectedValueOnce(conflictError())
    const onClose = vi.fn()
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={onClose} onDone={() => {}} />)
    await screen.findByText('开始迁移')
    await userEvent.click(screen.getByText('开始迁移'))
    await screen.findByText('取消')

    await userEvent.click(screen.getByText('取消'))
    await screen.findByText('开始迁移')
    expect(onClose).not.toHaveBeenCalled()
    expect(mockApi.startModelsRootMigrate).toHaveBeenCalledTimes(1)
  })

  it('非 conflict 错误 → error 态展示原因', async () => {
    mockApi.startModelsRootMigrate.mockRejectedValue(
      Object.assign(new Error('目标位置无效'), { status: 422, code: 'models_root.target_invalid' }),
    )
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={() => {}} onDone={() => {}} />)
    await screen.findByText('开始迁移')
    await userEvent.click(screen.getByText('开始迁移'))
    await screen.findByText(/目标位置无效/)
  })
})
