/** StudioDataMigrateModal：confirm 信息展示 / 启动调用 / running 不可关。
 *  SSE 在 jsdom 下不连（useEventStream 内部 EventSource guard），相位推进
 *  只测到 running —— done/error 由 SSE 事件驱动，后端测试覆盖事件发布。 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'

import StudioDataMigrateModal from './StudioDataMigrateModal'

const mockApi = {
  getStudioDataInfo: vi.fn(),
  startStudioDataMigrate: vi.fn(),
  getStudioDataMigrateStatus: vi.fn(),
}
vi.mock('../api/client', () => ({
  get api() { return mockApi },
}))

const INFO = {
  current: 'G:\\AnimaLoraStudio\\studio_data',
  default: 'G:\\AnimaLoraStudio\\studio_data',
  is_custom: false,
  scan: {
    total_files: 42,
    total_bytes: 5 * 1024 * 1024,
    entries: [
      { name: 'projects', is_dir: true, files: 30, bytes: 4 * 1024 * 1024 },
      { name: 'studio.db', is_dir: false, files: 1, bytes: 1024 * 1024 },
    ],
  },
}

describe('StudioDataMigrateModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getStudioDataInfo.mockResolvedValue(INFO)
    mockApi.startStudioDataMigrate.mockResolvedValue({ ok: true })
  })

  it('confirm 态展示来源/目标路径 + 文件数与大小 + 顶层明细', async () => {
    render(
      <StudioDataMigrateModal target="D:\data" onClose={() => {}} onRestart={() => {}} />,
    )
    await waitFor(() => {
      expect(screen.getByText(/共 42 个文件/)).toBeInTheDocument()
    })
    expect(screen.getByText(/5\.0 MB/)).toBeInTheDocument()
    expect(screen.getByText('D:\\data')).toBeInTheDocument()
    expect(screen.getByText('projects/')).toBeInTheDocument()
    expect(screen.getByText('studio.db')).toBeInTheDocument()
  })

  it('点开始迁移 → 调 startStudioDataMigrate(target) 并进入 running（不可关）', async () => {
    const onClose = vi.fn()
    render(
      <StudioDataMigrateModal target="D:\data" onClose={onClose} onRestart={() => {}} />,
    )
    await screen.findByText('开始迁移')
    await userEvent.click(screen.getByText('开始迁移'))
    expect(mockApi.startStudioDataMigrate).toHaveBeenCalledWith('D:\\data')
    await screen.findByText('正在复制…')
    // running 态：header 的关闭 × 不渲染
    expect(screen.queryByLabelText('关闭')).not.toBeInTheDocument()
    expect(onClose).not.toHaveBeenCalled()
  })

  it('confirm 态取消 → onClose', async () => {
    const onClose = vi.fn()
    render(
      <StudioDataMigrateModal target="D:\data" onClose={onClose} onRestart={() => {}} />,
    )
    await screen.findByText('取消')
    await userEvent.click(screen.getByText('取消'))
    expect(onClose).toHaveBeenCalled()
  })

  it('启动被后端拒绝（422）→ error 态展示原因，可关闭', async () => {
    mockApi.startStudioDataMigrate.mockRejectedValue(new Error('目标目录非空'))
    render(
      <StudioDataMigrateModal target="D:\data" onClose={() => {}} onRestart={() => {}} />,
    )
    await screen.findByText('开始迁移')
    await userEvent.click(screen.getByText('开始迁移'))
    await screen.findByText(/目标目录非空/)
    expect(screen.getByLabelText('关闭')).toBeInTheDocument()
  })
})
