/** StudioDataMigrateModal：confirm 信息展示 / 启动调用 / running 不可关。
 *  SSE 在 jsdom 下不连（useEventStream 内部 EventSource guard），相位推进
 *  只测到 running —— done/error 由 SSE 事件驱动，后端测试覆盖事件发布。 */
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'

import StudioDataMigrateModal from './StudioDataMigrateModal'

let onEventCb: ((evt: Record<string, unknown>) => void) | null = null

vi.mock('../lib/useEventStream', () => ({
  useEventStream: (cb: (evt: Record<string, unknown>) => void) => {
    onEventCb = cb
  },
}))

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
    onEventCb = null
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
    expect(screen.getByRole('dialog')).toHaveAttribute('aria-labelledby')
    expect(screen.getByText(/5\.0 MB/)).toBeInTheDocument()
    // 目标显示实际落地目录 target\studio_data（用户选的是父目录）
    expect(screen.getByText('D:\\data\\studio_data')).toBeInTheDocument()
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
    const progress = screen.getByRole('progressbar', { name: '正在复制…' })
    expect(progress).not.toHaveAttribute('aria-valuenow')
    expect(document.activeElement).toBe(screen.getByTestId('studio-data-migration-phase'))
    act(() => {
      onEventCb?.({
        type: 'studio_data_migrate_progress',
        done_files: 21,
        total_files: 42,
        done_bytes: 2.5 * 1024 * 1024,
        total_bytes: 5 * 1024 * 1024,
        current_file: 'projects/example.json',
      })
    })
    expect(progress).toHaveAttribute('aria-valuenow', '50')
    expect(progress).toHaveAttribute('aria-valuetext', expect.stringContaining('50%'))
    // running 态：没有关闭操作，Escape / backdrop 也由 Modal 禁用
    expect(screen.queryByRole('button', { name: '关闭' })).not.toBeInTheDocument()
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
    expect(screen.getByRole('button', { name: '关闭' })).toBeInTheDocument()
  })
})
