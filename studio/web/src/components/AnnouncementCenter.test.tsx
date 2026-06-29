import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type AnnouncementPost } from '../api/client'
import { AnnouncementsProvider } from '../lib/Announcements'
import { AnnouncementCenter } from './AnnouncementCenter'

// 公告栏的更新按钮要 useSettingsDrawer；测试里 stub 掉，省去 provider 嵌套。
vi.mock('../lib/SettingsDrawer', () => ({
  useSettingsDrawer: () => ({ open: () => {} }),
}))

const POSTS: AnnouncementPost[] = [
  { id: 'p-migration', date: '2026-06-28', tag: 'migration', pin: true, version: '0.16.0',
    title: { zh: '迁移标题', en: 'Migration title' }, body: { zh: '迁移正文', en: 'Migration body' } },
  { id: 'p-notice', date: '2026-06-20', tag: 'notice', pin: false, version: null,
    title: { zh: '公告标题', en: 'Notice title' }, body: { zh: '公告正文', en: 'Notice body' } },
]

function renderCenter() {
  return render(
    <AnnouncementsProvider>
      <AnnouncementCenter />
    </AnnouncementsProvider>,
  )
}

describe('AnnouncementCenter', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.spyOn(api, 'getAnnouncements').mockResolvedValue(POSTS)
    vi.spyOn(api, 'health').mockResolvedValue({ version: '0.16.0' } as never)
    vi.spyOn(api, 'checkSystemUpdate').mockResolvedValue({ has_update: false } as never)
  })
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('首次安装：不自动弹，且当前全部标记已读', async () => {
    renderCenter()
    await waitFor(() =>
      expect(localStorage.getItem('studio.announcements.lastVersion')).toBe('0.16.0'))
    expect(screen.queryByTestId('announcement-center')).toBeNull()
    const read = JSON.parse(localStorage.getItem('studio.announcements.read') ?? '[]')
    expect(new Set(read)).toEqual(new Set(['p-migration', 'p-notice']))
  })

  it('版本变化 + 有未读 → 自动弹；pin 篇选中已读、其余有红点', async () => {
    localStorage.setItem('studio.announcements.lastVersion', '0.15.0')
    localStorage.setItem('studio.announcements.read', '[]')
    renderCenter()
    await waitFor(() =>
      expect(screen.getByTestId('announcement-center')).toBeInTheDocument())
    // 默认选中第一篇（pin 的 migration）→ 已读、无红点
    expect(screen.queryByTestId('announcement-dot-p-migration')).toBeNull()
    // notice 仍未读 → 有红点
    expect(screen.getByTestId('announcement-dot-p-notice')).toBeInTheDocument()
    // 正文显示选中篇的中文正文
    expect(screen.getByText('迁移正文')).toBeInTheDocument()
  })

  it('点另一篇 → 标记已读，红点消失，正文切换', async () => {
    localStorage.setItem('studio.announcements.lastVersion', '0.15.0')
    renderCenter()
    await waitFor(() =>
      expect(screen.getByTestId('announcement-center')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('announcement-item-p-notice'))
    await waitFor(() =>
      expect(screen.queryByTestId('announcement-dot-p-notice')).toBeNull())
    expect(screen.getByText('公告正文')).toBeInTheDocument()
  })

  it('tag 过滤只显示该类', async () => {
    localStorage.setItem('studio.announcements.lastVersion', '0.15.0')
    renderCenter()
    await waitFor(() =>
      expect(screen.getByTestId('announcement-center')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('announcement-filter-notice'))
    expect(screen.queryByTestId('announcement-item-p-migration')).toBeNull()
    expect(screen.getByTestId('announcement-item-p-notice')).toBeInTheDocument()
  })
})
