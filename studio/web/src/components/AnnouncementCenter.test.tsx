import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type AnnouncementPost } from '../api/client'
import { AnnouncementsProvider } from '../lib/Announcements'
import {
  AnnouncementCenter,
  announcementUrlTransform,
  parseSettingsLink,
} from './AnnouncementCenter'

// 公告栏的更新按钮 + 正文应用内链接要 useSettingsDrawer；测试里 stub 掉，省去
// provider 嵌套。openSpy 用来断言深链真的打开了对应 section。
const openSpy = vi.fn()
vi.mock('../lib/SettingsDrawer', () => ({
  useSettingsDrawer: () => ({ open: openSpy }),
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
    // 默认选中第一篇（pin 的 migration）→ 已读、无红点。标记已读是选中后的 effect，
    // 比 modal 出现晚一个 tick，用 waitFor 等状态稳定（同步断言在慢机/CI 上会 flake）。
    await waitFor(() =>
      expect(screen.queryByTestId('announcement-dot-p-migration')).toBeNull())
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
    // 正文切换与红点消失不同 tick → waitFor 等正文到位。曾两次 CI flake
    // （#349 补 waitFor、后又超时）：真因是组件默认选中 effect 的 stale closure
    // 会覆盖点击选中（已在 AnnouncementCenter 内改函数式更新修复），非等待不够长。
    await waitFor(() =>
      expect(screen.getByText('公告正文')).toBeInTheDocument())
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

  it('点蒙版退出；点面板内部不退出', async () => {
    localStorage.setItem('studio.announcements.lastVersion', '0.15.0')
    renderCenter()
    await waitFor(() =>
      expect(screen.getByTestId('announcement-center')).toBeInTheDocument())
    // 点面板内部（某篇）→ 不关
    fireEvent.click(screen.getByTestId('announcement-item-p-notice'))
    expect(screen.getByTestId('announcement-center')).toBeInTheDocument()
    // 点蒙版（外层）→ 关
    fireEvent.click(screen.getByTestId('announcement-center'))
    await waitFor(() =>
      expect(screen.queryByTestId('announcement-center')).toBeNull())
  })

  it('正文按 markdown 渲染（### → 标题、- → 列表，而非原文）', async () => {
    vi.spyOn(api, 'getAnnouncements').mockResolvedValue([
      { id: 'md', date: '2026-06-28', tag: 'release', pin: false, version: '9.0.0',
        title: { zh: 'v9 更新', en: 'v9 release' },
        body: { zh: '### 新增\n\n- 甲项\n- 乙项', en: '### Added\n\n- a' } },
    ])
    localStorage.setItem('studio.announcements.lastVersion', '0.15.0')
    renderCenter()
    await waitFor(() =>
      expect(screen.getByTestId('announcement-center')).toBeInTheDocument())
    // ### 解析成标题（原文 "### 新增" 不会有 heading role）
    expect(screen.getByRole('heading', { name: '新增' })).toBeInTheDocument()
    // - 解析成列表项
    expect(screen.getByText('甲项')).toBeInTheDocument()
  })

  // ── 应用内深链（迁移类公告靠它把用户直接送到工具那一屏）────────────────
  describe('parseSettingsLink', () => {
    it('普通外链返回 null（照旧新标签页打开）', () => {
      expect(parseSettingsLink('https://example.com')).toBeNull()
      expect(parseSettingsLink('#anchor')).toBeNull()
      expect(parseSettingsLink(undefined)).toBeNull()
    })

    it('已注册的 section → 带 section 跳转', () => {
      expect(parseSettingsLink('app://settings/eval-metrics'))
        .toEqual({ section: 'eval-metrics' })
    })

    it('未注册 / 已随工具退役的 section → 只开设置，不静默失效', () => {
      expect(parseSettingsLink('app://settings/gone-with-the-tool'))
        .toEqual({ section: null })
      expect(parseSettingsLink('app://settings/')).toEqual({ section: null })
    })
  })

  describe('announcementUrlTransform', () => {
    it('放行设置协议', () => {
      expect(announcementUrlTransform('app://settings/eval-metrics'))
        .toBe('app://settings/eval-metrics')
    })

    it('普通 URL 原样通过', () => {
      expect(announcementUrlTransform('https://example.com/a?b=1'))
        .toBe('https://example.com/a?b=1')
      expect(announcementUrlTransform('#section')).toBe('#section')
    })

    it('危险协议仍被上游默认净化剥掉（放行设置协议没有削弱这层）', () => {
      expect(announcementUrlTransform('javascript:alert(1)')).toBe('')
      expect(announcementUrlTransform('data:text/html,<script>')).toBe('')
    })
  })

  it('正文里的 app:// 链接走应用内跳转并关掉公告栏', async () => {
    vi.spyOn(api, 'getAnnouncements').mockResolvedValue([
      { id: 'mig', date: '2026-07-24', tag: 'migration', pin: false, version: '9.0.0',
        title: { zh: '清理残留', en: 'Clean up' },
        body: {
          zh: '[去清理](app://settings/eval-metrics)',
          en: '[Clean up](app://settings/eval-metrics)',
        } },
    ])
    localStorage.setItem('studio.announcements.lastVersion', '0.15.0')
    renderCenter()
    await waitFor(() =>
      expect(screen.getByTestId('announcement-center')).toBeInTheDocument())

    fireEvent.click(screen.getByText('去清理'))

    expect(openSpy).toHaveBeenCalledWith({ section: 'eval-metrics' })
    // 跳转后公告栏让位给设置抽屉，不叠两层 modal
    await waitFor(() =>
      expect(screen.queryByTestId('announcement-center')).toBeNull())
  })
})
