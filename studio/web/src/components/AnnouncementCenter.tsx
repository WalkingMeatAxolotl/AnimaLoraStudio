// AnnouncementCenter —— 公告栏弹窗（announcement-center Phase 1）。
//
// 游戏式：左 list（tag 过滤 + 未读红点）/ 右正文。开关 / read 状态 / 更新检查
// 由 lib/Announcements 的 context 持有；Topbar 铃铛点击 → openCenter()。
// 正文暂用 whitespace-pre-wrap 显示 markdown 原文（沿用现有 release notes detail
// 约定，依赖最少）；真 markdown 渲染留作后续（需引 react-markdown 依赖，单独定）。
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { AnnouncementPost } from '../api/client'
import { useAnnouncements } from '../lib/Announcements'
import { useSettingsDrawer } from '../lib/SettingsDrawer'

const TAG_ORDER: AnnouncementPost['tag'][] = ['release', 'notice', 'migration']

function tagChipClass(tag: AnnouncementPost['tag']): string {
  switch (tag) {
    case 'release': return 'text-accent bg-accent-soft'
    case 'migration': return 'text-warn bg-warn-soft'
    default: return 'text-info bg-info-soft' // notice
  }
}

export function AnnouncementCenter() {
  const { t, i18n } = useTranslation()
  const { posts, readIds, open, closeCenter, markRead, updateInfo } = useAnnouncements()
  const settingsDrawer = useSettingsDrawer()
  const [activeTag, setActiveTag] = useState<'all' | AnnouncementPost['tag']>('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const lang: 'zh' | 'en' = i18n.language.toLowerCase().startsWith('zh') ? 'zh' : 'en'

  const filtered = useMemo(
    () => (activeTag === 'all' ? posts : posts.filter((p) => p.tag === activeTag)),
    [posts, activeTag],
  )
  const tagsPresent = useMemo(
    () => TAG_ORDER.filter((tg) => posts.some((p) => p.tag === tg)),
    [posts],
  )

  // 打开 / 切换过滤后，默认选中第一篇并标记已读。
  useEffect(() => {
    if (!open) return
    if (filtered.length === 0) { setSelectedId(null); return }
    if (selectedId === null || !filtered.some((p) => p.id === selectedId)) {
      setSelectedId(filtered[0].id)
      markRead(filtered[0].id)
    }
  }, [open, filtered, selectedId, markRead])

  if (!open) return null

  const selected = posts.find((p) => p.id === selectedId) ?? null
  const select = (p: AnnouncementPost) => { setSelectedId(p.id); markRead(p.id) }
  const tagLabel = (tg: 'all' | AnnouncementPost['tag']) => t(`announcements.tags.${tg}`)

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 p-4"
      onClick={closeCenter}
      data-testid="announcement-center"
    >
      {/* 点蒙版退出（公告栏不是 onboarding 那种不可打断）；点面板内部不冒泡 */}
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="announcement-center-title"
        onClick={(e) => e.stopPropagation()}
        /* 宽度：居中弹窗，大屏最宽 1080px、小屏 92vw 自适应（左 list 256px 固定，
           其余给正文）。借鉴 SettingsDrawer 的响应式思路但居中、不占满。 */
        className="w-[min(1080px,92vw)] h-[78vh] flex flex-col bg-elevated border border-dim rounded-lg shadow-xl overflow-hidden"
      >
        {/* header */}
        <div className="flex items-center gap-3 px-5 py-3 border-b border-dim shrink-0">
          <h1 id="announcement-center-title" className="m-0 text-lg font-semibold text-fg-primary flex-1">
            {t('announcements.title')}
          </h1>
          {updateInfo?.has_update && (
            <button
              type="button"
              onClick={() => { settingsDrawer.open({ section: 'version' }); closeCenter() }}
              title={t('announcements.updateAvailable', { tag: updateInfo.latest_tag ?? updateInfo.latest_commit.slice(0, 8) })}
              className="flex items-center gap-1.5 px-2 py-[5px] rounded-md text-xs font-mono text-accent bg-accent-soft border border-accent cursor-pointer hover:bg-accent/10 transition-colors shrink-0"
              data-testid="announcement-update-btn"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-accent" />
              <span>{updateInfo.latest_tag ?? t('announcements.updateAvailable', { tag: '' }).trim()}</span>
            </button>
          )}
          <button
            type="button"
            onClick={closeCenter}
            aria-label={t('announcements.close')}
            className="flex items-center justify-center w-7 h-7 rounded-md text-fg-tertiary hover:text-fg-primary hover:bg-surface bg-transparent border-none cursor-pointer shrink-0"
            data-testid="announcement-close"
          >
            ✕
          </button>
        </div>

        {/* tag filter */}
        {tagsPresent.length > 1 && (
          <div className="flex gap-1.5 px-5 py-2 border-b border-dim shrink-0">
            {(['all', ...tagsPresent] as const).map((tg) => (
              <button
                key={tg}
                type="button"
                onClick={() => setActiveTag(tg)}
                className={`px-2.5 py-1 text-xs rounded transition-colors cursor-pointer border ${
                  activeTag === tg
                    ? 'border-accent bg-accent-soft text-fg-primary'
                    : 'border-dim bg-surface text-fg-secondary hover:border-accent/50'
                }`}
                data-testid={`announcement-filter-${tg}`}
              >
                {tagLabel(tg)}
              </button>
            ))}
          </div>
        )}

        {/* master-detail */}
        <div className="flex-1 flex min-h-0">
          <ul className="w-64 shrink-0 overflow-y-auto border-r border-dim m-0 p-0 list-none">
            {filtered.length === 0 && (
              <li className="px-4 py-6 text-sm text-fg-tertiary">{t('announcements.empty')}</li>
            )}
            {filtered.map((p) => {
              const unread = !readIds.has(p.id)
              const isSel = p.id === selectedId
              return (
                <li key={p.id}>
                  <button
                    type="button"
                    onClick={() => select(p)}
                    className={`w-full text-left px-4 py-3 bg-transparent border-none border-b border-subtle cursor-pointer hover:bg-surface transition-colors ${
                      isSel ? 'bg-surface' : ''
                    }`}
                    data-testid={`announcement-item-${p.id}`}
                  >
                    <div className="flex items-center gap-2">
                      {unread && (
                        <span
                          className="w-2 h-2 rounded-full bg-err shrink-0"
                          data-testid={`announcement-dot-${p.id}`}
                        />
                      )}
                      <span className="text-sm font-medium text-fg-primary truncate">{p.title[lang]}</span>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`px-1.5 py-0.5 text-[10px] rounded ${tagChipClass(p.tag)}`}>
                        {tagLabel(p.tag)}
                      </span>
                      <span className="text-xs text-fg-tertiary">{p.date}</span>
                    </div>
                  </button>
                </li>
              )
            })}
          </ul>

          <div className="flex-1 overflow-y-auto px-6 py-5">
            {selected ? (
              <>
                <h2 className="m-0 text-xl font-semibold text-fg-primary">{selected.title[lang]}</h2>
                <div className="mt-1 text-xs text-fg-tertiary">
                  {selected.date}{selected.version ? ` · v${selected.version}` : ''}
                </div>
                <div className="mt-4 text-sm text-fg-secondary whitespace-pre-wrap leading-relaxed">
                  {selected.body[lang]}
                </div>
              </>
            ) : (
              <div className="text-sm text-fg-tertiary">{t('announcements.empty')}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
