// AnnouncementCenter —— 公告栏弹窗（announcement-center Phase 1）。
//
// 游戏式：左 list（tag 过滤 + 未读红点）/ 右正文。开关 / read 状态 / 更新检查
// 由 lib/Announcements 的 context 持有；Topbar 铃铛点击 → openCenter()。
// 正文用 react-markdown + remark-gfm 渲染；元素经 components 映射到现有 Tailwind
// token（标题/列表/链接/代码），不另起 CSS、不猜 CSS 变量名。
import type { ComponentPropsWithoutRef } from 'react'
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { AnnouncementPost } from '../api/client'
import { useAnnouncements } from '../lib/Announcements'
import { useSettingsDrawer } from '../lib/SettingsDrawer'

const TAG_ORDER: AnnouncementPost['tag'][] = ['release', 'notice', 'migration']

// markdown 元素 → Tailwind class（公告正文用，复用 modal 既有 token）。
type MdProps<T extends keyof React.JSX.IntrinsicElements> = ComponentPropsWithoutRef<T>
const MD_COMPONENTS = {
  // # / ## 罕见（正文最高用 ###）；都按版块标题处理
  h1: (p: MdProps<'h1'>) => <h3 className="mt-5 mb-2 pb-1 text-[15px] font-bold text-fg-primary border-b border-dim" {...p} />,
  h2: (p: MdProps<'h2'>) => <h3 className="mt-5 mb-2 pb-1 text-[15px] font-bold text-fg-primary border-b border-dim" {...p} />,
  // ### = 分组标题（新增/变更/改进/修复…）：加粗 + 下划线，清晰分段
  h3: (p: MdProps<'h3'>) => <h4 className="mt-5 mb-2 pb-1 text-sm font-bold text-fg-primary border-b border-dim first:mt-1" {...p} />,
  h4: (p: MdProps<'h4'>) => <h4 className="mt-4 mb-1.5 text-sm font-semibold text-fg-primary" {...p} />,
  p: (p: MdProps<'p'>) => <p className="my-2 leading-7" {...p} />,
  // 要点首句加粗 → 用主色，跟正文（次色）拉开
  strong: (p: MdProps<'strong'>) => <strong className="font-semibold text-fg-primary" {...p} />,
  ul: (p: MdProps<'ul'>) => <ul className="my-2 pl-5 list-disc space-y-2 marker:text-fg-tertiary" {...p} />,
  ol: (p: MdProps<'ol'>) => <ol className="my-2 pl-5 list-decimal space-y-2 marker:text-fg-tertiary" {...p} />,
  li: (p: MdProps<'li'>) => <li className="leading-7" {...p} />,
  a: (p: MdProps<'a'>) => <a className="text-accent underline hover:opacity-80" target="_blank" rel="noreferrer" {...p} />,
  code: (p: MdProps<'code'>) => <code className="rounded bg-surface border border-dim px-1.5 py-0.5 text-[0.85em] font-mono text-fg-primary" {...p} />,
  hr: () => <hr className="my-4 border-dim" />,
  blockquote: (p: MdProps<'blockquote'>) => <blockquote className="my-2 pl-3 border-l-2 border-dim text-fg-tertiary" {...p} />,
} as const

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

  // 打开 / 切换过滤后，默认选中第一篇。必须用函数式更新读「当前」选中值：
  // 旧实现直接读闭包里的 selectedId，慢机上用户点击（setSelectedId）可能发生在
  // 本 effect 的 passive flush 之前，effect 随后带着 stale 的 selectedId=null 跑，
  // 把用户刚点的选中覆盖回第一篇——红点已消（markRead 生效）但正文永远不切换
  // （#349 / #354 两次 CI flake 的真因，测试侧加 waitFor 等不来）。
  useEffect(() => {
    if (!open) return
    if (filtered.length === 0) { setSelectedId(null); return }
    setSelectedId((cur) =>
      cur !== null && filtered.some((p) => p.id === cur) ? cur : filtered[0].id)
  }, [open, filtered])

  // 选中即已读——默认选中和点击选中共用这一条路径。
  useEffect(() => {
    if (selectedId !== null) markRead(selectedId)
  }, [selectedId, markRead])

  if (!open) return null

  const selected = posts.find((p) => p.id === selectedId) ?? null
  const select = (p: AnnouncementPost) => setSelectedId(p.id)
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
        /* 宽度：居中弹窗，大屏最宽 1440px、小屏 80vw 自适应（左 list 256px 固定，
           其余给正文）。借鉴 SettingsDrawer 的响应式思路但居中、不占满。 */
        className="w-[min(1440px,80vw)] h-[78vh] flex flex-col bg-elevated border border-dim rounded-lg shadow-xl overflow-hidden"
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
                <div className="mt-4 text-sm text-fg-secondary [&_ul_ul]:list-[circle] [&_li_p]:my-1">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                    {selected.body[lang]}
                  </ReactMarkdown>
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
