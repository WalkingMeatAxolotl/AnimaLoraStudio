import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { api, type CaptionEntry, type PresetSummary, type ProjectSummary } from '../api/client'
import { useProjectCtx } from '../context/ProjectContext'
import { useSettingsDrawer } from '../lib/SettingsDrawer'

interface Item {
  id: string
  label: string
  sub?: string
  group: string
  /** 路由跳转。跟 action 二选一。 */
  path?: string
  /** 自定义动作（如打开抽屉）。优先于 path。 */
  action?: () => void
}

const SEARCH_ICON = (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" />
  </svg>
)

const ENTER_ICON = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 10 4 15 9 20" /><path d="M20 4v7a4 4 0 0 1-4 4H4" />
  </svg>
)

const TAG_ICON = (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2H2v10l9.29 9.29a1 1 0 0 0 1.41 0l7.3-7.3a1 1 0 0 0 0-1.41L12 2z" />
    <path d="M7 7h.01" />
  </svg>
)

interface Props {
  open: boolean
  onClose: () => void
  /** 锚点元素，面板将定位在其下方右对齐 */
  anchorEl?: HTMLElement | null
}

export default function CommandPalette({ open, onClose, anchorEl }: Props) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const ctx = useProjectCtx()
  const settingsDrawer = useSettingsDrawer()
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const [query, setQuery] = useState('')
  const [activeIdx, setActiveIdx] = useState(0)

  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [projectsLoaded, setProjectsLoaded] = useState(false)

  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [presetsLoaded, setPresetsLoaded] = useState(false)

  const [captions, setCaptions] = useState<CaptionEntry[]>([])
  const [captionsCacheKey, setCaptionsCacheKey] = useState<string | null>(null)
  const [captionsLoading, setCaptionsLoading] = useState(false)

  const [panelStyle, setPanelStyle] = useState<React.CSSProperties>({
    position: 'fixed', top: 56, right: 20, width: 520,
  })

  useLayoutEffect(() => {
    if (!open) return
    if (anchorEl) {
      const r = anchorEl.getBoundingClientRect()
      setPanelStyle({
        position: 'fixed',
        top: r.bottom + 4,
        right: window.innerWidth - r.right,
        width: Math.max(520, r.width + 200),
      })
    } else {
      setPanelStyle({ position: 'fixed', top: 56, right: 20, width: 520 })
    }
  }, [open, anchorEl])

  useEffect(() => {
    if (open) {
      setQuery('')
      setActiveIdx(0)
      setTimeout(() => inputRef.current?.focus(), 40)
    } else {
      setProjectsLoaded(false)
      setPresetsLoaded(false)
    }
  }, [open])

  useEffect(() => {
    if (!open || projectsLoaded) return
    let cancelled = false
    api.listProjects().then((items) => {
      if (!cancelled) { setProjects(items); setProjectsLoaded(true) }
    }).catch(() => {
      if (!cancelled) setProjectsLoaded(true)
    })
    return () => { cancelled = true }
  }, [open, projectsLoaded])

  useEffect(() => {
    if (!open || presetsLoaded) return
    let cancelled = false
    api.listPresets().then((items) => {
      if (!cancelled) { setPresets(items); setPresetsLoaded(true) }
    }).catch(() => {
      if (!cancelled) setPresetsLoaded(true)
    })
    return () => { cancelled = true }
  }, [open, presetsLoaded])

  const pid = ctx?.project?.id
  const vid = ctx?.activeVersion?.id
  const queryEnoughForTags = query.length >= 2

  useEffect(() => {
    if (!open || !queryEnoughForTags || !pid || !vid) return
    const key = `${pid}:${vid}`
    if (captionsCacheKey === key) return

    let cancelled = false
    setCaptionsLoading(true)
    api.listCaptionsFull(pid, vid).then((result) => {
      if (!cancelled) {
        setCaptions(result.items)
        setCaptionsCacheKey(key)
        setCaptionsLoading(false)
      }
    }).catch(() => {
      if (!cancelled) setCaptionsLoading(false)
    })
    return () => { cancelled = true }
  }, [open, queryEnoughForTags, pid, vid, captionsCacheKey])

  const allItems = useMemo<Item[]>(() => {
    const items: Item[] = []

    items.push({ id: 'home',     label: t('commandPalette.home'),     sub: t('commandPalette.homeSub'),     group: t('commandPalette.pages'), path: '/' })
    items.push({ id: 'queue',    label: t('nav.queue'),               sub: t('commandPalette.queueSub'),    group: t('commandPalette.pages'), path: '/queue' })
    items.push({ id: 'presets',  label: t('nav.presets'),             sub: t('commandPalette.presetsSub'), group: t('commandPalette.pages'), path: '/tools/presets' })
    items.push({ id: 'monitor',  label: t('nav.monitor'),             sub: t('commandPalette.monitorSub'), group: t('commandPalette.pages'), path: '/tools/monitor' })
    items.push({ id: 'settings', label: t('nav.settings'),            sub: t('commandPalette.settingsSub'), group: t('commandPalette.pages'), action: () => settingsDrawer.open() })

    for (const p of presets) {
      items.push({
        id: `preset:${p.name}`,
        label: p.name,
        sub: t('commandPalette.presetItem'),
        group: t('commandPalette.presets'),
        path: '/tools/presets',
      })
    }

    for (const p of projects) {
      items.push({
        id: `project:${p.id}`,
        label: p.title || `#${p.id}`,
        sub: p.slug ? `/${p.slug}` : t('commandPalette.projectItem', { id: p.id }),
        group: t('commandPalette.projects'),
        path: `/projects/${p.id}`,
      })
    }

    if (ctx) {
      const cpid = ctx.project.id
      const cvid = ctx.activeVersion?.id
      const group = t('commandPalette.currentProject')
      items.push({ id: `overview:${cpid}`, label: t('nav.overview'), sub: ctx.project.title, group, path: `/projects/${cpid}` })
      items.push({ id: `download:${cpid}`, label: t('nav.download'), sub: ctx.project.title, group, path: `/projects/${cpid}/download` })
      if (cvid) {
        const base = `/projects/${cpid}/v/${cvid}`
        items.push({ id: `curate:${cpid}`, label: t('nav.curate'),   sub: ctx.project.title, group, path: `${base}/curate` })
        items.push({ id: `tag:${cpid}`,    label: t('nav.tag'),      sub: ctx.project.title, group, path: `${base}/tag` })
        items.push({ id: `edit:${cpid}`,   label: t('nav.tagEdit'),  sub: ctx.project.title, group, path: `${base}/edit` })
        items.push({ id: `reg:${cpid}`,    label: t('nav.reg'),      sub: ctx.project.title, group, path: `${base}/reg` })
        items.push({ id: `train:${cpid}`,  label: t('nav.train'),    sub: ctx.project.title, group, path: `${base}/train` })
      }
    }

    return items
  }, [projects, presets, ctx, t, settingsDrawer])

  const filteredNav = useMemo(() => {
    if (!query.trim()) return allItems
    const q = query.toLowerCase()
    return allItems.filter(
      (item) =>
        item.label.toLowerCase().includes(q) ||
        (item.sub ?? '').toLowerCase().includes(q) ||
        item.group.toLowerCase().includes(q),
    )
  }, [allItems, query])

  const tagItems = useMemo<Item[]>(() => {
    if (!queryEnoughForTags || !ctx?.activeVersion || captions.length === 0) return []
    const cpid = ctx.project.id
    const cvid = ctx.activeVersion.id
    const q = query.toLowerCase()
    const tagCounts = new Map<string, number>()

    for (const c of captions) {
      for (const tag of c.tags) {
        if (tag.toLowerCase().includes(q)) {
          tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1)
        }
      }
    }

    return Array.from(tagCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([tag, count]) => ({
        id: `tag:${tag}`,
        label: tag,
        sub: t('commandPalette.imageCount', { n: count }),
        group: t('commandPalette.tags'),
        path: `/projects/${cpid}/v/${cvid}/edit`,
      }))
  }, [captions, queryEnoughForTags, query, ctx, t])

  const filtered = useMemo(() => [...filteredNav, ...tagItems], [filteredNav, tagItems])

  const grouped = useMemo(() => {
    const map = new Map<string, Item[]>()
    for (const item of filtered) {
      if (!map.has(item.group)) map.set(item.group, [])
      map.get(item.group)!.push(item)
    }
    return map
  }, [filtered])

  const flatItems = useMemo(() => {
    const out: Item[] = []
    for (const [, items] of grouped) out.push(...items)
    return out
  }, [grouped])

  const select = useCallback(
    (item: Item) => {
      if (item.action) item.action()
      else if (item.path) navigate(item.path)
      onClose()
    },
    [navigate, onClose],
  )

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIdx((i) => Math.min(i + 1, flatItems.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIdx((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (flatItems[activeIdx]) select(flatItems[activeIdx])
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    const active = el.querySelector(`[data-palette-idx="${activeIdx}"]`) as HTMLElement | null
    if (active) active.scrollIntoView({ block: 'nearest' })
  }, [activeIdx])

  if (!open) return null

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />

      <div
        className="z-50 rounded-lg border border-subtle bg-overlay shadow-xl flex flex-col overflow-hidden"
        style={{
          ...panelStyle,
          maxHeight: 'min(60vh, 520px)',
          maxWidth: 'calc(100vw - 40px)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 px-4 py-3 border-b border-subtle">
          <span className="text-fg-tertiary">{SEARCH_ICON}</span>
          <input
            ref={inputRef}
            type="text"
            className="flex-1 bg-transparent border-none outline-none text-sm text-fg-primary placeholder:text-fg-tertiary"
            placeholder={t('commandPalette.placeholder')}
            value={query}
            onChange={(e) => { setQuery(e.target.value); setActiveIdx(0) }}
            onKeyDown={handleKeyDown}
          />
          {captionsLoading && (
            <span className="text-2xs text-fg-tertiary animate-pulse">{t('commandPalette.searchingTags')}</span>
          )}
          <kbd className="kbd">esc</kbd>
        </div>

        <div ref={listRef} className="flex-1 overflow-y-auto p-1.5">
          {filtered.length === 0 ? (
            <div className="text-sm text-fg-tertiary text-center py-8">{t('commandPalette.noResults')}</div>
          ) : (
            [...grouped.entries()].map(([group, items]) => (
              <div key={group} className="mb-1">
                <div className="flex items-center gap-1.5 px-3 py-1.5">
                  {group === t('commandPalette.tags') && <span className="text-fg-tertiary">{TAG_ICON}</span>}
                  <span className="text-2xs text-fg-tertiary font-semibold uppercase tracking-wider">
                    {group}
                  </span>
                </div>
                {items.map((item) => {
                  const idx = flatItems.indexOf(item)
                  const isActive = idx === activeIdx
                  return (
                    <button
                      key={item.id}
                      data-palette-idx={idx}
                      onClick={() => select(item)}
                      className={`w-full flex items-center gap-3 px-3 py-2 rounded-sm text-left border-none cursor-pointer transition-colors ${
                        isActive ? 'bg-accent-soft text-accent' : 'bg-transparent text-fg-primary hover:bg-surface'
                      }`}
                    >
                      <span className="text-sm flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
                        {item.label}
                      </span>
                      {item.sub && (
                        <span className={`text-xs overflow-hidden text-ellipsis whitespace-nowrap max-w-[160px] ${
                          isActive ? 'text-accent/70' : 'text-fg-tertiary'
                        }`}>
                          {item.sub}
                        </span>
                      )}
                      {isActive && <span className="text-fg-tertiary shrink-0">{ENTER_ICON}</span>}
                    </button>
                  )
                })}
              </div>
            ))
          )}
        </div>

        <div className="flex items-center gap-4 px-4 py-2 border-t border-subtle text-2xs text-fg-tertiary">
          <span className="flex items-center gap-1"><kbd className="kbd">↑↓</kbd> {t('commandPalette.navigate')}</span>
          <span className="flex items-center gap-1"><kbd className="kbd">enter</kbd> {t('commandPalette.select')}</span>
          <span className="flex items-center gap-1"><kbd className="kbd">esc</kbd> {t('commandPalette.close')}</span>
          {queryEnoughForTags && ctx?.activeVersion && (
            <span className="ml-auto opacity-70">{t('commandPalette.searchTagsHint', { n: captions.length })}</span>
          )}
        </div>
      </div>
    </>
  )
}
