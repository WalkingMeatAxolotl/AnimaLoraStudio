import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useBlocker, useOutletContext } from 'react-router-dom'
import {
  api,
  type CommitItem,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import BulkActionBar from '../../../components/BulkActionBar'
import { useDialog } from '../../../components/Dialog'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import PaneResizer from '../../../components/PaneResizer'
import SaveBar from '../../../components/SaveBar'
import StepShell from '../../../components/StepShell'
import TagEditor from '../../../components/TagEditor'
import TagStatsPanel from '../../../components/TagStatsPanel'
import { useToast } from '../../../components/Toast'
import ZoomableImage from '../../../components/ZoomableImage'
import { compareImagePath } from '../../../lib/imageSort'
import { useEventStream } from '../../../lib/useEventStream'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  setVersionSwitchGuard: (g: (() => Promise<boolean>) | null) => void
}

const keyOf = (folder: string, name: string) => `${folder}/${name}`

interface CaptionMeta {
  folder: string
  name: string
  format: 'txt' | 'json' | 'none'
}

function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

export default function TagEditPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload, setVersionSwitchGuard } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const versionId = activeVersion?.id ?? null

  const [cache, setCache] = useState<Map<string, string[]>>(new Map())
  const [initial, setInitial] = useState<Map<string, string[]>>(new Map())
  const [meta, setMeta] = useState<Map<string, CaptionMeta>>(new Map())
  const [keys, setKeys] = useState<string[]>([])

  const [activeKey, setActiveKey] = useState<string>('')
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  // '' = 全部；否则限定到该 folder（1_data / 2_data ...）。命名特意区分于下面
  // editing 时用的 `activeFolder`（那个是当前编辑图所在 folder，纯展示）。
  const [folderFilter, setFolderFilter] = useState<string>('')

  // 左栏（图片网格）/ 右栏（标签工作区）各占整行宽度的百分比，两条分隔条可拖；
  // 中间预览栏 flex-1 吃掉剩余空间。默认 40 / 32 = 改造前写死的 flex 1.5 : 1 : 32%。
  const rowRef = useRef<HTMLDivElement>(null)
  const [gridPct, setGridPct] = useLocalStorageState('studio:tagEdit:grid_pct', 40)
  const [sidePct, setSidePct] = useLocalStorageState('studio:tagEdit:side_pct', 32)

  // 两栏共享同一条宽度预算，各自的上限得看对方吃掉多少，否则中间预览栏会被挤没。
  const MIN_PREVIEW_PCT = 15
  const gridMax = Math.max(15, 100 - sidePct - MIN_PREVIEW_PCT)
  const sideMax = Math.max(20, 100 - gridPct - MIN_PREVIEW_PCT)

  const reloadCache = useCallback(async () => {
    if (versionId == null) return
    try {
      const r = await api.listCaptionsFull(project.id, versionId)
      const c = new Map<string, string[]>()
      const m = new Map<string, CaptionMeta>()
      const ks: string[] = []
      const sorted = [...r.items].sort((a, b) =>
        compareImagePath(keyOf(a.folder, a.name), keyOf(b.folder, b.name))
      )
      for (const it of sorted) {
        const k = keyOf(it.folder, it.name)
        c.set(k, it.tags)
        m.set(k, { folder: it.folder, name: it.name, format: it.format })
        ks.push(k)
      }
      setCache(c); setInitial(new Map(c)); setMeta(m); setKeys(ks)
    } catch (e) { toast(String(e), 'error') }
  }, [project.id, versionId, toast])

  useEffect(() => { void reloadCache() }, [reloadCache])

  useEventStream((evt) => {
    if (
      evt.type === 'version_state_changed' &&
      versionId != null &&
      evt.version_id === versionId
    ) {
      void reloadCache(); void reload()
    } else if (
      evt.type === 'job_state_changed' &&
      evt.project_id === project.id &&
      (evt.status === 'done' || evt.status === 'failed')
    ) {
      void reloadCache(); void reload()
    }
  })

  const dirtyKeys = useMemo(() => {
    const out: string[] = []
    for (const k of keys) {
      const cur = cache.get(k) ?? []
      const ini = initial.get(k) ?? []
      if (!arraysEqual(cur, ini)) out.push(k)
    }
    return out
  }, [cache, initial, keys])
  const dirty = dirtyKeys.length > 0

  useEffect(() => {
    if (!dirty) return
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = '' }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [dirty])

  // 应用内 React-Router 导航不触发 beforeunload，得靠 useBlocker（v6.4+）。
  // dirty=false 时 blocker 自动放行；dirty=true 时拦下导航 → confirm 弹窗
  // → 用户选"放弃"调 proceed()、"留下"调 reset()。
  const blocker = useBlocker(dirty)
  useEffect(() => {
    if (blocker.state !== 'blocked') return
    let cancelled = false
    void confirm(
      t('tagEdit.unsavedConfirmMessage', { n: dirtyKeys.length }),
      {
        tone: 'danger',
        title: t('tagEdit.unsavedConfirmTitle'),
        okText: t('tagEdit.unsavedConfirmDiscard'),
        cancelText: t('tagEdit.unsavedConfirmStay'),
      },
    ).then((ok) => {
      if (cancelled) return
      if (ok) blocker.proceed?.()
      else blocker.reset?.()
    })
    return () => { cancelled = true }
  }, [blocker, confirm, t, dirtyKeys.length])

  // 切版本会重挂载本页（Layout 的 Outlet key），不走路由导航、useBlocker 拦不住；
  // dirty 时注册切换守卫，复用同一套 confirm 文案。
  useEffect(() => {
    if (!dirty) return
    setVersionSwitchGuard(() =>
      confirm(
        t('tagEdit.unsavedConfirmMessage', { n: dirtyKeys.length }),
        {
          tone: 'danger',
          title: t('tagEdit.unsavedConfirmTitle'),
          okText: t('tagEdit.unsavedConfirmDiscard'),
          cancelText: t('tagEdit.unsavedConfirmStay'),
        },
      )
    )
    return () => setVersionSwitchGuard(null)
  }, [dirty, dirtyKeys.length, setVersionSwitchGuard, confirm, t])

  // folder 列表 + 每个 folder 的原始张数（不受 filterTag 影响，让 tab 数字稳定
  // 不抖动 — 同 Preprocess chip 风格）。单 folder 项目时 UI 不显示 tabs。
  const folderNames = useMemo(() => {
    const set = new Set<string>()
    for (const m of meta.values()) set.add(m.folder)
    return Array.from(set).sort()
  }, [meta])
  const folderCounts = useMemo(() => {
    const c = new Map<string, number>()
    for (const m of meta.values()) c.set(m.folder, (c.get(m.folder) ?? 0) + 1)
    return c
  }, [meta])

  const filteredKeys = useMemo(() => {
    if (!folderFilter) return keys
    return keys.filter((k) => meta.get(k)?.folder === folderFilter)
  }, [keys, meta, folderFilter])

  const captionItems = useMemo(
    () =>
      filteredKeys.map((k) => {
        const m = meta.get(k)!
        const tags = cache.get(k) ?? []
        return {
          name: k,
          thumbUrl:
            activeVersion != null
              ? api.versionThumbUrl(project.id, activeVersion.id, 'train', m.name, m.folder)
              : '',
          meta: tags.slice(0, 5).join(', '),
        }
      }),
    [filteredKeys, meta, cache, project.id, activeVersion]
  )

  const selectedKeys = useMemo(
    () => filteredKeys.filter((k) => sel.has(k)),
    [filteredKeys, sel]
  )
  const navKeys = selectedKeys.length > 0 ? selectedKeys : filteredKeys
  const activeIndex = activeKey ? navKeys.indexOf(activeKey) : -1

  const tagSuggestions = useMemo(() => {
    const set = new Set<string>()
    for (const tags of cache.values()) for (const tag of tags) set.add(tag)
    return Array.from(set).sort((a, b) => a.localeCompare(b))
  }, [cache])

  const handlePickTag = useCallback(
    (tag: string) => {
      const matched = new Set<string>()
      for (const k of keys) {
        if ((cache.get(k) ?? []).includes(tag)) matched.add(k)
      }
      setSel(matched); setAnchor(null)
      toast(t('tagEdit.selectedContaining', { tag, n: matched.size }), 'success')
    },
    [keys, cache, toast, t]
  )

  if (!activeVersion) {
    return <p className="text-fg-tertiary p-6">{t('tagEdit.noVersion')}</p>
  }

  const handleClick = (key: string, e: React.MouseEvent) => {
    const r = applySelection(sel, key, e, filteredKeys, anchor)
    setSel(r.next); setAnchor(r.anchor)
  }

  const navActive = (delta: number) => {
    if (navKeys.length === 0) return
    const i = activeKey ? navKeys.indexOf(activeKey) : -1
    const next = i < 0 ? 0 : (i + delta + navKeys.length) % navKeys.length
    setActiveKey(navKeys[next])
  }

  const updateActiveTags = (tags: string[]) => {
    if (!activeKey) return
    setCache((prev) => {
      const next = new Map(prev); next.set(activeKey, [...tags]); return next
    })
  }

  const applyBulkUpdates = (updates: Map<string, string[]>) => {
    setCache((prev) => {
      const next = new Map(prev)
      for (const [k, v] of updates) next.set(k, v)
      return next
    })
  }

  // 标签分布行内 × 触发：从当前选中图删除该 tag。pre-compute updates 拿真实
  // 影响数 → confirm modal 显示精确张数 → 用户点确认后才 apply。
  const removeTagFromSelected = async (tag: string) => {
    if (selectedKeys.length === 0) return
    const updates = new Map<string, string[]>()
    for (const k of selectedKeys) {
      const cur = cache.get(k) ?? []
      if (!cur.includes(tag)) continue
      updates.set(k, cur.filter((tt) => tt !== tag))
    }
    if (updates.size === 0) return
    const ok = await confirm(
      t('bulkAction.confirmMessage', {
        op: t('bulkAction.opLabelRemove', { tags: tag }),
        n: updates.size,
      }),
      { tone: 'danger', title: t('bulkAction.confirmTitle') },
    )
    if (!ok) return
    applyBulkUpdates(updates)
    toast(t('tagEdit.removedFromN', { tag, n: updates.size }), 'success')
  }

  // 标签分布行内 ✎ inline edit 提交：把选中图里的 oldTag 替换成 newTag，去重。
  const replaceTagInSelected = async (oldTag: string, newTag: string) => {
    if (selectedKeys.length === 0 || !newTag || newTag === oldTag) return
    const updates = new Map<string, string[]>()
    for (const k of selectedKeys) {
      const cur = cache.get(k) ?? []
      if (!cur.includes(oldTag)) continue
      const next: string[] = []
      const seen = new Set<string>()
      for (const tt of cur) {
        const out = tt === oldTag ? newTag : tt
        if (seen.has(out)) continue
        seen.add(out); next.push(out)
      }
      updates.set(k, next)
    }
    if (updates.size === 0) return
    const ok = await confirm(
      t('bulkAction.confirmMessage', {
        op: t('bulkAction.opLabelReplace', { from: oldTag, to: newTag }),
        n: updates.size,
      }),
      { tone: 'danger', title: t('bulkAction.confirmTitle') },
    )
    if (!ok) return
    applyBulkUpdates(updates)
    toast(t('tagEdit.replacedInN', { from: oldTag, to: newTag, n: updates.size }), 'success')
  }

  const onSave = async () => {
    if (!dirty || versionId == null) return
    const items: CommitItem[] = dirtyKeys.map((k) => {
      const m = meta.get(k)!
      return { folder: m.folder, name: m.name, tags: cache.get(k) ?? [] }
    })
    try {
      const r = await api.commitCaptions(project.id, versionId, items)
      setInitial(new Map(cache))
      toast(t('tagEdit.savedToast', { written: r.written, id: r.snapshot.id }), 'success')
      void reload()
    } catch (e) { toast(String(e), 'error') }
  }

  const onAfterRestore = async () => {
    await reloadCache()
    setActiveKey('')
    setSel(new Set())
    setAnchor(null)
    setFolderFilter('')
    await reload()
  }

  const stats = activeVersion.stats
  const trainTotal = stats?.train_image_count ?? 0
  const taggedTotal = stats?.tagged_image_count ?? 0
  const allTagged = trainTotal > 0 && taggedTotal >= trainTotal

  const activeMeta = activeKey ? meta.get(activeKey) : undefined
  const activeFolder = activeMeta?.folder ?? ''
  const activeName = activeMeta?.name ?? ''
  const activeTags = activeKey ? cache.get(activeKey) ?? [] : []

  const isEditing = Boolean(activeKey)

  return (
    <StepShell
      idx={4}
      title={t('tagEdit.title')}
      subtitle={t('tagEdit.subtitle')}
      actions={
        <>
          {activeVersion.trigger_word && (
            <span className="badge badge-neutral" title={t('tagEdit.triggerWordHint')}>
              {t('tagEdit.triggerWord')}:{' '}
              <code className="font-mono">{activeVersion.trigger_word}</code>
            </span>
          )}
          {stats && (
            <span className={allTagged ? 'badge badge-ok' : 'badge badge-neutral'}>
              {t('tagEdit.taggedBadge', { tagged: taggedTotal, total: trainTotal })}
            </span>
          )}
          <SaveBar
            pid={project.id}
            vid={activeVersion.id}
            dirtyCount={dirtyKeys.length}
            onSave={onSave}
            onAfterRestore={onAfterRestore}
          />
        </>
      }
    >
      <div ref={rowRef} className="flex flex-1 min-h-0 gap-2.5">

        <section
          className="rounded-md border border-subtle bg-surface flex flex-col min-w-0 overflow-hidden"
          style={{ flex: isEditing ? `0 0 ${gridPct}%` : 1 }}
        >
          {folderNames.length > 1 && (
            <div className="px-2 pt-2 pb-1.5 flex items-center gap-1 flex-wrap shrink-0 border-b border-subtle">
              {['', ...folderNames].map((f) => {
                const isActive = f === folderFilter
                const label = f || t('common.all')
                const count = f ? folderCounts.get(f) ?? 0 : keys.length
                return (
                  <button
                    key={f || '__all__'}
                    type="button"
                    onClick={() => setFolderFilter(f)}
                    className={
                      'px-2 py-0.5 rounded-full text-xs font-medium transition-colors ' +
                      (isActive
                        ? 'bg-accent text-white'
                        : 'bg-overlay text-fg-secondary hover:bg-accent-soft')
                    }
                  >
                    {label} {count}
                  </button>
                )
              })}
            </div>
          )}
          <div className="flex-1 overflow-y-auto p-2">
            <ImageGrid
              items={captionItems}
              selected={sel}
              activeName={activeKey || undefined}
              onSelect={handleClick}
              onActivate={setActiveKey}
              clickMode="activate"
              ariaLabel="tag-edit-grid"
              emptyHint={
                folderFilter
                  ? t('tagEdit.noImagesInFolder', { folder: folderFilter })
                  : t('tagEdit.noImagesHint')
              }
            />
          </div>
        </section>

        {isEditing && (
          <PaneResizer
            containerRef={rowRef}
            value={gridPct}
            onChange={setGridPct}
            min={15}
            max={gridMax}
            ariaLabel={t('tagEdit.resizeGrid')}
          />
        )}

        {isEditing && (
          <section className="flex-1 rounded-md border border-subtle bg-surface flex flex-col min-w-0 overflow-hidden">
            <div className="px-3 py-2 border-b border-subtle shrink-0 flex items-center gap-2">
              <span className="text-xs text-fg-tertiary">{t('tagEdit.singleEdit')}</span>
              <code className="flex-1 min-w-0 text-xs font-mono text-fg-secondary truncate">
                {activeFolder}/{activeName}
              </code>
            </div>
            <div className="flex-1 relative p-2 min-h-0">
              {/* 原图分辨率 + zoom/pan（核对细节 tag 需要看清局部）；
                  size=0 = 原图直出，本地服务加载可接受。
                  ZoomableImage 自带视口样式 + readout 条 */}
              <ZoomableImage
                key={activeKey}
                src={api.versionThumbUrl(project.id, activeVersion.id, 'train', activeName, activeFolder, 0)}
                alt={activeName}
              />
            </div>
          </section>
        )}

        <PaneResizer
          containerRef={rowRef}
          value={sidePct}
          onChange={setSidePct}
          min={20}
          max={sideMax}
          anchor="end"
          ariaLabel={t('tagEdit.resizeSide')}
        />

        <div className="flex flex-col gap-2.5 min-w-0 min-h-0" style={{ flex: `0 0 ${sidePct}%` }}>
          {isEditing ? (
            // editing 时：bulk + 标签分布 都和"调单图标签"无关，整个侧栏让位给
            // TagEditor。退出 editing 后自动回来，sel / folderFilter 等 state
            // 保留（隐藏的是 UI 不是状态）。
            <section className="flex-1 rounded-md border border-subtle bg-surface p-2.5 flex flex-col gap-2 min-h-0 overflow-hidden">
              <div className="flex items-center gap-1.5 shrink-0">
                <button onClick={() => navActive(-1)} disabled={navKeys.length === 0} aria-label={t('tagEdit.prevImage')} className="btn btn-secondary btn-sm">◀</button>
                <span className="text-xs text-fg-tertiary font-mono flex-1 text-center">
                  {activeIndex >= 0 ? `${activeIndex + 1} / ${navKeys.length}` : `– / ${navKeys.length}`}
                </span>
                <button onClick={() => navActive(1)} disabled={navKeys.length === 0} aria-label={t('tagEdit.nextImage')} className="btn btn-secondary btn-sm">▶</button>
                <button onClick={() => setActiveKey('')} className="btn btn-ghost btn-sm ml-1" aria-label={t('tagEdit.closeEdit')}>✕</button>
              </div>
              <TagEditor tags={activeTags} onChange={updateActiveTags} />
            </section>
          ) : (
            // BulkActionBar + TagStatsPanel 合到同一个外框 section（"标签编辑
            // 工作区"），视觉上是一个面板：上半是 batch 输入区，下半是标签分布
            // 兼快捷单 tag 操作区。两者共享"操作 = 给当前选中图做"的语义。
            <section className="flex-1 min-h-0 rounded-md border border-subtle bg-surface flex flex-col overflow-hidden">
              <BulkActionBar
                cache={cache}
                selectedKeys={selectedKeys}
                onApply={applyBulkUpdates}
                tagSuggestions={tagSuggestions}
                onClearSelection={() => setSel(new Set())}
                onSelectAll={() => setSel(new Set(filteredKeys))}
                totalCount={filteredKeys.length}
              />
              <TagStatsPanel
                cache={cache}
                selectedKeys={selectedKeys}
                onPickTag={handlePickTag}
                onRemoveTag={removeTagFromSelected}
                onReplaceTag={replaceTagInSelected}
              />
            </section>
          )}
        </div>
      </div>
    </StepShell>
  )
}
