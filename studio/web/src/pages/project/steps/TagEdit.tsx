import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useBlocker, useOutletContext } from 'react-router-dom'
import {
  api,
  type CommitItem,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import Alert from '../../../components/Alert'
import Badge from '../../../components/Badge'
import BulkActionBar from '../../../components/BulkActionBar'
import Button from '../../../components/Button'
import Card from '../../../components/Card'
import { useDialog } from '../../../components/Dialog'
import EmptyState from '../../../components/EmptyState'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import PaneResizer, { normalizePanePair } from '../../../components/PaneResizer'
import SaveBar from '../../../components/SaveBar'
import { SegmentedControl } from '../../../components/SelectionGroup'
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

const TAG_EDIT_GRID_MIN = 15
const TAG_EDIT_SIDE_MIN = 20
const TAG_EDIT_PREVIEW_MIN = 15
const TAG_EDIT_GRID_PANE_ID = 'tag-edit-grid-pane'
const TAG_EDIT_SIDE_PANE_ID = 'tag-edit-side-pane'

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
  const dirtyRef = useRef(false)
  const editRevisionRef = useRef(0)
  const reloadRequestRef = useRef(0)
  const saveInFlightRef = useRef(false)
  const [initial, setInitial] = useState<Map<string, string[]>>(new Map())
  const [meta, setMeta] = useState<Map<string, CaptionMeta>>(new Map())
  const [keys, setKeys] = useState<string[]>([])
  const [hasLoaded, setHasLoaded] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [externalUpdatePending, setExternalUpdatePending] = useState(false)

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
  const normalizedPanes = normalizePanePair(gridPct, sidePct, {
    startMin: TAG_EDIT_GRID_MIN,
    endMin: TAG_EDIT_SIDE_MIN,
    flexibleMin: TAG_EDIT_PREVIEW_MIN,
  })
  const boundedGridPct = normalizedPanes.start
  const boundedSidePct = normalizedPanes.end

  // The two persisted fixed panes share one width budget. Repair invalid old values
  // together so the flexible preview pane always retains its declared minimum.
  useEffect(() => {
    if (gridPct !== boundedGridPct) setGridPct(boundedGridPct)
    if (sidePct !== boundedSidePct) setSidePct(boundedSidePct)
  }, [boundedGridPct, boundedSidePct, gridPct, setGridPct, setSidePct, sidePct])

  const gridMax = 100 - boundedSidePct - TAG_EDIT_PREVIEW_MIN
  const sideMax = 100 - boundedGridPct - TAG_EDIT_PREVIEW_MIN

  const reloadCache = useCallback(async (
    mode: 'initial' | 'refresh' | 'replace' = 'refresh',
  ) => {
    if (versionId == null) return false
    const requestId = ++reloadRequestRef.current
    const editRevision = editRevisionRef.current
    if (mode === 'initial') setIsLoading(true)
    try {
      const r = await api.listCaptionsFull(project.id, versionId)
      if (requestId !== reloadRequestRef.current) return false
      if (
        mode === 'refresh' &&
        (dirtyRef.current || editRevisionRef.current !== editRevision)
      ) {
        setExternalUpdatePending(true)
        return false
      }
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
      setCache(c)
      setInitial(new Map(c))
      setMeta(m)
      setKeys(ks)
      setHasLoaded(true)
      setLoadError(null)
      setExternalUpdatePending(false)
      return true
    } catch (e) {
      if (requestId !== reloadRequestRef.current) return false
      setLoadError(String(e))
      return false
    } finally {
      if (mode === 'initial' && requestId === reloadRequestRef.current) {
        setIsLoading(false)
      }
    }
  }, [project.id, versionId])

  useEffect(() => {
    setCache(new Map())
    setInitial(new Map())
    setMeta(new Map())
    setKeys([])
    setHasLoaded(false)
    setIsLoading(true)
    setLoadError(null)
    setExternalUpdatePending(false)
    setActiveKey('')
    setSel(new Set())
    setAnchor(null)
    setFolderFilter('')
    void reloadCache('initial')
  }, [reloadCache])

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
  const dirtyKeySet = useMemo(() => new Set(dirtyKeys), [dirtyKeys])
  dirtyRef.current = dirty

  useEventStream((evt) => {
    const relevantVersion = versionId != null && evt.version_id === versionId
    const versionChanged = evt.type === 'version_state_changed' && relevantVersion
    const tagJobFinished =
      evt.type === 'job_state_changed' &&
      relevantVersion &&
      evt.project_id === project.id &&
      evt.kind === 'tag' &&
      (evt.status === 'done' || evt.status === 'failed')
    if (!versionChanged && !tagJobFinished) return

    void reload()
    if (dirty) {
      setExternalUpdatePending(true)
      return
    }
    void reloadCache('refresh')
  })

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

  useEffect(() => {
    const valid = new Set(keys)
    setSel((prev) => {
      const next = new Set(Array.from(prev).filter((key) => valid.has(key)))
      return next.size === prev.size ? prev : next
    })
    setActiveKey((prev) => (prev && !valid.has(prev) ? '' : prev))
    if (folderFilter && !folderNames.includes(folderFilter)) {
      setFolderFilter('')
      setAnchor(null)
    }
  }, [folderFilter, folderNames, keys])

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
          badge: dirtyKeySet.has(k) ? t('tagEdit.unsavedBadge') : undefined,
          badgeTone: dirtyKeySet.has(k) ? 'warning' as const : undefined,
        }
      }),
    [filteredKeys, meta, cache, dirtyKeySet, project.id, activeVersion, t]
  )

  const selectedKeys = useMemo(
    () => filteredKeys.filter((k) => sel.has(k)),
    [filteredKeys, sel]
  )
  const navKeys = activeKey && selectedKeys.includes(activeKey) ? selectedKeys : filteredKeys
  const activeIndex = activeKey ? navKeys.indexOf(activeKey) : -1

  const tagSuggestions = useMemo(() => {
    const set = new Set<string>()
    for (const tags of cache.values()) for (const tag of tags) set.add(tag)
    return Array.from(set).sort((a, b) => a.localeCompare(b))
  }, [cache])

  const handlePickTag = useCallback(
    (tag: string) => {
      const matched = new Set<string>()
      for (const k of filteredKeys) {
        if ((cache.get(k) ?? []).includes(tag)) matched.add(k)
      }
      setSel(matched); setAnchor(null)
      toast(t('tagEdit.selectedContaining', { tag, n: matched.size }), 'success')
    },
    [filteredKeys, cache, toast, t]
  )

  if (!activeVersion) {
    return <EmptyState title={t('tagEdit.title')} description={t('tagEdit.noVersion')} />
  }

  const handleFolderChange = (folder: string) => {
    if (folder === folderFilter) return
    setFolderFilter(folder)
    setSel(new Set())
    setAnchor(null)
    setActiveKey('')
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
    editRevisionRef.current += 1
    dirtyRef.current = true
    setCache((prev) => {
      const next = new Map(prev); next.set(activeKey, [...tags]); return next
    })
  }

  const applyBulkUpdates = (updates: Map<string, string[]>) => {
    if (updates.size === 0) return
    editRevisionRef.current += 1
    dirtyRef.current = true
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

  const onSave = async (confirmExternal = true) => {
    if (!dirty || versionId == null || saveInFlightRef.current) return
    if (externalUpdatePending && confirmExternal) {
      const proceed = await confirm(
        t('tagEdit.externalSaveMessage', { n: dirtyKeys.length }),
        { tone: 'warn', title: t('tagEdit.externalUpdateTitle') },
      )
      if (!proceed) return
    }

    const editRevisionAtSave = editRevisionRef.current
    const items: CommitItem[] = dirtyKeys.map((k) => {
      const m = meta.get(k)!
      return { folder: m.folder, name: m.name, tags: [...(cache.get(k) ?? [])] }
    })
    const submitted = new Map(items.map((item) => [
      keyOf(item.folder, item.name),
      [...item.tags],
    ]))
    saveInFlightRef.current = true
    try {
      const r = await api.commitCaptions(project.id, versionId, items)
      const skipped = new Set(r.skipped)
      const writtenKeys = dirtyKeys.filter((k) => !skipped.has(k))
      setInitial((prev) => {
        const next = new Map(prev)
        for (const k of writtenKeys) next.set(k, [...(submitted.get(k) ?? [])])
        return next
      })

      if (r.skipped.length > 0) {
        toast(t('tagEdit.saveSkippedToast', {
          written: r.written,
          skipped: r.skipped.length,
        }), 'error')
      } else {
        toast(t('tagEdit.savedToast', { written: r.written, id: r.snapshot.id }), 'success')
      }

      const editedDuringSave = editRevisionRef.current !== editRevisionAtSave
      if (externalUpdatePending && r.skipped.length === 0 && !editedDuringSave) {
        // The submitted snapshot is now the clean baseline. A normal refresh still
        // refuses to apply if the user edits while the follow-up request is in flight.
        dirtyRef.current = false
        await reloadCache('refresh')
      }
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      saveInFlightRef.current = false
    }
  }

  const discardAndRefresh = async () => {
    if (dirty) {
      const discard = await confirm(
        t('tagEdit.externalDiscardMessage', { n: dirtyKeys.length }),
        {
          tone: 'danger',
          title: t('tagEdit.externalUpdateTitle'),
          okText: t('tagEdit.discardAndRefresh'),
          cancelText: t('tagEdit.unsavedConfirmStay'),
        },
      )
      if (!discard) return
    }
    const refreshed = await reloadCache('replace')
    if (!refreshed) return
    setExternalUpdatePending(false)
    setActiveKey('')
    setSel(new Set())
    setAnchor(null)
    await reload()
  }

  const onAfterRestore = async () => {
    const refreshed = await reloadCache('replace')
    if (!refreshed) return
    setExternalUpdatePending(false)
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
  const activeDirty = activeKey ? dirtyKeySet.has(activeKey) : false

  const isEditing = Boolean(activeKey)

  return (
    <StepShell
      title={t('tagEdit.title')}
      subtitle={t('tagEdit.subtitle')}
      actions={
        <>
          {activeVersion.trigger_word && (
            <Badge tone="neutral" title={t('tagEdit.triggerWordHint')}>
              {t('tagEdit.triggerWord')}:{' '}
              <code className="font-mono">{activeVersion.trigger_word}</code>
            </Badge>
          )}
          {stats && (
            <Badge tone={allTagged ? 'success' : 'neutral'}>
              {t('tagEdit.taggedBadge', { tagged: taggedTotal, total: trainTotal })}
            </Badge>
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
      {externalUpdatePending && (
        <Alert
          tone="warning"
          size="sm"
          title={t('tagEdit.externalUpdateTitle')}
          className="mb-related shrink-0"
          action={(
            <div className="flex flex-wrap items-center gap-related">
              <Button variant="primary" size="sm" onClick={() => void onSave(false)}>
                {t('tagEdit.saveAndRefresh')}
              </Button>
              <Button variant="danger" size="sm" onClick={() => void discardAndRefresh()}>
                {t('tagEdit.discardAndRefresh')}
              </Button>
            </div>
          )}
        >
          {t('tagEdit.externalUpdateMessage', { n: dirtyKeys.length })}
        </Alert>
      )}

      {hasLoaded && loadError && (
        <Alert
          tone="danger"
          size="sm"
          role="alert"
          title={t('tagEdit.refreshErrorTitle')}
          className="mb-related shrink-0"
          action={(
            <Button variant="secondary" size="sm" onClick={() => void reloadCache('refresh')}>
              {t('common.retry')}
            </Button>
          )}
        >
          {loadError}
        </Alert>
      )}

      {isLoading && !hasLoaded ? (
        <Card
          role="status"
          aria-busy="true"
          padding="lg"
          className="flex flex-1 min-h-0 items-center justify-center text-sm text-fg-secondary"
        >
          {t('tagEdit.loading')}
        </Card>
      ) : !hasLoaded && loadError ? (
        <Alert
          tone="danger"
          role="alert"
          title={t('tagEdit.loadErrorTitle')}
          action={(
            <Button variant="secondary" size="sm" onClick={() => void reloadCache('initial')}>
              {t('common.retry')}
            </Button>
          )}
        >
          {loadError}
        </Alert>
      ) : hasLoaded && keys.length === 0 ? (
        <EmptyState
          title={t('tagEdit.emptyTitle')}
          description={t('tagEdit.noImagesHint')}
          className="flex-1"
        />
      ) : (
      <div
        data-tag-edit-workspace
        ref={rowRef}
        role="region"
        aria-label={t('tagEdit.workspaceLabel')}
        tabIndex={0}
        className="flex flex-1 min-h-0 gap-2.5 overflow-x-auto overflow-y-hidden focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent"
      >

        <Card
          as="section"
          id={TAG_EDIT_GRID_PANE_ID}
          radius="compact"
          className="flex flex-col min-w-[240px] min-h-0 overflow-hidden"
          style={{ flex: isEditing ? `0 0 ${boundedGridPct}%` : 1 }}
        >
          <div className="px-field py-related flex flex-col gap-related shrink-0 border-b border-subtle">
            <div className="flex items-center gap-related min-w-0">
              <h2 className="type-panel-title m-0 flex-1">{t('tagEdit.imageListTitle')}</h2>
              <span className="text-xs text-fg-tertiary tnum">
                {t('tagEdit.visibleCount', { n: filteredKeys.length })}
              </span>
            </div>
            {folderNames.length > 1 && (
              <SegmentedControl
                items={['', ...folderNames].map((folder) => ({
                  value: folder,
                  label: `${folder || t('common.all')} ${folder ? folderCounts.get(folder) ?? 0 : keys.length}`,
                }))}
                value={folderFilter}
                onChange={handleFolderChange}
                ariaLabel={t('tagEdit.folderFilterLabel')}
                idPrefix="tag-edit-folder"
                size="sm"
                layout="content"
              />
            )}
          </div>
          <ImageGrid
            className="flex-1 min-h-0"
            contentClassName="p-2"
            items={captionItems}
            selected={sel}
            activeName={activeKey || undefined}
            onSelect={handleClick}
            onActivate={setActiveKey}
            clickMode={sel.size > 0 ? 'select' : 'activate'}
            ariaLabel={t('tagEdit.gridLabel')}
            emptyHint={
              folderFilter
                ? t('tagEdit.noImagesInFolder', { folder: folderFilter })
                : t('tagEdit.noImagesHint')
            }
          />
        </Card>

        {isEditing && (
          <PaneResizer
            containerRef={rowRef}
            value={boundedGridPct}
            onChange={setGridPct}
            min={TAG_EDIT_GRID_MIN}
            max={gridMax}
            ariaLabel={t('tagEdit.resizeGrid')}
            ariaControls={TAG_EDIT_GRID_PANE_ID}
          />
        )}

        {isEditing && (
          <Card as="section" radius="compact" className="flex-1 min-w-[240px] min-h-0 flex flex-col overflow-hidden">
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
          </Card>
        )}

        <PaneResizer
          containerRef={rowRef}
          value={boundedSidePct}
          onChange={setSidePct}
          min={TAG_EDIT_SIDE_MIN}
          max={sideMax}
          anchor="end"
          ariaLabel={t('tagEdit.resizeSide')}
          ariaControls={TAG_EDIT_SIDE_PANE_ID}
        />

        <div
          id={TAG_EDIT_SIDE_PANE_ID}
          className="flex flex-col gap-2.5 min-w-[260px] min-h-0"
          style={{ flex: `0 0 ${boundedSidePct}%` }}
        >
          {isEditing ? (
            // editing 时：bulk + 标签分布 都和"调单图标签"无关，整个侧栏让位给
            // TagEditor。退出 editing 后自动回来，sel / folderFilter 等 state
            // 保留（隐藏的是 UI 不是状态）。
            <Card
              as="section"
              radius="compact"
              className="flex-1 flex flex-col min-h-0 overflow-hidden"
              aria-labelledby="tag-edit-editor-title"
            >
              <header className="px-2.5 py-2 border-b border-subtle flex items-center gap-2 shrink-0 min-w-0">
                <div className="flex items-baseline gap-2 min-w-0">
                  <h2 id="tag-edit-editor-title" className="type-panel-title m-0 whitespace-nowrap">
                    {t('tagEdit.title')}
                  </h2>
                  <span className="text-xs text-fg-tertiary tnum whitespace-nowrap">
                    {t('tagEditor.tagCount', { n: activeTags.length })}
                  </span>
                  {activeDirty && (
                    <Badge tone="warning" size="sm">
                      {t('tagEdit.unsavedBadge')}
                    </Badge>
                  )}
                </div>
                <div className="ml-auto flex items-center gap-1 shrink-0">
                  <Button
                    variant="secondary"
                    size="xs"
                    iconOnly
                    onClick={() => navActive(-1)}
                    disabled={navKeys.length === 0}
                    aria-label={t('tagEdit.prevImage')}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="m15 18-6-6 6-6" />
                    </svg>
                  </Button>
                  <span
                    className="min-w-[2.75rem] text-xs text-fg-tertiary font-mono text-center tnum"
                    title={t(selectedKeys.includes(activeKey) ? 'tagEdit.navSelectedScope' : 'tagEdit.navFolderScope')}
                  >
                    {`${activeIndex + 1}/${navKeys.length}`}
                  </span>
                  <Button
                    variant="secondary"
                    size="xs"
                    iconOnly
                    onClick={() => navActive(1)}
                    disabled={navKeys.length === 0}
                    aria-label={t('tagEdit.nextImage')}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="m9 18 6-6-6-6" />
                    </svg>
                  </Button>
                  <span className="mx-0.5 h-4 w-px bg-subtle" aria-hidden="true" />
                  <Button
                    variant="ghost"
                    size="xs"
                    iconOnly
                    onClick={() => setActiveKey('')}
                    aria-label={t('tagEdit.closeEdit')}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                      <path d="M6 6l12 12M18 6 6 18" />
                    </svg>
                  </Button>
                </div>
              </header>
              <div className="p-2.5 flex-1 min-h-0 flex flex-col">
                <TagEditor
                  resetKey={activeKey}
                  tags={activeTags}
                  onChange={updateActiveTags}
                  showTagCount={false}
                />
              </div>
            </Card>
          ) : (
            // BulkActionBar + TagStatsPanel 合到同一个外框 section（"标签编辑
            // 工作区"），视觉上是一个面板：上半是 batch 输入区，下半是标签分布
            // 兼快捷单 tag 操作区。两者共享"操作 = 给当前选中图做"的语义。
            <Card as="section" radius="compact" className="flex-1 min-h-0 flex flex-col overflow-hidden">
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
            </Card>
          )}
        </div>
      </div>
      )}
    </StepShell>
  )
}
