import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type CurationItem,
  type CurationView,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import StepShell from '../../../components/StepShell'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

// ---------- 排序 ----------
type SortMode =
  | 'id-asc'
  | 'id-desc'
  | 'name-asc'
  | 'name-desc'
  | 'mtime-asc'
  | 'mtime-desc'

const SORT_STORAGE_KEY = 'curation:sort'
const DEFAULT_SORT: SortMode = 'id-asc'

function numericIdKey(name: string): number {
  const stem = name.replace(/\.[^.]+$/, '')
  return /^\d+$/.test(stem) ? Number(stem) : Number.POSITIVE_INFINITY
}

function compareItems(a: CurationItem, b: CurationItem, mode: SortMode): number {
  switch (mode) {
    case 'id-asc':
    case 'id-desc': {
      const ka = numericIdKey(a.name)
      const kb = numericIdKey(b.name)
      const d = ka === kb ? a.name.localeCompare(b.name) : ka - kb
      return mode === 'id-asc' ? d : -d
    }
    case 'name-asc':
      return a.name.localeCompare(b.name)
    case 'name-desc':
      return b.name.localeCompare(a.name)
    case 'mtime-asc':
      return a.mtime - b.mtime || a.name.localeCompare(b.name)
    case 'mtime-desc':
      return b.mtime - a.mtime || a.name.localeCompare(b.name)
  }
}

function normalizeItem(it: CurationItem | string | undefined): CurationItem {
  if (typeof it === 'string') return { name: it, mtime: 0 }
  if (it && typeof it.name === 'string')
    return { name: it.name, mtime: typeof it.mtime === 'number' ? it.mtime : 0 }
  return { name: '', mtime: 0 }
}

function sortItems(
  items: (CurationItem | string)[],
  mode: SortMode
): CurationItem[] {
  return items.map(normalizeItem).sort((a, b) => compareItems(a, b, mode))
}

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface Preview {
  side: 'left' | 'right'
  name: string
  folder?: string
  url: string
  caption: string
  list: string[]
  index: number
  resolve: (name: string) => string
}

type Focus =
  | { side: 'left'; name: string; url: string }
  | { side: 'right'; folder: string; name: string; url: string }

const FOLDER_PATTERN = /^([0-9]+_)?[A-Za-z][A-Za-z0-9_-]*$/

const SCROLL_BOX = 'flex-1 min-h-0 overflow-y-auto pr-1'

export default function CurationPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const dialog = useDialog()
  const [view, setView] = useState<CurationView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const SORT_OPTIONS: { value: SortMode; label: string }[] = [
    { value: 'id-asc', label: 'ID ↑' },
    { value: 'id-desc', label: 'ID ↓' },
    { value: 'name-asc', label: t('common.filename') + ' ↑' },
    { value: 'name-desc', label: t('common.filename') + ' ↓' },
    { value: 'mtime-asc', label: t('curate.downloadTime') + ' ↑' },
    { value: 'mtime-desc', label: t('curate.downloadTime') + ' ↓' },
  ]

  const [leftSel, setLeftSel] = useState<Set<string>>(new Set())
  const [leftAnchor, setLeftAnchor] = useState<string | null>(null)
  const [rightFolder, setRightFolder] = useState<string>('')
  const [rightSel, setRightSel] = useState<Set<string>>(new Set())
  const [rightAnchor, setRightAnchor] = useState<string | null>(null)

  const [focus, setFocus] = useState<Focus | null>(null)
  const [altHeld, setAltHeld] = useState(false)
  useEffect(() => {
    const isAlt = (e: KeyboardEvent) =>
      e.key === 'Alt' || e.code === 'AltLeft' || e.code === 'AltRight'
    const down = (e: KeyboardEvent) => {
      if (isAlt(e)) { e.preventDefault(); setAltHeld(true) }
    }
    const up = (e: KeyboardEvent) => {
      if (isAlt(e)) { e.preventDefault(); setAltHeld(false) }
    }
    const move = (e: MouseEvent) => {
      if (e.altKey !== altHeld) setAltHeld(e.altKey)
    }
    const blur = () => setAltHeld(false)
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('mousemove', move)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('mousemove', move)
      window.removeEventListener('blur', blur)
    }
  }, [altHeld])

  const [newFolder, setNewFolder] = useState<string>('')
  const [renaming, setRenaming] = useState<{ target: string; value: string } | null>(null)
  const [preview, setPreview] = useState<Preview | null>(null)

  const [sortMode, setSortMode] = useState<SortMode>(() => {
    if (typeof window === 'undefined') return DEFAULT_SORT
    const v = window.localStorage.getItem(SORT_STORAGE_KEY)
    return (['id-asc','id-desc','name-asc','name-desc','mtime-asc','mtime-desc'] as SortMode[]).includes(v as SortMode)
      ? (v as SortMode)
      : DEFAULT_SORT
  })
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(SORT_STORAGE_KEY, sortMode)
    }
  }, [sortMode])

  const versionId = activeVersion?.id ?? null

  const refresh = useCallback(async () => {
    if (versionId == null) return
    try {
      const v = await api.getCuration(project.id, versionId)
      setView(v)
      setError(null)
      const fallback = v.folders.includes('1_data') ? '1_data' : v.folders[0] ?? ''
      if (!rightFolder || !v.folders.includes(rightFolder)) {
        setRightFolder(fallback)
        setRightSel(new Set())
        setRightAnchor(null)
      }
    } catch (e) {
      setError(String(e))
    }
  }, [project.id, versionId, rightFolder])

  useEffect(() => { void refresh() }, [refresh])

  useEventStream((evt) => {
    if (
      evt.type === 'version_state_changed' &&
      evt.project_id === project.id &&
      versionId != null &&
      evt.version_id === versionId
    ) {
      void refresh()
    }
  })

  const folderNames = view?.folders ?? []

  const leftSortedNames = useMemo(
    () => sortItems(view?.left ?? [], sortMode).map((e) => e.name),
    [view, sortMode]
  )
  const trainEntries = useMemo(
    () => (view && rightFolder ? view.right[rightFolder] ?? [] : []),
    [view, rightFolder]
  )
  const rightSortedNames = useMemo(
    () => sortItems(trainEntries, sortMode).map((e) => e.name),
    [trainEntries, sortMode]
  )

  // ADR 0010 fixup: train 区 thumb 走 download bucket + manifest.origin，
  // 显示"预处理前的样子"。trainEntries 已带 origin（backend list_train 加）。
  // 用 raw=1 跳过 resolve_origin —— 否则老 ADR 0004 设计会 hijack 到
  // preprocess/{派生} 派生（X.jpg → preprocess/X_c0.png），但 ADR 0010
  // 后 preprocess/ 不再被 worker 写 → 404 裂图。
  const rightOriginByName = useMemo(() => {
    const m = new Map<string, string>()
    for (const e of trainEntries) {
      m.set(e.name, e.origin ?? e.name)
    }
    return m
  }, [trainEntries])

  const leftItems = useMemo(
    () => leftSortedNames.map((n) => ({
      name: n,
      thumbUrl: api.projectThumbUrl(project.id, n, 'download', 256, undefined, true),
    })),
    [leftSortedNames, project.id]
  )
  const rightItems = useMemo(
    () =>
      versionId == null
        ? []
        : rightSortedNames.map((n) => ({
            name: n,
            thumbUrl: api.projectThumbUrl(
              project.id, rightOriginByName.get(n) ?? n, 'download', 256,
              undefined, true,
            ),
          })),
    [rightSortedNames, project.id, versionId, rightOriginByName]
  )

  const onLeftHover = useCallback(
    (name: string) =>
      setFocus({
        side: 'left', name,
        url: api.projectThumbUrl(project.id, name, 'download', 768, undefined, true),
      }),
    [project.id]
  )

  const onRightHover = useCallback(
    (name: string) => {
      if (versionId == null || !rightFolder) return
      const origin = rightOriginByName.get(name) ?? name
      setFocus({
        side: 'right',
        folder: rightFolder,
        name,
        url: api.projectThumbUrl(project.id, origin, 'download', 768, undefined, true),
      })
    },
    [versionId, project.id, rightFolder, rightOriginByName]
  )

  if (!activeVersion) {
    return <p className="text-fg-tertiary p-6">{t('curate.noVersion')}</p>
  }
  if (error) {
    return (
      <div className="p-3 rounded-md bg-err-soft border border-err text-err font-mono text-sm">
        {error}
      </div>
    )
  }
  if (!view) return <p className="text-fg-tertiary p-6">{t('curate.loading')}</p>

  const switchRightFolder = (next: string) => {
    setRightFolder(next)
    setRightSel(new Set())
    setRightAnchor(null)
  }

  const handleLeftClick = (name: string, e: React.MouseEvent) => {
    const r = applySelection(leftSel, name, e, leftSortedNames, leftAnchor)
    setLeftSel(r.next)
    setLeftAnchor(r.anchor)
  }

  const handleRightClick = (name: string, e: React.MouseEvent) => {
    const r = applySelection(rightSel, name, e, rightSortedNames, rightAnchor)
    setRightSel(r.next)
    setRightAnchor(r.anchor)
  }

  const copyLeftFiles = async (files: string[], options: { clearSelection?: boolean } = {}) => {
    if (!rightFolder) { toast(t('curate.noTargetFolder'), 'error'); return false }
    if (!FOLDER_PATTERN.test(rightFolder)) { toast(t('curate.invalidFolder'), 'error'); return false }
    if (files.length === 0 || busy) return false
    setBusy(true)
    try {
      const r = await api.copyToTrain(project.id, activeVersion.id, { files, dest_folder: rightFolder })
      toast(
        t('curate.copiedN', { n: r.copied.length }) +
        (r.skipped.length ? t('curate.copiedSkipped', { n: r.skipped.length }) : ''),
        'success'
      )
      if (options.clearSelection) setLeftSel(new Set())
      await refresh()
      await reload()
      return true
    } catch (e) {
      toast(String(e), 'error')
      return false
    } finally {
      setBusy(false)
    }
  }

  const removeRightFiles = async (
    folder: string,
    files: string[],
    options: { clearSelection?: boolean; confirm?: boolean } = {}
  ) => {
    if (!folder || files.length === 0 || busy) return false
    if (options.confirm &&
        !(await dialog.confirm(t('curate.confirmRemove', { folder, n: files.length }), { tone: 'warn', okText: t('curate.removeOkText') }))) {
      return false
    }
    setBusy(true)
    try {
      const r = await api.removeFromTrain(project.id, activeVersion.id, { folder, files })
      toast(t('curate.removedN', { n: r.removed.length }), 'success')
      if (options.clearSelection) setRightSel(new Set())
      await refresh()
      await reload()
      return true
    } catch (e) {
      toast(String(e), 'error')
      return false
    } finally {
      setBusy(false)
    }
  }

  const doCopy = async () => {
    await copyLeftFiles(Array.from(leftSel), { clearSelection: true })
  }

  const doRemove = async () => {
    await removeRightFiles(rightFolder, Array.from(rightSel), { clearSelection: true, confirm: true })
  }

  const doCreateFolder = async () => {
    const name = newFolder.trim()
    if (!name) return
    if (!FOLDER_PATTERN.test(name)) return toast(t('curate.invalidFolder'), 'error')
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, { op: 'create', name })
      setNewFolder('')
      switchRightFolder(name)
      await refresh()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const doRenameFolder = async () => {
    if (!renaming) return
    const target = renaming.target
    const next = renaming.value.trim()
    if (!next || next === target) { setRenaming(null); return }
    if (!FOLDER_PATTERN.test(next)) return toast(t('curate.invalidFolder'), 'error')
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, { op: 'rename', name: target, new_name: next })
      if (rightFolder === target) switchRightFolder(next)
      setRenaming(null)
      toast(t('curate.renamedToast', { from: target, to: next }), 'success')
      await refresh()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const doDeleteFolder = async (name: string) => {
    const cnt = view.right[name]?.length ?? 0
    if (!(await dialog.confirm(
      t('curate.confirmDeleteFolder', { name, n: cnt }),
      { tone: 'warn', okText: t('curate.deleteFolderOkText') },
    ))) return
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, { op: 'delete', name })
      if (rightFolder === name) switchRightFolder('')
      await refresh()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const openLeftPreview = (name: string) => {
    setPreview({
      side: 'left', name,
      url: api.projectThumbUrl(project.id, name, 'download', 1600),
      caption: name,
      list: leftSortedNames,
      index: leftSortedNames.indexOf(name),
      resolve: (n) => api.projectThumbUrl(project.id, n, 'download', 1600),
    })
  }
  const openRightPreview = (name: string) => {
    if (versionId == null) return
    const folder = rightFolder
    setPreview({
      side: 'right', name, folder,
      url: api.versionThumbUrl(project.id, versionId, 'train', name, folder, 1600),
      caption: `${folder}/${name}`,
      list: rightSortedNames,
      index: rightSortedNames.indexOf(name),
      resolve: (n) => api.versionThumbUrl(project.id, versionId, 'train', n, folder, 1600),
    })
  }
  const stepPreview = (delta: number) => {
    if (!preview) return
    const idx = preview.index + delta
    if (idx < 0 || idx >= preview.list.length) return
    const name = preview.list[idx]
    setPreview({
      ...preview, name,
      url: preview.resolve(name),
      caption: preview.side === 'right' && preview.folder ? `${preview.folder}/${name}` : name,
      index: idx,
    })
  }

  const advancePreviewAfterAction = (doneName: string) => {
    if (!preview) return
    const list = preview.list.filter((name) => name !== doneName)
    if (list.length === 0) { setPreview(null); return }
    const index = Math.min(preview.index, list.length - 1)
    const name = list[index]
    setPreview({
      ...preview, name,
      url: preview.resolve(name),
      caption: preview.side === 'right' && preview.folder ? `${preview.folder}/${name}` : name,
      list, index,
    })
  }

  const copyPreviewImage = async () => {
    if (!preview || preview.side !== 'left' || busy) return
    const name = preview.name
    if (await copyLeftFiles([name])) advancePreviewAfterAction(name)
  }

  const removePreviewImage = async () => {
    if (!preview || preview.side !== 'right' || !preview.folder || busy) return
    const folder = preview.folder
    const name = preview.name
    if (await removeRightFiles(folder, [name])) advancePreviewAfterAction(name)
  }

  return (
    <StepShell
      idx={2}
      title={t('steps.curate.title')}
      subtitle={t('steps.curate.subtitle')}
      actions={
        <label className="flex items-center gap-1.5 text-sm text-fg-secondary whitespace-nowrap shrink-0">
          {t('curate.sortLabel')}
          <select
            value={sortMode}
            onChange={(e) => setSortMode(e.target.value as SortMode)}
            className="input px-2 py-0.5 text-sm"
            title={t('curate.sortTitle')}
          >
            {SORT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
      }
    >
    <div className="flex flex-col h-full gap-3">

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 items-stretch flex-1 min-h-0">
        <PanelCard
          accent="emerald"
          title={t('curate.downloadPanelTitle')}
          subtitle={t('curate.downloadSubtitle', { unused: view.left.length, total: view.download_total, sel: leftSel.size })}
          actions={
            <>
              <BtnSecondary
                onClick={() => setLeftSel(new Set(leftSortedNames))}
                disabled={busy || leftSortedNames.length === 0}
              >
                {t('curate.selectAll')}
              </BtnSecondary>
              <BtnSecondary
                onClick={() => setLeftSel(new Set())}
                disabled={busy || leftSel.size === 0}
              >
                {t('curate.deselect')}
              </BtnSecondary>
              <BtnPrimary
                onClick={doCopy}
                disabled={busy || leftSel.size === 0 || !rightFolder}
                title={rightFolder ? t('curate.copyToTitle', { folder: rightFolder }) : t('curate.noFolderTitle')}
              >
                {t('curate.copyToBtn', { n: leftSel.size, folder: rightFolder || '?' })}
              </BtnPrimary>
            </>
          }
        >
          <div className={SCROLL_BOX}>
            <ImageGrid
              items={leftItems}
              selected={leftSel}
              activeName={preview?.side === 'left' ? preview.name : undefined}
              onSelect={handleLeftClick}
              onHover={onLeftHover}
              onPreview={openLeftPreview}
              onActivate={openLeftPreview}
              clickMode="activate"
              ariaLabel="download-grid"
              emptyHint={t('curate.downloadEmptyHint')}
            />
          </div>
        </PanelCard>

        <PanelCard
          accent="cyan"
          title={t('curate.trainPanelTitle')}
          subtitle={t('curate.trainSubtitle', { total: view.train_total, folders: folderNames.length, sel: rightSel.size })}
          actions={
            <>
              <input
                value={newFolder}
                onChange={(e) => setNewFolder(e.target.value)}
                placeholder={t('curate.newFolderPlaceholder')}
                className="input input-mono px-2 py-0.5 text-sm"
                style={{ width: 144 }}
              />
              <BtnSecondary onClick={doCreateFolder} disabled={busy || !newFolder.trim()}>
                {t('curate.createFolderBtn')}
              </BtnSecondary>
              <BtnSecondary
                onClick={() => setRightSel(new Set(rightSortedNames))}
                disabled={busy || rightSortedNames.length === 0}
              >
                {t('curate.selectAll')}
              </BtnSecondary>
              <BtnSecondary
                onClick={() => setRightSel(new Set())}
                disabled={busy || rightSel.size === 0}
              >
                {t('curate.deselect')}
              </BtnSecondary>
              <BtnDanger onClick={doRemove} disabled={busy || rightSel.size === 0 || !rightFolder}>
                {t('curate.removeNBtn', { n: rightSel.size })}
              </BtnDanger>
            </>
          }
        >
          <FolderSummary
            folders={folderNames}
            counts={Object.fromEntries(folderNames.map((f) => [f, view.right[f]?.length ?? 0]))}
            activeFolder={rightFolder}
            busy={busy}
            onSwitch={switchRightFolder}
            onRename={(name) => setRenaming({ target: name, value: name })}
            onDelete={doDeleteFolder}
          />

          {renaming && (
            <div className="flex items-center gap-2 my-3 text-sm">
              <span className="text-fg-secondary">{t('curate.renameLabel', { name: renaming.target })}</span>
              <input
                autoFocus
                value={renaming.value}
                onChange={(e) => setRenaming({ ...renaming, value: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') doRenameFolder()
                  if (e.key === 'Escape') setRenaming(null)
                }}
                className="input input-mono px-2 py-0.5"
                style={{ width: 176 }}
              />
              <BtnPrimary onClick={doRenameFolder} disabled={busy}>
                {t('curate.renameOk')}
              </BtnPrimary>
              <button onClick={() => setRenaming(null)} className="btn btn-ghost btn-sm">
                {t('common.cancel')}
              </button>
            </div>
          )}

          <div className={`${SCROLL_BOX} mt-3`}>
            <ImageGrid
              items={rightItems}
              selected={rightSel}
              activeName={preview?.side === 'right' ? preview.name : undefined}
              onSelect={handleRightClick}
              onHover={onRightHover}
              onPreview={openRightPreview}
              onActivate={openRightPreview}
              clickMode="activate"
              ariaLabel="train-grid"
              emptyHint={
                rightFolder
                  ? t('curate.trainEmptyFolder', { folder: rightFolder })
                  : t('curate.trainNoFolder')
              }
            />
          </div>
        </PanelCard>
      </div>

      {altHeld && focus && <AltHoverPreview focus={focus} />}

      {preview && (
        <ImagePreviewModal
          src={preview.url}
          caption={preview.caption}
          hasPrev={preview.index > 0}
          hasNext={preview.index < preview.list.length - 1}
          onClose={() => setPreview(null)}
          onPrev={() => stepPreview(-1)}
          onNext={() => stepPreview(1)}
          onAccept={preview.side === 'left' ? copyPreviewImage : undefined}
          onDelete={preview.side === 'right' ? removePreviewImage : undefined}
          shortcutHint={
            preview.side === 'left'
              ? t('curate.previewHintLeft')
              : t('curate.previewHintRight')
          }
        />
      )}
    </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// 子组件
// ---------------------------------------------------------------------------

function FolderSummary({
  folders, counts, activeFolder, busy, onSwitch, onRename, onDelete,
}: {
  folders: string[]
  counts: Record<string, number>
  activeFolder: string
  busy: boolean
  onSwitch: (name: string) => void
  onRename: (name: string) => void
  onDelete: (name: string) => void
}) {
  const { t } = useTranslation()
  if (folders.length === 0) {
    return <p className="text-sm text-fg-tertiary">{t('curate.noTrainFolders')}</p>
  }
  const total = folders.reduce((s, f) => s + (counts[f] ?? 0), 0)
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-sm">
      {folders.map((f) => {
        const isActive = f === activeFolder
        return (
          <span
            key={f}
            className={`group inline-flex items-center transition-colors rounded-md ${
              isActive ? 'border border-accent bg-accent-soft' : 'border border-dim bg-surface'
            }`}
          >
            <button
              onClick={() => onSwitch(f)}
              title={isActive ? t('curate.folderActiveTitle') : t('curate.folderSwitchTitle')}
              className={`px-2 py-0.5 font-mono ${isActive ? 'text-accent' : 'text-fg-secondary'}`}
            >
              {f}
              <span className="text-fg-tertiary"> ({counts[f] ?? 0})</span>
            </button>
            <button
              onClick={() => onRename(f)}
              disabled={busy}
              title={t('common.rename')}
              className="opacity-0 group-hover:opacity-100 px-1 py-0.5 text-xs text-fg-tertiary"
            >
              ✎
            </button>
            <button
              onClick={() => onDelete(f)}
              disabled={busy}
              title={t('common.delete')}
              className="opacity-0 group-hover:opacity-100 px-1 py-0.5 text-xs text-fg-tertiary"
            >
              ×
            </button>
          </span>
        )
      })}
      <span className="text-fg-tertiary ml-2">{t('curate.folderTotal', { total })}</span>
    </div>
  )
}

function AltHoverPreview({ focus }: { focus: Focus }) {
  const { t } = useTranslation()
  const sourceLabel = focus.side === 'left'
    ? t('curate.sourceLabelDownload')
    : t('curate.sourceLabelTrain', { folder: focus.folder })
  return (
    <div
      aria-hidden
      className="fixed inset-0 z-40 pointer-events-none flex items-center justify-center p-6"
    >
      <div className="relative flex flex-col overflow-hidden rounded-lg border border-bold max-w-[95vw] max-h-[95vh] bg-black/90 shadow-xl">
        <img src={focus.url} alt={focus.name} className="max-w-[95vw] max-h-[88vh] object-contain" />
        <div className="flex items-center gap-2 shrink-0 px-3 py-1.5 border-t border-white/[0.08]">
          <span className={`shrink-0 ${focus.side === 'left' ? 'badge badge-ok' : 'badge badge-info'}`}>
            {sourceLabel}
          </span>
          <code className="mono truncate flex-1 min-w-0 text-fg-inverse text-sm">{focus.name}</code>
          <span className="text-xs shrink-0 text-white/40">{t('curate.altHoverClose')}</span>
        </div>
      </div>
    </div>
  )
}

const ACCENT_BAR_CLS: Record<'emerald' | 'cyan', string> = {
  emerald: 'bg-ok',
  cyan: 'bg-info',
}

function PanelCard({
  accent, title, subtitle, actions, children,
}: {
  accent: 'emerald' | 'cyan'
  title: string
  subtitle: string
  actions: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section className="flex flex-col min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <div className={`h-0.5 ${ACCENT_BAR_CLS[accent]}`} />
      <header className="flex flex-wrap items-center gap-1.5 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">{title}</h3>
        <span className="text-xs text-fg-tertiary">{subtitle}</span>
        <span className="flex-1" />
        {actions}
      </header>
      <div className="flex-1 min-h-0 flex flex-col p-2">{children}</div>
    </section>
  )
}

function BtnPrimary({ children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button {...rest} className="btn btn-primary btn-sm">{children}</button>
}

function BtnSecondary({ children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button {...rest} className="btn btn-secondary btn-sm">{children}</button>
}

function BtnDanger({ children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button {...rest} className="btn btn-sm bg-err-soft text-err border-err">{children}</button>
}
