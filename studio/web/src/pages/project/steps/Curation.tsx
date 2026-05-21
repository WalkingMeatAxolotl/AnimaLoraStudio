import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type CurationItem,
  type CurationView,
  type DuplicateGroup,
  type DuplicateItem,
  type DuplicateScanOptions,
  type DuplicateScanResult,
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
  side: 'left' | 'right' | 'duplicate'
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

const DEFAULT_DUPLICATE_OPTIONS: DuplicateScanOptions = {
  target: 'unused',
  match_scope: 'both',
  hash_size: 768,
  hash_workers: 4,
  tile_grids: [4, 6],
  structure_threshold: 6,
  variant_score: 72,
  aspect_tolerance: 0.045,
  min_close_tiles: 0.48,
  tile_median: 14,
  min_gray_close: 0.42,
}

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
  const [dupOptions, setDupOptions] = useState<DuplicateScanOptions>(DEFAULT_DUPLICATE_OPTIONS)
  const [dupResult, setDupResult] = useState<DuplicateScanResult | null>(null)
  const [dupSel, setDupSel] = useState<Set<string>>(new Set())
  const [dupBusy, setDupBusy] = useState(false)

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

  const leftItems = useMemo(
    () => leftSortedNames.map((n) => ({ name: n, thumbUrl: api.projectThumbUrl(project.id, n) })),
    [leftSortedNames, project.id]
  )
  const rightItems = useMemo(
    () =>
      versionId == null
        ? []
        : rightSortedNames.map((n) => ({
            name: n,
            thumbUrl: api.versionThumbUrl(project.id, versionId, 'train', n, rightFolder),
          })),
    [rightSortedNames, project.id, versionId, rightFolder]
  )

  const duplicateSuggested = useMemo(
    () =>
      dupResult
        ? dupResult.groups.flatMap((group) =>
            group.items.filter((item) => !item.keep).map((item) => item.name)
          )
        : [],
    [dupResult]
  )

  const duplicatePreviewNames = useMemo(
    () =>
      dupResult
        ? dupResult.groups.flatMap((group) => group.items.map((item) => item.name))
        : [],
    [dupResult]
  )

  const onLeftHover = useCallback(
    (name: string) =>
      setFocus({ side: 'left', name, url: api.projectThumbUrl(project.id, name, 'download', 768) }),
    [project.id]
  )

  const onRightHover = useCallback(
    (name: string) => {
      if (versionId == null || !rightFolder) return
      setFocus({
        side: 'right',
        folder: rightFolder,
        name,
        url: api.versionThumbUrl(project.id, versionId, 'train', name, rightFolder, 768),
      })
    },
    [versionId, project.id, rightFolder]
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

  const scanDuplicates = async () => {
    if (versionId == null || dupBusy) return
    setDupBusy(true)
    try {
      const result = await api.scanDuplicates(project.id, versionId, dupOptions)
      setDupResult(result)
      setDupSel(new Set(
        result.groups.flatMap((group) =>
          group.items.filter((item) => !item.keep).map((item) => item.name)
        )
      ))
      toast(t('curate.dupScanDone', { groups: result.group_count, candidates: result.candidate_count }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDupBusy(false)
    }
  }

  const openDuplicatePreview = (name: string) => {
    const list = duplicatePreviewNames.length ? duplicatePreviewNames : [name]
    setPreview({
      side: 'duplicate', name,
      url: api.projectThumbUrl(project.id, name, 'download', 1600),
      caption: name,
      list,
      index: Math.max(0, list.indexOf(name)),
      resolve: (n) => api.projectThumbUrl(project.id, n, 'download', 1600),
    })
  }

  const applyDuplicateAction = async (action: 'move' | 'delete') => {
    if (versionId == null || dupBusy || dupSel.size === 0) return
    const names = Array.from(dupSel)
    const ok = await dialog.confirm(
      action === 'move'
        ? t('curate.dupConfirmMove', { n: names.length })
        : t('curate.dupConfirmDelete', { n: names.length }),
      {
        tone: action === 'delete' ? 'danger' : 'warn',
        okText: action === 'move' ? t('curate.dupMoveOk') : t('curate.dupDeleteOk'),
      },
    )
    if (!ok) return
    setDupBusy(true)
    try {
      const result = await api.applyDuplicateAction(project.id, versionId, { action, names })
      const changed = action === 'move' ? result.moved.length : result.deleted.length
      toast(
        action === 'move'
          ? t('curate.dupMovedToast', { n: changed })
          : t('curate.dupDeletedToast', { n: changed }),
        'success',
      )
      setDupSel(new Set())
      setDupResult(null)
      await refresh()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDupBusy(false)
    }
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
      <DuplicateReviewPanel
        projectId={project.id}
        options={dupOptions}
        result={dupResult}
        selected={dupSel}
        busy={dupBusy}
        suggested={duplicateSuggested}
        onOptionsChange={setDupOptions}
        onScan={scanDuplicates}
        onSelect={setDupSel}
        onMove={() => applyDuplicateAction('move')}
        onDelete={() => applyDuplicateAction('delete')}
        onPreview={openDuplicatePreview}
      />

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
              : preview.side === 'right'
                ? t('curate.previewHintRight')
                : t('curate.previewHintDuplicate')
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

function DuplicateReviewPanel({
  projectId,
  options,
  result,
  selected,
  busy,
  suggested,
  onOptionsChange,
  onScan,
  onSelect,
  onMove,
  onDelete,
  onPreview,
}: {
  projectId: number
  options: DuplicateScanOptions
  result: DuplicateScanResult | null
  selected: Set<string>
  busy: boolean
  suggested: string[]
  onOptionsChange: (next: DuplicateScanOptions) => void
  onScan: () => void
  onSelect: (next: Set<string>) => void
  onMove: () => void
  onDelete: () => void
  onPreview: (name: string) => void
}) {
  const { t } = useTranslation()
  const patch = <K extends keyof DuplicateScanOptions>(key: K, value: DuplicateScanOptions[K]) => {
    onOptionsChange({ ...options, [key]: value })
  }
  const toggleName = (name: string) => {
    const next = new Set(selected)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    onSelect(next)
  }
  return (
    <section className="rounded-md border border-subtle bg-surface overflow-hidden shrink-0">
      <div className="h-0.5 bg-warn" />
      <header className="flex flex-wrap items-center gap-2 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">{t('curate.dupTitle')}</h3>
        <span className="text-xs text-fg-tertiary">
          {result
            ? t('curate.dupSummary', {
                groups: result.group_count,
                candidates: result.candidate_count,
                total: result.total_images,
              })
            : t('curate.dupSubtitle')}
        </span>
        <span className="flex-1" />
        <BtnSecondary
          onClick={() => onSelect(new Set(suggested))}
          disabled={busy || suggested.length === 0}
        >
          {t('curate.dupSelectSuggested')}
        </BtnSecondary>
        <BtnSecondary
          onClick={() => onSelect(new Set())}
          disabled={busy || selected.size === 0}
        >
          {t('curate.deselect')}
        </BtnSecondary>
        <BtnSecondary onClick={onMove} disabled={busy || selected.size === 0}>
          {t('curate.dupMoveBtn', { n: selected.size })}
        </BtnSecondary>
        <BtnDanger onClick={onDelete} disabled={busy || selected.size === 0}>
          {t('curate.dupDeleteBtn', { n: selected.size })}
        </BtnDanger>
      </header>

      <div className="grid grid-cols-1 2xl:grid-cols-[360px,1fr] gap-3 p-2">
        <div className="flex flex-col gap-2 text-sm">
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-fg-tertiary">{t('curate.dupTarget')}</span>
              <select
                className="input px-2 py-1 text-sm"
                value={options.target}
                onChange={(e) => patch('target', e.target.value as DuplicateScanOptions['target'])}
                disabled={busy}
              >
                <option value="unused">{t('curate.dupTargetUnused')}</option>
                <option value="download">{t('curate.dupTargetDownload')}</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-fg-tertiary">{t('curate.dupScope')}</span>
              <select
                className="input px-2 py-1 text-sm"
                value={options.match_scope}
                onChange={(e) => patch('match_scope', e.target.value as DuplicateScanOptions['match_scope'])}
                disabled={busy}
              >
                <option value="strict">{t('curate.dupScopeStrict')}</option>
                <option value="both">{t('curate.dupScopeBoth')}</option>
              </select>
            </label>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 2xl:grid-cols-2 gap-2">
            <NumOption
              label={t('curate.dupHashSize')}
              value={options.hash_size}
              min={0}
              max={2048}
              step={64}
              disabled={busy}
              onChange={(value) => patch('hash_size', value)}
            />
            <NumOption
              label={t('curate.dupWorkers')}
              value={options.hash_workers}
              min={1}
              max={32}
              step={1}
              disabled={busy}
              onChange={(value) => patch('hash_workers', value)}
            />
            <NumOption
              label={t('curate.dupStructure')}
              value={options.structure_threshold}
              min={0}
              max={24}
              step={1}
              disabled={busy}
              onChange={(value) => patch('structure_threshold', value)}
            />
            <NumOption
              label={t('curate.dupVariantScore')}
              value={options.variant_score}
              min={40}
              max={98}
              step={1}
              disabled={busy}
              onChange={(value) => patch('variant_score', value)}
            />
            <NumOption
              label={t('curate.dupAspect')}
              value={options.aspect_tolerance}
              min={0.005}
              max={0.2}
              step={0.005}
              disabled={busy}
              onChange={(value) => patch('aspect_tolerance', value)}
            />
            <NumOption
              label={t('curate.dupCloseTiles')}
              value={options.min_close_tiles}
              min={0}
              max={1}
              step={0.01}
              disabled={busy}
              onChange={(value) => patch('min_close_tiles', value)}
            />
            <NumOption
              label={t('curate.dupTileMedian')}
              value={options.tile_median}
              min={0}
              max={40}
              step={1}
              disabled={busy}
              onChange={(value) => patch('tile_median', value)}
            />
            <NumOption
              label={t('curate.dupGrayClose')}
              value={options.min_gray_close}
              min={0}
              max={1}
              step={0.01}
              disabled={busy}
              onChange={(value) => patch('min_gray_close', value)}
            />
          </div>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-fg-tertiary">{t('curate.dupTileGrids')}</span>
            <input
              className="input input-mono px-2 py-1 text-sm"
              value={options.tile_grids.join(',')}
              disabled={busy}
              onChange={(e) => {
                const grids = e.target.value
                  .split(',')
                  .map((part) => Number(part.trim()))
                  .filter((value) => Number.isFinite(value))
                patch('tile_grids', grids.length ? grids : options.tile_grids)
              }}
            />
          </label>
          <BtnPrimary onClick={onScan} disabled={busy}>
            {busy ? t('curate.dupScanning') : t('curate.dupScanBtn')}
          </BtnPrimary>
        </div>

        <div className="min-h-[160px] max-h-[38vh] overflow-y-auto pr-1">
          {!result ? (
            <p className="text-sm text-fg-tertiary py-2">{t('curate.dupEmpty')}</p>
          ) : result.groups.length === 0 ? (
            <p className="text-sm text-fg-tertiary py-2">
              {t('curate.dupNoGroups', { total: result.total_images })}
            </p>
          ) : (
            <div className="flex flex-col gap-2">
              {result.groups.map((group) => (
                <DuplicateGroupCard
                  key={group.group_id}
                  projectId={projectId}
                  group={group}
                  selected={selected}
                  onToggle={toggleName}
                  onPreview={onPreview}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function NumOption({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  disabled: boolean
  onChange: (value: number) => void
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-fg-tertiary">{label}</span>
      <input
        type="number"
        className="input input-mono px-2 py-1 text-sm"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

function DuplicateGroupCard({
  projectId,
  group,
  selected,
  onToggle,
  onPreview,
}: {
  projectId: number
  group: DuplicateGroup
  selected: Set<string>
  onToggle: (name: string) => void
  onPreview: (name: string) => void
}) {
  const { t } = useTranslation()
  return (
    <article className="rounded-md border border-subtle bg-sunken p-2">
      <div className="flex flex-wrap items-center gap-2 mb-2 text-xs">
        <span className="badge badge-neutral">#{group.group_id}</span>
        <span className="text-fg-secondary">
          {t('curate.dupKeepSuggested')} <code className="mono">{group.keep}</code>
        </span>
        {group.best && (
          <span className="badge badge-warn">
            {group.best.match_type} · {group.best.score}
          </span>
        )}
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(112px,1fr))] gap-1.5">
        {group.items.map((item) => (
          <DuplicateItemCell
            key={item.name}
            projectId={projectId}
            item={item}
            selected={selected.has(item.name)}
            onToggle={() => onToggle(item.name)}
            onPreview={() => onPreview(item.name)}
          />
        ))}
      </div>
    </article>
  )
}

function DuplicateItemCell({
  projectId,
  item,
  selected,
  onToggle,
  onPreview,
}: {
  projectId: number
  item: DuplicateItem
  selected: boolean
  onToggle: () => void
  onPreview: () => void
}) {
  const { t } = useTranslation()
  const metrics = item.metrics
  return (
    <div
      className={
        'group relative rounded-md border overflow-hidden bg-surface ' +
        (item.keep ? 'border-ok' : selected ? 'border-warn ring-2 ring-warn-soft' : 'border-subtle')
      }
    >
      <button
        type="button"
        onClick={onPreview}
        className="block w-full aspect-square bg-sunken"
        title={item.name}
      >
        <img
          src={api.projectThumbUrl(projectId, item.name, 'download', 256)}
          alt={item.name}
          loading="lazy"
          decoding="async"
          className="w-full h-full object-cover"
        />
      </button>
      <div className="p-1.5 flex flex-col gap-1 text-[11px]">
        <div className="flex items-center gap-1 min-w-0">
          {item.keep ? (
            <span className="badge badge-ok shrink-0">{t('curate.dupKeep')}</span>
          ) : (
            <button
              type="button"
              onClick={onToggle}
              className={`shrink-0 w-5 h-5 rounded-sm border text-[12px] font-bold ${
                selected ? 'bg-warn text-white border-warn' : 'bg-surface border-dim text-transparent'
              }`}
              aria-label={`${selected ? t('common.deselect') : t('common.select')} ${item.name}`}
            >
              ✓
            </button>
          )}
          <code className="mono truncate min-w-0">{item.name}</code>
        </div>
        <div className="text-fg-tertiary">
          {item.width}x{item.height} · {item.filesize_kb}KB
        </div>
        {metrics && (
          <div className="text-fg-tertiary truncate" title={metrics.note}>
            {metrics.match_type} · {metrics.score}
          </div>
        )}
      </div>
    </div>
  )
}

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
