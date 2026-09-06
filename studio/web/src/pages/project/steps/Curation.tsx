import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type CurationItem,
  type CurationValidationView,
  type CurationView,
  type ProjectDetail,
  type ValidationItem,
  type Version,
} from '../../../api/client'
import ActionGroup from '../../../components/ActionGroup'
import Alert from '../../../components/Alert'
import Button from '../../../components/Button'
import { Input, Select } from '../../../components/FormControl'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import PaneResizer, { clampPaneValue } from '../../../components/PaneResizer'
import StepShell from '../../../components/StepShell'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { compareImageName } from '../../../lib/imageSort'
import { useEventStream } from '../../../lib/useEventStream'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'

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

// 右栏目标桶：训练集（默认，现行为）/ 验证集（held-out，扁平无文件夹）
type Bucket = 'train' | 'validation'
const BUCKET_STORAGE_KEY = 'curation:bucket'
const CURATION_PANE_MIN = 25
const CURATION_PANE_MAX = 70
const CURATION_DOWNLOAD_PANE_ID = 'curation-download-pane'

function compareItems(a: CurationItem, b: CurationItem, mode: SortMode): number {
  switch (mode) {
    case 'id-asc':
    case 'id-desc': {
      const d = compareImageName(a.name, b.name)
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

export default function CurationPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const dialog = useDialog()
  const [view, setView] = useState<CurationView | null>(null)
  const [valView, setValView] = useState<CurationValidationView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const [bucket, setBucket] = useState<Bucket>(() => {
    if (typeof window === 'undefined') return 'train'
    const v = window.localStorage.getItem(BUCKET_STORAGE_KEY)
    return v === 'validation' ? 'validation' : 'train'
  })
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(BUCKET_STORAGE_KEY, bucket)
    }
  }, [bucket])

  const SORT_OPTIONS: { value: SortMode; label: string }[] = [
    { value: 'id-asc', label: 'ID ↑' },
    { value: 'id-desc', label: 'ID ↓' },
    { value: 'name-asc', label: t('common.filename') + ' ↑' },
    { value: 'name-desc', label: t('common.filename') + ' ↓' },
    { value: 'mtime-asc', label: t('curate.downloadTime') + ' ↑' },
    { value: 'mtime-desc', label: t('curate.downloadTime') + ' ↓' },
  ]

  // 左栏（候选池）占整行宽度的百分比，中间分隔条可拖；右栏吃剩余空间。
  // 默认 50 = 改造前的 xl:grid-cols-2 等分。窄屏（<1280px）堆叠时该值不生效。
  const rowRef = useRef<HTMLDivElement>(null)
  const [leftPct, setLeftPct] = useLocalStorageState('studio:curate:left_pct', 50)
  const boundedLeftPct = clampPaneValue(leftPct, CURATION_PANE_MIN, CURATION_PANE_MAX)

  // Repair stale/corrupt persisted ratios before they can collapse the flexible pane.
  useEffect(() => {
    if (leftPct !== boundedLeftPct) setLeftPct(boundedLeftPct)
  }, [boundedLeftPct, leftPct, setLeftPct])

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
      if (isAlt(e)) setAltHeld(true)
    }
    const up = (e: KeyboardEvent) => {
      if (isAlt(e)) setAltHeld(false)
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

  const fetchTrain = useCallback(async () => {
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

  const fetchValidation = useCallback(async () => {
    if (versionId == null) return
    try {
      const vv = await api.getCurationValidation(project.id, versionId)
      setValView(vv)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [project.id, versionId])

  // 切换 bucket 只在目标视图没缓存时拉一次：左栏 download−train−validation 两个
  // bucket 完全一致，纯来回切不该重拉、不该整页 loading（左栏用已有数据兜底）。
  useEffect(() => {
    if (versionId == null) return
    if (bucket === 'validation') {
      if (valView == null) void fetchValidation()
    } else if (view == null) {
      void fetchTrain()
    }
  }, [bucket, versionId, view, valView, fetchTrain, fetchValidation])

  // 改动后刷新当前 bucket + 作废另一个的缓存：跨 bucket 增删会改共享的左栏候选
  // 池 / 另一侧右栏，下次切过去自动重拉，保证不显示陈旧的左栏。
  const refresh = useCallback(async () => {
    if (bucket === 'validation') {
      await fetchValidation()
      setView(null)
    } else {
      await fetchTrain()
      setValView(null)
    }
  }, [bucket, fetchTrain, fetchValidation])

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

  const isVal = bucket === 'validation'
  const folderNames = view?.folders ?? []

  // 左栏候选 download − train − validation，两个 bucket 共用同一池 —— 目标视图
  // 尚未加载时回退到另一视图的（内容相同的）左栏，切换时左栏不空屏、不闪。
  const currentLeft = useMemo(
    () => (isVal ? valView?.left ?? view?.left ?? [] : view?.left ?? valView?.left ?? []),
    [isVal, valView, view]
  )
  const leftSortedNames = useMemo(
    () => sortItems(currentLeft, sortMode).map((e) => e.name),
    [currentLeft, sortMode]
  )

  // train 右栏：当前文件夹的 entries（带 origin）。validation 右栏：全量扁平。
  const trainEntries = useMemo(
    () => (view && rightFolder ? view.right[rightFolder] ?? [] : []),
    [view, rightFolder]
  )
  const valEntries = useMemo<ValidationItem[]>(() => valView?.right ?? [], [valView])
  const rightSortedNames = useMemo(
    () => sortItems(isVal ? valEntries : trainEntries, sortMode).map((e) => e.name),
    [isVal, valEntries, trainEntries, sortMode]
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
  // validation 图无 manifest / 无 origin：按 name → 物理 folder 反查，供
  // 缩略图寻址（version thumb 的 validation bucket 需 folder）+ 精确删除。
  const valFolderByName = useMemo(() => {
    const m = new Map<string, string>()
    for (const e of valEntries) m.set(e.name, e.folder)
    return m
  }, [valEntries])

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
            thumbUrl: isVal
              ? api.versionThumbUrl(
                  project.id, versionId, 'validation', n, valFolderByName.get(n), 256,
                )
              : api.projectThumbUrl(
                  project.id, rightOriginByName.get(n) ?? n, 'download', 256,
                  undefined, true,
                ),
          })),
    [rightSortedNames, project.id, versionId, isVal, rightOriginByName, valFolderByName]
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
      if (versionId == null) return
      if (isVal) {
        const folder = valFolderByName.get(name)
        if (!folder) return
        setFocus({
          side: 'right', folder, name,
          url: api.versionThumbUrl(project.id, versionId, 'validation', name, folder, 768),
        })
        return
      }
      if (!rightFolder) return
      const origin = rightOriginByName.get(name) ?? name
      setFocus({
        side: 'right',
        folder: rightFolder,
        name,
        url: api.projectThumbUrl(project.id, origin, 'download', 768, undefined, true),
      })
    },
    [versionId, project.id, isVal, valFolderByName, rightFolder, rightOriginByName]
  )

  if (!activeVersion) {
    return <p className="text-fg-tertiary p-6">{t('curate.noVersion')}</p>
  }
  // A first-load failure blocks the workspace because there is no trustworthy
  // dataset to operate on. Later refresh failures keep the last good grids
  // mounted and are rendered as a non-blocking alert below.
  if (error && view == null && valView == null) {
    return (
      <StepShell title={t('steps.curate.title')} subtitle={t('steps.curate.subtitle')}>
        <Alert
          tone="danger"
          title={t('curate.loadErrorTitle')}
          action={
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void (isVal ? fetchValidation() : fetchTrain())}
            >
              {t('common.retry')}
            </Button>
          }
          role="alert"
        >
          {error}
        </Alert>
      </StepShell>
    )
  }
  // 整页 loading 只在首次（两个视图都没数据）出现；切换 bucket 时另一视图已有
  // 数据，页面照常渲染，只有右栏在等自己那侧的数据。
  if (view == null && valView == null) {
    return <p className="text-fg-tertiary p-6">{t('curate.loading')}</p>
  }

  // 当前 bucket 自己那侧的右栏数据是否还在加载（切到一个从没看过的 bucket 时）
  const rightLoading = isVal ? valView == null : view == null
  const downloadTotal = isVal
    ? valView?.download_total ?? view?.download_total ?? 0
    : view?.download_total ?? valView?.download_total ?? 0

  const switchBucket = (next: Bucket) => {
    if (next === bucket) return
    setError(null)
    setBucket(next)
    setLeftSel(new Set())
    setLeftAnchor(null)
    setRightSel(new Set())
    setRightAnchor(null)
    setRenaming(null)
  }

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
    if (files.length === 0 || busy) return false
    if (!isVal) {
      if (!rightFolder) { toast(t('curate.noTargetFolder'), 'error'); return false }
      if (!FOLDER_PATTERN.test(rightFolder)) { toast(t('curate.invalidFolder'), 'error'); return false }
    }
    setBusy(true)
    try {
      const r = isVal
        ? await api.copyToValidation(project.id, activeVersion.id, { files })
        : await api.copyToTrain(project.id, activeVersion.id, { files, dest_folder: rightFolder })
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
    if (files.length === 0 || busy) return false
    if (!isVal && !folder) return false
    const folderLabel = isVal ? t('curate.bucketValidation') : folder
    if (options.confirm &&
        !(await dialog.confirm(t('curate.confirmRemove', { folder: folderLabel, n: files.length }), { tone: 'warn', okText: t('curate.removeOkText') }))) {
      return false
    }
    setBusy(true)
    try {
      const r = isVal
        ? await api.removeFromValidation(project.id, activeVersion.id, {
            items: files
              .map((name) => ({ folder: valFolderByName.get(name) ?? '', name }))
              .filter((it) => it.folder),
          })
        : await api.removeFromTrain(project.id, activeVersion.id, { folder, files })
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
    const cnt = view?.right[name]?.length ?? 0
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
    if (isVal) {
      const folder = valFolderByName.get(name) ?? ''
      setPreview({
        side: 'right', name, folder,
        url: api.versionThumbUrl(project.id, versionId, 'validation', name, folder, 1600),
        caption: name,
        list: rightSortedNames,
        index: rightSortedNames.indexOf(name),
        resolve: (n) =>
          api.versionThumbUrl(project.id, versionId, 'validation', n, valFolderByName.get(n) ?? '', 1600),
      })
      return
    }
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
      caption: preview.side === 'right' && preview.folder && !isVal ? `${preview.folder}/${name}` : name,
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
      caption: preview.side === 'right' && preview.folder && !isVal ? `${preview.folder}/${name}` : name,
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

  const addActionLabel = isVal
    ? t('curate.copyToValBtn', { n: leftSel.size })
    : rightFolder
      ? t('curate.copyToBtn', { n: leftSel.size, folder: rightFolder })
      : t('curate.copyToTrainBtn', { n: leftSel.size })
  const removeActionLabel = isVal
    ? t('curate.removeFromValidationBtn', { n: rightSel.size })
    : t('curate.removeFromTrainBtn', {
        n: rightSel.size,
        folder: rightFolder || t('curate.bucketTrain'),
      })

  return (
    <StepShell
      title={t('steps.curate.title')}
      subtitle={t('steps.curate.subtitle')}
      actions={
        <ActionGroup
          aria-label={t('curate.pageActionsLabel')}
          secondary={
            <label className="flex items-center gap-1.5 text-sm text-fg-secondary whitespace-nowrap shrink-0">
              {t('curate.sortLabel')}
              <Select
                controlSize="sm"
                value={sortMode}
                onChange={(e) => setSortMode(e.target.value as SortMode)}
                title={t('curate.sortTitle')}
              >
                {SORT_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </Select>
            </label>
          }
          primary={
            <label className="flex items-center gap-1.5 text-sm font-medium text-fg whitespace-nowrap shrink-0">
              {t('curate.destinationLabel')}
              <Select
                controlSize="sm"
                value={bucket}
                onChange={(e) => switchBucket(e.target.value as Bucket)}
                data-testid="curate-destination-select"
              >
                <option value="train">{t('curate.destinationTrain')}</option>
                <option value="validation">{t('curate.destinationValidation')}</option>
              </Select>
            </label>
          }
        />
      }
    >
    <div className="flex flex-col h-full gap-3 min-h-0">
      {error && (
        <Alert
          tone="warning"
          size="sm"
          title={t('curate.refreshErrorTitle')}
          action={
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void (isVal ? fetchValidation() : fetchTrain())}
            >
              {t('common.retry')}
            </Button>
          }
          role="alert"
        >
          {error}
        </Alert>
      )}

      <div
        ref={rowRef}
        className="split-row gap-3 items-stretch flex-1 min-h-0"
        style={{ '--split-pct': `${boundedLeftPct}%` } as React.CSSProperties}
      >
        <PanelCard
          id={CURATION_DOWNLOAD_PANE_ID}
          className="split-pane-fixed"
          accent="emerald"
          title={t('curate.downloadPanelTitle')}
          subtitle={t('curate.downloadSubtitle', { unused: currentLeft.length, total: downloadTotal, sel: leftSel.size })}
          actions={
            <ActionGroup
              aria-label={t('curate.sourceActionsLabel')}
              secondary={
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setLeftSel(new Set(leftSortedNames))}
                    disabled={busy || leftSortedNames.length === 0}
                  >
                    {t('curate.selectAll')}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setLeftSel(new Set())}
                    disabled={busy || leftSel.size === 0}
                  >
                    {t('curate.deselect')}
                  </Button>
                </>
              }
              primary={
                <Button
                  variant="primary"
                  size="sm"
                  onClick={doCopy}
                  disabled={busy || leftSel.size === 0 || (!isVal && !rightFolder)}
                  aria-label={addActionLabel}
                  title={isVal
                    ? t('curate.copyToValTitle')
                    : rightFolder
                      ? t('curate.copyToTitle', { folder: rightFolder })
                      : t('curate.noFolderTitle')}
                >
                  {t('curate.addSelectedBtn', { n: leftSel.size })}
                </Button>
              }
            />
          }
        >
          <ImageGrid
            className="flex-1 min-h-0"
            contentClassName="p-2"
            items={leftItems}
            selected={leftSel}
            activeName={preview?.side === 'left' ? preview.name : undefined}
            onSelect={handleLeftClick}
            onHover={onLeftHover}
            onPreview={openLeftPreview}
            onActivate={openLeftPreview}
            clickMode="activate"
            ariaLabel={t('curate.downloadGridLabel')}
            emptyHint={t('curate.downloadEmptyHint')}
          />
        </PanelCard>

        <PaneResizer
          containerRef={rowRef}
          value={boundedLeftPct}
          onChange={setLeftPct}
          min={CURATION_PANE_MIN}
          max={CURATION_PANE_MAX}
          ariaLabel={t('curate.resizePanels')}
          ariaControls={CURATION_DOWNLOAD_PANE_ID}
          className="split-resizer"
        />

        <PanelCard
          className="split-pane-flex"
          accent="cyan"
          title={isVal ? t('curate.valPanelTitle') : t('curate.trainPanelTitle')}
          subtitle={
            isVal
              ? t('curate.valSubtitle', { total: valView?.val_total ?? 0, sel: rightSel.size })
              : t('curate.trainSubtitle', { total: view?.train_total ?? 0, folders: folderNames.length, sel: rightSel.size })
          }
          actions={
            <ActionGroup
              aria-label={t('curate.destinationActionsLabel')}
              secondary={
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setRightSel(new Set(rightSortedNames))}
                    disabled={busy || rightSortedNames.length === 0}
                  >
                    {t('curate.selectAll')}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setRightSel(new Set())}
                    disabled={busy || rightSel.size === 0}
                  >
                    {t('curate.deselect')}
                  </Button>
                </>
              }
              primary={
                <Button
                  variant="danger"
                  size="sm"
                  onClick={doRemove}
                  disabled={busy || rightSel.size === 0 || (!isVal && !rightFolder)}
                  aria-label={removeActionLabel}
                  title={removeActionLabel}
                >
                  {t('curate.removeSelectedBtn', { n: rightSel.size })}
                </Button>
              }
            />
          }
        >
          {isVal ? (
            <p className="px-2 pt-2 text-xs text-fg-tertiary">{t('curate.valHint')}</p>
          ) : (
            <div className="px-2 pt-2 flex flex-wrap items-center gap-related">
              <FolderSummary
                folders={folderNames}
                counts={Object.fromEntries(folderNames.map((f) => [f, view?.right[f]?.length ?? 0]))}
                activeFolder={rightFolder}
                busy={busy}
                onSwitch={switchRightFolder}
                onRename={(name) => setRenaming({ target: name, value: name })}
                onDelete={doDeleteFolder}
              />

              <form
                className="ml-auto flex flex-wrap items-center gap-related"
                aria-label={t('curate.createFolderFormLabel')}
                onSubmit={(e) => {
                  e.preventDefault()
                  void doCreateFolder()
                }}
              >
                <label className="sr-only" htmlFor="curation-new-folder">
                  {t('curate.newFolderLabel')}
                </label>
                <Input
                  id="curation-new-folder"
                  controlSize="sm"
                  mono
                  value={newFolder}
                  onChange={(e) => setNewFolder(e.target.value)}
                  placeholder={t('curate.newFolderPlaceholder')}
                  className="w-40 max-w-full"
                />
                <Button type="submit" variant="secondary" size="sm" disabled={busy || !newFolder.trim()}>
                  {t('curate.createFolderBtn')}
                </Button>
              </form>

              {renaming && (
                <form
                  className="basis-full flex flex-wrap items-center gap-related text-sm"
                  aria-label={t('curate.renameFormLabel', { name: renaming.target })}
                  onSubmit={(e) => {
                    e.preventDefault()
                    void doRenameFolder()
                  }}
                >
                  <label className="text-fg-secondary" htmlFor="curation-rename-folder">
                    {t('curate.renameLabel', { name: renaming.target })}
                  </label>
                  <Input
                    id="curation-rename-folder"
                    controlSize="sm"
                    mono
                    autoFocus
                    value={renaming.value}
                    onChange={(e) => setRenaming({ ...renaming, value: e.target.value })}
                    onKeyDown={(e) => {
                      if (e.key === 'Escape') setRenaming(null)
                    }}
                    className="w-44 max-w-full"
                  />
                  <Button type="submit" variant="primary" size="sm" disabled={busy}>
                    {t('curate.renameOk')}
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setRenaming(null)}>
                    {t('common.cancel')}
                  </Button>
                </form>
              )}
            </div>
          )}

          <ImageGrid
            className="flex-1 min-h-0 mt-3"
            contentClassName="px-2 pb-2"
            items={rightItems}
            selected={rightSel}
            activeName={preview?.side === 'right' ? preview.name : undefined}
            onSelect={handleRightClick}
            onHover={onRightHover}
            onPreview={openRightPreview}
            onActivate={openRightPreview}
            clickMode="activate"
            ariaLabel={isVal ? t('curate.validationGridLabel') : t('curate.trainGridLabel')}
            emptyHint={
              rightLoading
                ? t('curate.loading')
                : isVal
                  ? t('curate.valEmpty')
                  : rightFolder
                    ? t('curate.trainEmptyFolder', { folder: rightFolder })
                    : folderNames.length === 0
                      ? t('curate.noTrainFolders')
                      : t('curate.trainNoFolder')
            }
          />
        </PanelCard>
      </div>

      {altHeld && focus && <AltHoverPreview focus={focus} isVal={isVal} />}

      {preview && (
        <ImagePreviewModal
          src={preview.url}
          caption={preview.caption}
          index={preview.index}
          total={preview.list.length}
          hasPrev={preview.index > 0}
          hasNext={preview.index < preview.list.length - 1}
          onClose={() => setPreview(null)}
          onPrev={() => stepPreview(-1)}
          onNext={() => stepPreview(1)}
          onAccept={preview.side === 'left' ? copyPreviewImage : undefined}
          onDelete={preview.side === 'right' ? removePreviewImage : undefined}
          shortcutHint={
            preview.side === 'left'
              ? isVal
                ? t('curate.previewHintAddValidation')
                : t('curate.previewHintAddTrain')
              : isVal
                ? t('curate.previewHintRemoveValidation')
                : t('curate.previewHintRemoveTrain')
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
  if (folders.length === 0) return null
  const total = folders.reduce((s, f) => s + (counts[f] ?? 0), 0)
  return (
    <div
      className="flex flex-wrap items-center gap-1.5 text-sm"
      role="group"
      aria-label={t('curate.folderListLabel')}
    >
      {folders.map((f) => {
        const isActive = f === activeFolder
        return (
          <span
            key={f}
            className="inline-flex items-center gap-0.5"
          >
            <Button
              variant="ghost"
              size="xs"
              onClick={() => onSwitch(f)}
              aria-pressed={isActive}
              title={isActive ? t('curate.folderActiveTitle') : t('curate.folderSwitchTitle')}
              className={isActive ? 'text-accent' : 'text-fg-secondary'}
            >
              <span className="font-mono">{f}</span>
              <span className="text-fg-tertiary">({counts[f] ?? 0})</span>
            </Button>
            <Button
              variant="ghost"
              size="xs"
              iconOnly
              onClick={() => onRename(f)}
              disabled={busy}
              aria-label={t('curate.renameFolderAction', { name: f })}
              className="opacity-70 hover:opacity-100 focus:opacity-100"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 20h9" />
                <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z" />
              </svg>
            </Button>
            <Button
              variant="ghost"
              size="xs"
              iconOnly
              onClick={() => onDelete(f)}
              disabled={busy}
              aria-label={t('curate.deleteFolderAction', { name: f })}
              className="text-err opacity-70 hover:opacity-100 focus:opacity-100"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M3 6h18" />
                <path d="M8 6V4h8v2" />
                <path d="M19 6l-1 14H6L5 6" />
                <path d="M10 11v5M14 11v5" />
              </svg>
            </Button>
          </span>
        )
      })}
      <span className="text-fg-tertiary ml-2">{t('curate.folderTotal', { total })}</span>
    </div>
  )
}

function AltHoverPreview({ focus, isVal }: { focus: Focus; isVal: boolean }) {
  const { t } = useTranslation()
  const sourceLabel = focus.side === 'left'
    ? t('curate.sourceLabelDownload')
    : isVal
      ? t('curate.bucketValidation')
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
  id, accent, title, subtitle, actions, children, className = '',
}: {
  id?: string
  accent: 'emerald' | 'cyan'
  title: string
  subtitle: string
  actions: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <section id={id} className={`flex flex-col min-h-0 rounded-md border border-subtle bg-surface overflow-hidden ${className}`}>
      <div className={`h-0.5 ${ACCENT_BAR_CLS[accent]}`} />
      <header className="flex flex-wrap items-center gap-1.5 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">{title}</h3>
        <span className="text-xs text-fg-tertiary">{subtitle}</span>
        <span className="flex-1" />
        {actions}
      </header>
      <div className="flex-1 min-h-0 flex flex-col">{children}</div>
    </section>
  )
}
