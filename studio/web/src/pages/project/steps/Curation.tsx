import { useCallback, useEffect, useMemo, useState } from 'react'
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

const SORT_OPTIONS: { value: SortMode; label: string }[] = [
  { value: 'id-asc', label: 'ID ↑' },
  { value: 'id-desc', label: 'ID ↓' },
  { value: 'name-asc', label: '文件名 ↑' },
  { value: 'name-desc', label: '文件名 ↓' },
  { value: 'mtime-asc', label: '下载时间 ↑' },
  { value: 'mtime-desc', label: '下载时间 ↓' },
]

const SORT_STORAGE_KEY = 'curation:sort'
const DEFAULT_SORT: SortMode = 'id-asc'

/**
 * 取文件名「数字 id」key —— booru 默认下载文件名是 `<post_id>.<ext>`，stem 整段
 * 是数字时按数字排；否则给一个超大值，让非数字名排到末尾再按 name 兜底。
 */
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

/**
 * 把后端可能返回的两种形状统一成 CurationItem：
 * - 新格式：`{name, mtime}`
 * - 老格式（后端尚未升级 / 字段缺失兜底）：纯 string 文件名 → mtime 取 0
 *
 * 这样前端 sort 不会因为 `a.name` 为 undefined 直接崩。
 */
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

// 网格内部滚动：让面板撑满外层可用高度（外层是 flex-col h-full），
// 页面头 / 共享预览 / 面板 header 不动，仅图片区域滚。
const SCROLL_BOX = 'flex-1 min-h-0 overflow-y-auto pr-1'

export default function CurationPage() {
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  // Curation 有 `options.confirm: boolean` 字段,跟 useDialog().confirm 同名 — 不解构,用 dialog.confirm。
  const dialog = useDialog()
  const [view, setView] = useState<CurationView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // 选中状态
  const [leftSel, setLeftSel] = useState<Set<string>>(new Set())
  const [leftAnchor, setLeftAnchor] = useState<string | null>(null)
  const [rightFolder, setRightFolder] = useState<string>('')
  const [rightSel, setRightSel] = useState<Set<string>>(new Set())
  const [rightAnchor, setRightAnchor] = useState<string | null>(null)

  // alt + hover 触发的悬浮大图预览
  const [focus, setFocus] = useState<Focus | null>(null)
  const [altHeld, setAltHeld] = useState(false)
  useEffect(() => {
    // 浏览器默认行为：单按 Alt 会聚焦菜单栏，抢走页面键盘焦点，导致后续
    // Alt keydown/keyup 事件不再触发 window 监听器（必须点回页面才能恢复）。
    // 这里 preventDefault 阻止默认菜单激活；同时用 mousemove 的 altKey 兜底同步，
    // 万一焦点真的被抢走，鼠标一动也能重新拿到正确状态。
    const isAlt = (e: KeyboardEvent) =>
      e.key === 'Alt' || e.code === 'AltLeft' || e.code === 'AltRight'
    const down = (e: KeyboardEvent) => {
      if (isAlt(e)) {
        e.preventDefault()
        setAltHeld(true)
      }
    }
    const up = (e: KeyboardEvent) => {
      if (isAlt(e)) {
        e.preventDefault()
        setAltHeld(false)
      }
    }
    const move = (e: MouseEvent) => {
      // 鼠标事件随时带最新的 altKey 状态，作为 keyboard 监听的兜底
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

  // 复制目标 = rightFolder（当前查看的就是复制目标），不再单独维护。
  const [newFolder, setNewFolder] = useState<string>('')
  const [renaming, setRenaming] = useState<{
    target: string
    value: string
  } | null>(null)
  const [preview, setPreview] = useState<Preview | null>(null)

  // 排序模式（左右两 grid 共享）；持久化到 localStorage 让用户偏好跨页保留。
  const [sortMode, setSortMode] = useState<SortMode>(() => {
    if (typeof window === 'undefined') return DEFAULT_SORT
    const v = window.localStorage.getItem(SORT_STORAGE_KEY)
    return SORT_OPTIONS.some((o) => o.value === v)
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
      const fallback = v.folders.includes('1_data')
        ? '1_data'
        : v.folders[0] ?? ''
      if (!rightFolder || !v.folders.includes(rightFolder)) {
        setRightFolder(fallback)
        setRightSel(new Set())
        setRightAnchor(null)
      }
    } catch (e) {
      setError(String(e))
    }
  }, [project.id, versionId, rightFolder])

  useEffect(() => {
    void refresh()
  }, [refresh])

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

  // sortMode 应用到左右两侧后得到「显示顺序」的名字数组；范围选择 / preview
  // 翻页 / 全选 都基于这个顺序。
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
    () =>
      leftSortedNames.map((n) => ({
        name: n,
        thumbUrl: api.projectThumbUrl(project.id, n),
      })),
    [leftSortedNames, project.id]
  )
  const rightItems = useMemo(
    () =>
      versionId == null
        ? []
        : rightSortedNames.map((n) => ({
            name: n,
            thumbUrl: api.versionThumbUrl(
              project.id,
              versionId,
              'train',
              n,
              rightFolder
            ),
          })),
    [rightSortedNames, project.id, versionId, rightFolder]
  )

  // 大图预览用 768px 缓存版本（足够清晰，文件比原图小一两个数量级）
  // 这两个 useCallback 必须在所有 early-return 之前调用，否则不同 render
  // 之间 hook 数量会变 → React #310。
  const onLeftHover = useCallback(
    (name: string) =>
      setFocus({
        side: 'left',
        name,
        url: api.projectThumbUrl(project.id, name, 'download', 768),
      }),
    [project.id]
  )

  const onRightHover = useCallback(
    (name: string) => {
      if (versionId == null || !rightFolder) return
      setFocus({
        side: 'right',
        folder: rightFolder,
        name,
        url: api.versionThumbUrl(
          project.id,
          versionId,
          'train',
          name,
          rightFolder,
          768
        ),
      })
    },
    [versionId, project.id, rightFolder]
  )

  if (!activeVersion) {
    return (
      <p className="text-fg-tertiary p-6">
        请先选择 / 创建一个版本（左上 VersionTabs）
      </p>
    )
  }
  if (error) {
    return (
      <div className="p-3 rounded-md bg-err-soft border border-err text-err font-mono text-sm">
        {error}
      </div>
    )
  }
  if (!view) return <p className="text-fg-tertiary p-6">加载...</p>

  const switchRightFolder = (next: string) => {
    setRightFolder(next)
    setRightSel(new Set())
    setRightAnchor(null)
  }

  // ---------- handlers ----------
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

  // ---------- copy / remove / folder ops ----------
  const copyLeftFiles = async (
    files: string[],
    options: { clearSelection?: boolean } = {}
  ) => {
    if (!rightFolder) {
      toast('请先在右侧 Train 选一个文件夹', 'error')
      return false
    }
    if (!FOLDER_PATTERN.test(rightFolder)) {
      toast('文件夹名非法', 'error')
      return false
    }
    if (files.length === 0 || busy) return false
    setBusy(true)
    try {
      const r = await api.copyToTrain(project.id, activeVersion.id, {
        files,
        dest_folder: rightFolder,
      })
      toast(
        `已复制 ${r.copied.length} 张${
          r.skipped.length ? `（跳过 ${r.skipped.length}）` : ''
        }`,
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
        !(await dialog.confirm(`从 ${folder}/ 移除 ${files.length} 张?`, { tone: 'warn', okText: '移除' }))) {
      return false
    }
    setBusy(true)
    try {
      const r = await api.removeFromTrain(project.id, activeVersion.id, {
        folder,
        files,
      })
      toast(`已移除 ${r.removed.length} 张`, 'success')
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
    await removeRightFiles(rightFolder, Array.from(rightSel), {
      clearSelection: true,
      confirm: true,
    })
  }

  const doCreateFolder = async () => {
    const name = newFolder.trim()
    if (!name) return
    if (!FOLDER_PATTERN.test(name)) return toast('文件夹名非法', 'error')
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, {
        op: 'create',
        name,
      })
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
    if (!next || next === target) {
      setRenaming(null)
      return
    }
    if (!FOLDER_PATTERN.test(next)) return toast('文件夹名非法', 'error')
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, {
        op: 'rename',
        name: target,
        new_name: next,
      })
      if (rightFolder === target) switchRightFolder(next)
      setRenaming(null)
      toast(`${target} → ${next}`, 'success')
      await refresh()
      // 关键：reload 项目 context，让 activeVersion.stats.train_folders 跟上磁盘真实
      // 名字。否则 Train 页仍按旧 folder 名解析 N_label，repeat 显示错（没生效是
      // 误解，启动训练时后端按真实磁盘读，但前端展示错本身就是 bug）。
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
      `删除文件夹 ${name}? 将清掉 ${cnt} 张训练副本（download/ 不动）`,
      { tone: 'warn', okText: '删文件夹' },
    ))) return
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, {
        op: 'delete',
        name,
      })
      if (rightFolder === name) switchRightFolder('')
      await refresh()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  // ---------- modal preview ----------
  const openLeftPreview = (name: string) => {
    setPreview({
      side: 'left',
      name,
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
      side: 'right',
      name,
      folder,
      url: api.versionThumbUrl(
        project.id,
        versionId,
        'train',
        name,
        folder,
        1600
      ),
      caption: `${folder}/${name}`,
      list: rightSortedNames,
      index: rightSortedNames.indexOf(name),
      resolve: (n) =>
        api.versionThumbUrl(project.id, versionId, 'train', n, folder, 1600),
    })
  }
  const stepPreview = (delta: number) => {
    if (!preview) return
    const idx = preview.index + delta
    if (idx < 0 || idx >= preview.list.length) return
    const name = preview.list[idx]
    setPreview({
      ...preview,
      name,
      url: preview.resolve(name),
      caption: preview.side === 'right' && preview.folder ? `${preview.folder}/${name}` : name,
      index: idx,
    })
  }

  const advancePreviewAfterAction = (doneName: string) => {
    if (!preview) return
    const list = preview.list.filter((name) => name !== doneName)
    if (list.length === 0) {
      setPreview(null)
      return
    }
    const index = Math.min(preview.index, list.length - 1)
    const name = list[index]
    setPreview({
      ...preview,
      name,
      url: preview.resolve(name),
      caption: preview.side === 'right' && preview.folder ? `${preview.folder}/${name}` : name,
      list,
      index,
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
      title="筛选图片"
      subtitle="download → train"
      actions={
        <label className="flex items-center gap-1.5 text-sm text-fg-secondary whitespace-nowrap shrink-0">
          排序
          <select
            value={sortMode}
            onChange={(e) => setSortMode(e.target.value as SortMode)}
            className="input px-2 py-0.5 text-sm"
            title="排序作用于左右两个网格"
          >
            {SORT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
      }
    >
    <div className="flex flex-col h-full gap-3">

      {/* Download + Train 两列平分整宽；预览改为 alt+hover 浮层，不占布局位置。 */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 items-stretch flex-1 min-h-0">
        <PanelCard
          accent="emerald"
          title="Download — 全量备份"
          subtitle={`${view.left.length} 未用 / ${view.download_total} 全量 · 已选 ${leftSel.size}`}
          actions={
            <>
              <BtnSecondary
                onClick={() => setLeftSel(new Set(leftSortedNames))}
                disabled={busy || leftSortedNames.length === 0}
              >
                全选
              </BtnSecondary>
              <BtnSecondary
                onClick={() => setLeftSel(new Set())}
                disabled={busy || leftSel.size === 0}
              >
                清空
              </BtnSecondary>
              <BtnPrimary
                onClick={doCopy}
                disabled={busy || leftSel.size === 0 || !rightFolder}
                title={
                  rightFolder
                    ? `复制到 train/${rightFolder}/`
                    : '请先在右侧 Train 选一个文件夹'
                }
              >
                → 复制 {leftSel.size} → {rightFolder || '?'}
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
              emptyHint="download/ 已经全部用完，或还没下载"
            />
          </div>
        </PanelCard>

        <PanelCard
          accent="cyan"
          title="Train — 当前版本"
          subtitle={`${view.train_total} 张 · ${folderNames.length} 个文件夹 · 已选 ${rightSel.size}`}
          actions={
            <>
              <input
                value={newFolder}
                onChange={(e) => setNewFolder(e.target.value)}
                placeholder="+ 新建:5_concept"
                className="input input-mono px-2 py-0.5 text-sm"
                style={{ width: 144 }}
              />
              <BtnSecondary
                onClick={doCreateFolder}
                disabled={busy || !newFolder.trim()}
              >
                创建
              </BtnSecondary>
              <BtnSecondary
                onClick={() => setRightSel(new Set(rightSortedNames))}
                disabled={busy || rightSortedNames.length === 0}
              >
                全选
              </BtnSecondary>
              <BtnSecondary
                onClick={() => setRightSel(new Set())}
                disabled={busy || rightSel.size === 0}
              >
                清空
              </BtnSecondary>
              <BtnDanger
                onClick={doRemove}
                disabled={busy || rightSel.size === 0 || !rightFolder}
              >
                ← 移除 {rightSel.size}
              </BtnDanger>
            </>
          }
        >
          {/* 文件夹 chip 行：active = 当前查看 = 复制目标；hover 显示 ✎/× */}
          <FolderSummary
            folders={folderNames}
            counts={Object.fromEntries(
              folderNames.map((f) => [f, view.right[f]?.length ?? 0])
            )}
            activeFolder={rightFolder}
            busy={busy}
            onSwitch={switchRightFolder}
            onRename={(name) => setRenaming({ target: name, value: name })}
            onDelete={doDeleteFolder}
          />

          {renaming && (
            <div className="flex items-center gap-2 my-3 text-sm">
              <span className="text-fg-secondary">改名 {renaming.target} →</span>
              <input
                autoFocus
                value={renaming.value}
                onChange={(e) =>
                  setRenaming({ ...renaming, value: e.target.value })
                }
                onKeyDown={(e) => {
                  if (e.key === 'Enter') doRenameFolder()
                  if (e.key === 'Escape') setRenaming(null)
                }}
                className="input input-mono px-2 py-0.5"
                style={{ width: 176 }}
              />
              <BtnPrimary onClick={doRenameFolder} disabled={busy}>
                确认
              </BtnPrimary>
              <button
                onClick={() => setRenaming(null)}
                className="btn btn-ghost btn-sm"
              >
                取消
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
                  ? `${rightFolder}/ 还是空的`
                  : '上方点一个文件夹 chip 切换查看'
              }
            />
          </div>
        </PanelCard>
      </div>

      {/* alt + hover 触发的浮层大图：pointer-events-none 让 hover 事件继续命中底层缩略图，
       * 用户按住 alt 在网格上滑动时，预览随 focus 切换，不阻塞选择。 */}
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
              ? '←/→ 浏览 · Enter/Space 复制到 Train'
              : '←/→ 浏览 · Delete/Backspace 从 Train 移除'
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
  folders,
  counts,
  activeFolder,
  busy,
  onSwitch,
  onRename,
  onDelete,
}: {
  folders: string[]
  counts: Record<string, number>
  activeFolder: string
  busy: boolean
  onSwitch: (name: string) => void
  onRename: (name: string) => void
  onDelete: (name: string) => void
}) {
  if (folders.length === 0) {
    return (
      <p className="text-sm text-fg-tertiary">
        还没有训练文件夹，点击「+ 新建」创建
      </p>
    )
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
              title={isActive ? '当前查看（也是复制目标）' : '点击切换查看 + 复制目标'}
              className={`px-2 py-0.5 font-mono ${isActive ? 'text-accent' : 'text-fg-secondary'}`}
            >
              {f}
              <span className="text-fg-tertiary"> ({counts[f] ?? 0})</span>
            </button>
            <button
              onClick={() => onRename(f)}
              disabled={busy}
              title="改名"
              className="opacity-0 group-hover:opacity-100 px-1 py-0.5 text-xs text-fg-tertiary"
            >
              ✎
            </button>
            <button
              onClick={() => onDelete(f)}
              disabled={busy}
              title="删除文件夹"
              className="opacity-0 group-hover:opacity-100 px-1 py-0.5 text-xs text-fg-tertiary"
            >
              ×
            </button>
          </span>
        )
      })}
      <span className="text-fg-tertiary ml-2">总 {total} 张</span>
    </div>
  )
}

/** 按住 Alt 时浮在所有内容上方的大图预览。
 *
 * 用 `pointer-events-none` 让鼠标事件透传到底层 ImageGrid，所以用户可以一边
 * 按 alt 一边在缩略图上滑动，预览随焦点切换；松开 alt（或 window blur）即消失。
 */
function AltHoverPreview({ focus }: { focus: Focus }) {
  const sourceLabel =
    focus.side === 'left' ? 'download' : `train / ${focus.folder}`
  return (
    <div
      aria-hidden
      className="fixed inset-0 z-40 pointer-events-none flex items-center justify-center p-6"
    >
      <div
        className="relative flex flex-col overflow-hidden rounded-lg border border-bold max-w-[95vw] max-h-[95vh] bg-black/90 shadow-xl"
      >
        <img
          src={focus.url}
          alt={focus.name}
          className="max-w-[95vw] max-h-[88vh] object-contain"
        />
        <div
          className="flex items-center gap-2 shrink-0 px-3 py-1.5 border-t border-white/[0.08]"
        >
          <span className={`shrink-0 ${focus.side === 'left' ? 'badge badge-ok' : 'badge badge-info'}`}>
            {sourceLabel}
          </span>
          <code className="mono truncate flex-1 min-w-0 text-fg-inverse text-sm">
            {focus.name}
          </code>
          <span className="text-xs shrink-0 text-white/40">
            松开 Alt 关闭
          </span>
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
  accent,
  title,
  subtitle,
  actions,
  children,
}: {
  accent: 'emerald' | 'cyan'
  title: string
  subtitle: string
  actions: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section className="flex flex-col min-h-0 rounded-md border border-subtle bg-surface overflow-hidden"
    >
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

function BtnPrimary({
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button {...rest} className="btn btn-primary btn-sm">
      {children}
    </button>
  )
}

function BtnSecondary({
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button {...rest} className="btn btn-secondary btn-sm">
      {children}
    </button>
  )
}

function BtnDanger({
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className="btn btn-sm bg-err-soft text-err border-err"
    >
      {children}
    </button>
  )
}
