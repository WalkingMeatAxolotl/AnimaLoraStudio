import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { api, type BundleImportResult, type ProjectSummary, type VersionStatus } from '../api/client'
import PageHeader from '../components/PageHeader'
import PathPicker from '../components/PathPicker'
import UploadProgressBar from '../components/UploadProgressBar'
import VersionStatusBadge from '../components/VersionStatusBadge'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'
import { useLocalStorageState } from '../lib/useLocalStorageState'
import { useUploadProgress } from '../lib/useUploadProgress'

export type ProjectSortKey = 'updated' | 'created' | 'title'
export type ProjectStatusFilter = VersionStatus | 'all'

const STATUS_OPTIONS: ProjectStatusFilter[] = [
  'all', 'preparing', 'training', 'completed', 'failed', 'canceled',
]
const SORT_OPTIONS: ProjectSortKey[] = ['updated', 'created', 'title']

/** 过滤 + 排序（纯函数，单测覆盖）。query 匹配 title / slug / note。 */
export function filterProjects(
  items: ProjectSummary[],
  opts: { query: string; status: ProjectStatusFilter; sort: ProjectSortKey },
): ProjectSummary[] {
  const q = opts.query.trim().toLowerCase()
  const matched = items.filter((p) => {
    if (q && !`${p.title}\n${p.slug}\n${p.note ?? ''}`.toLowerCase().includes(q)) return false
    if (opts.status !== 'all' && p.active_version_status !== opts.status) return false
    return true
  })
  const cmp: Record<ProjectSortKey, (a: ProjectSummary, b: ProjectSummary) => number> = {
    updated: (a, b) => b.updated_at - a.updated_at,
    created: (a, b) => b.created_at - a.created_at,
    title: (a, b) => a.title.localeCompare(b.title),
  }
  return [...matched].sort(cmp[opts.sort])
}

export default function ProjectsPage() {
  const { t } = useTranslation()
  const [items, setItems] = useState<ProjectSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [busy, setBusy] = useState(false)
  const [importing, setImporting] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)
  const [showImportPicker, setShowImportPicker] = useState(false)
  // 过滤面板：默认折叠；排序偏好持久化，搜索 / 状态 / 归档可见性每次进页重置
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<ProjectStatusFilter>('all')
  const [sortKey, setSortKey] = useLocalStorageState<ProjectSortKey>('studio:projects:sort', 'updated')
  const [showArchived, setShowArchived] = useState(false)
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const uploadProgress = useUploadProgress()

  const refresh = async () => {
    try {
      const list = await api.listProjects()
      setItems(list)
      setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])

  useEventStream((evt) => {
    if (evt.type === 'project_state_changed') void refresh()
  })

  const handleCreate = async (form: NewProjectForm) => {
    setBusy(true)
    try {
      const p = await api.createProject({
        title: form.title,
        note: form.note || undefined,
        initial_version_label: form.initial_version_label || 'v1',
      })
      toast(t('projects.created', { title: p.title }), 'success')
      setCreating(false)
      navigate(`/projects/${p.id}`)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const handleDelete = async (p: ProjectSummary, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!(await confirm(
      t('projects.deleteConfirm', { title: p.title, slug: p.slug }),
      { tone: 'danger', okText: t('projects.deleteProject') },
    ))) return
    try {
      await api.deleteProject(p.id)
      toast(t('projects.deleted', { title: p.title }), 'success')
      await refresh()
    } catch (err) {
      toast(String(err), 'error')
    }
  }

  const handleArchive = async (p: ProjectSummary, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!(await confirm(
      t('projects.archiveConfirm', { title: p.title }),
      { okText: t('projects.archiveBtn') },
    ))) return
    try {
      await api.archiveProject(p.id)
      toast(t('projects.archived', { title: p.title }), 'success')
      await refresh()
    } catch (err) {
      toast(String(err), 'error')
    }
  }

  const handleUnarchive = async (p: ProjectSummary, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!(await confirm(
      t('projects.unarchiveConfirm', { title: p.title }),
      { okText: t('common.restore') },
    ))) return
    try {
      await api.unarchiveProject(p.id)
      toast(t('projects.unarchived', { title: p.title }), 'success')
      await refresh()
    } catch (err) {
      toast(String(err), 'error')
    }
  }

  const finishBundleImport = (result: BundleImportResult) => {
    const stats = result.stats
    toast(
      t('projects.importedBundle', {
        title: result.project.title,
        train_count: stats.train_image_count,
        reg_count: stats.reg_image_count,
        preset_count: stats.preset_count,
      }),
      'success',
    )
    navigate(`/projects/${result.project.id}`)
  }

  const runBundleImport = async (job: () => Promise<BundleImportResult>) => {
    setImporting(true)
    try {
      finishBundleImport(await job())
    } catch (e) {
      toast(t('projects.importFailed', { e }), 'error')
    } finally {
      setImporting(false)
    }
  }

  const handleImportPath = async (path: string) => {
    setShowImportPicker(false)
    await runBundleImport(() => api.importBundleFromPath(path))
  }

  const handleImportUpload = async (file: File | null | undefined) => {
    if (!file) return
    // dialog 不立即关：进度条要显示在 dialog 里直到完成 / 失败
    uploadProgress.start(file.size)
    setImporting(true)
    try {
      const result = await api.importBundleUpload(file, uploadProgress.onProgress)
      uploadProgress.finish()
      setShowImportDialog(false)
      uploadProgress.reset()
      finishBundleImport(result)
    } catch (e) {
      uploadProgress.fail(e)
      toast(t('projects.importFailed', { e }), 'error')
    } finally {
      setImporting(false)
    }
  }

  const openProject = (p: ProjectSummary) => {
    navigate(`/projects/${p.id}`)
  }

  const archivedItems = items.filter((p) => p.archived_at != null)
  const filterOpts = { query, status: statusFilter, sort: sortKey }
  // 归档开关是视图切换（radio 语义）：开 = 只看已归档，关 = 只看活跃
  const visible = filterProjects(
    showArchived ? archivedItems : items.filter((p) => p.archived_at == null),
    filterOpts,
  )
  const filtering = query.trim() !== '' || statusFilter !== 'all'

  return (
    <div className="fade-in">
      <PageHeader
        title={t('projects.title')}
        subtitle={t('projects.description')}
        actions={
          <>
            {/* btn 词汇与 Queue / Generate 页 header 统一：轻操作 ghost、
             * 激活态 secondary、全部 btn-sm；唯一 primary 留给页面 CTA。 */}
            {/* 过滤 icon：折叠态完全不占行，开关过滤行；有筛选生效且收起时带小圆点 */}
            <button
              className={`btn btn-sm ${filtersOpen ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setFiltersOpen((o) => !o)}
              aria-expanded={filtersOpen}
              aria-label={t('projects.filters')}
              title={t('projects.filters')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 4h18l-7 8v6l-4 2v-8L3 4z" />
              </svg>
              {!filtersOpen && filtering && (
                <span className="dot dot-running" aria-label={t('projects.filtersActive')} />
              )}
            </button>
            {/* 已归档视图开关（radio 语义：开 = 列表只显示已归档项目） */}
            <button
              className={`btn btn-sm ${showArchived ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setShowArchived((v) => !v)}
              aria-pressed={showArchived}
              title={t('projects.archivedToggleHint')}
            >
              {t('projects.archivedToggle', { n: archivedItems.length })}
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setShowImportDialog(true)}
              disabled={importing}
              title={importing ? t('projects.importing') : t('projects.importZipHint')}
            >
              {importing ? t('projects.importing') : t('projects.importZip')}
            </button>
            <button className="btn btn-primary btn-sm" onClick={() => setCreating(true)}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span>{t('projects.newProject')}</span>
            </button>
          </>
        }
      />

      {filtersOpen && (
        <FilterBar
          query={query}
          onQuery={setQuery}
          status={statusFilter}
          onStatus={setStatusFilter}
          sort={sortKey}
          onSort={setSortKey}
        />
      )}

      <div className="p-6 pt-4">
        {error && (
          <div className="mb-4 px-3.5 py-2.5 rounded-md bg-err-soft border border-err text-err text-sm font-mono">{error}</div>
        )}

        {loading ? (
          <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))' }}>
            {[1, 2, 3].map(i => (
              <div key={i} className="card p-[18px]" style={{ height: 140 }}>
                <div className="w-3/5 h-4 rounded bg-overlay mb-2.5" />
                <div className="w-2/5 h-[11px] rounded-sm bg-overlay" />
              </div>
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="mt-20 text-center text-fg-tertiary">
            <div className="text-lg mb-2">{t('projects.noProjects')}</div>
            <div className="text-sm">{t('projects.noProjectsHint')}</div>
          </div>
        ) : visible.length === 0 ? (
          <div className="mt-20 text-center text-fg-tertiary text-sm">
            {t('common.noResults')}
          </div>
        ) : (
          <div className="grid gap-4 auto-rows-fr" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))' }}>
            {visible.map((p) => (
              <ProjectCard
                key={p.id}
                project={p}
                archived={showArchived}
                onClick={() => openProject(p)}
                onArchive={(e) => handleArchive(p, e)}
                onUnarchive={(e) => handleUnarchive(p, e)}
                onDelete={(e) => handleDelete(p, e)}
              />
            ))}
          </div>
        )}
      </div>

      {creating && (
        <NewProjectDialog
          busy={busy}
          onCancel={() => setCreating(false)}
          onSubmit={handleCreate}
        />
      )}

      {showImportDialog && (
        <BundleImportDialog
          importing={importing}
          uploadState={uploadProgress.state}
          onUpload={handleImportUpload}
          onPickPath={() => {
            setShowImportDialog(false)
            setShowImportPicker(true)
          }}
          onCancel={() => {
            setShowImportDialog(false)
            uploadProgress.reset()
          }}
        />
      )}

      {showImportPicker && (
        <PathPicker
          dirOnly={false}
          onClose={() => setShowImportPicker(false)}
          onPick={(path) => { void handleImportPath(path) }}
        />
      )}
    </div>
  )
}

/** Header 下方的过滤 / 排序行（仅 filtersOpen 时渲染，折叠完全不占空间；
 * 开关在 PageHeader 的过滤 icon 上）。单行：搜索 60% 居左，状态 / 排序
 * 各 10% 推到最右。 */
function FilterBar({
  query,
  onQuery,
  status,
  onStatus,
  sort,
  onSort,
}: {
  query: string
  onQuery: (v: string) => void
  status: ProjectStatusFilter
  onStatus: (v: ProjectStatusFilter) => void
  sort: ProjectSortKey
  onSort: (v: ProjectSortKey) => void
}) {
  const { t } = useTranslation()
  return (
    <div className="px-6 py-2 border-b border-subtle flex items-center gap-3">
      <input
        className="input"
        style={{ width: '60%' }}
        value={query}
        onChange={(e) => onQuery(e.target.value)}
        placeholder={t('projects.searchPlaceholder')}
        aria-label={t('common.search')}
      />
      <span className="flex-1" />
      <select
        className="input"
        style={{ width: '10%', minWidth: 104 }}
        value={status}
        onChange={(e) => onStatus(e.target.value as ProjectStatusFilter)}
        aria-label={t('common.status')}
      >
        {STATUS_OPTIONS.map((s) => (
          <option key={s} value={s}>
            {s === 'all' ? t('projects.statusAll') : t(`versionStatus.${s}`)}
          </option>
        ))}
      </select>
      <select
        className="input"
        style={{ width: '10%', minWidth: 104 }}
        value={sort}
        onChange={(e) => onSort(e.target.value as ProjectSortKey)}
        aria-label={t('projects.sortLabel')}
      >
        {SORT_OPTIONS.map((s) => (
          <option key={s} value={s}>{t(`projects.sort_${s}`)}</option>
        ))}
      </select>
    </div>
  )
}

function BundleImportDialog({
  importing,
  uploadState,
  onUpload,
  onPickPath,
  onCancel,
}: {
  importing: boolean
  uploadState: ReturnType<typeof useUploadProgress>['state']
  onUpload: (file: File | null | undefined) => void
  onPickPath: () => void
  onCancel: () => void
}) {
  const { t } = useTranslation()

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <div className="bg-elevated border border-dim rounded-lg w-[90%] max-w-[560px] p-6 flex flex-col gap-4 shadow-xl">
        <div>
          <h2 className="m-0 text-lg font-semibold text-fg-primary">
            {t('projects.importBundleTitle')}
          </h2>
          <p className="mt-1 mb-0 text-sm text-fg-secondary">
            {t('projects.importBundleHint')}
          </p>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <label className={`card p-4 cursor-pointer ${importing ? 'opacity-60 pointer-events-none' : ''}`}>
            <div className="font-medium text-fg-primary mb-1">{t('projects.importUpload')}</div>
            <div className="text-xs text-fg-tertiary mb-3">{t('projects.importUploadHint')}</div>
            <input
              type="file"
              accept=".zip,application/zip"
              className="text-xs text-fg-secondary w-full"
              disabled={importing}
              onChange={(e) => onUpload(e.target.files?.[0])}
            />
            {uploadState.phase !== 'idle' && (
              <UploadProgressBar state={uploadState} className="mt-3" />
            )}
          </label>

          <button
            type="button"
            className="card p-4 text-left cursor-pointer hover:border-dim disabled:opacity-60"
            disabled={importing}
            onClick={onPickPath}
          >
            <div className="font-medium text-fg-primary mb-1">{t('projects.importPath')}</div>
            <div className="text-xs text-fg-tertiary">{t('projects.importPathHint')}</div>
          </button>
        </div>

        <div className="flex justify-end">
          <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={importing}>
            {t('common.cancel')}
          </button>
        </div>
      </div>
    </div>
  )
}

function ProjectCard({
  project: p,
  archived = false,
  onClick,
  onArchive,
  onUnarchive,
  onDelete,
}: {
  project: ProjectSummary
  /** 已归档卡片：半透明 + ↺ 恢复（在 × 左边）+ × 真删；普通卡片 × = 归档。 */
  archived?: boolean
  onClick: () => void
  onArchive?: (e: React.MouseEvent) => void
  onUnarchive?: (e: React.MouseEvent) => void
  onDelete?: (e: React.MouseEvent) => void
}) {
  const { t } = useTranslation()
  const [hovered, setHovered] = useState(false)

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`p-[18px] text-left rounded-lg cursor-pointer flex flex-col gap-3.5 relative w-full ${hovered ? 'border-dim shadow-sm bg-surface' : 'border border-subtle bg-surface'} ${archived ? 'opacity-70' : ''}`}
      style={{ transition: 'border-color 0.15s, box-shadow 0.15s, opacity 0.15s' }}
    >
      {/* ADR-0007 §11.8-E: 右上角 = active version status；去 stage badge / 时间 / 产物 */}
      <div className="flex justify-between items-start gap-2">
        <div className="flex-1 min-w-0">
          <div className="text-md font-semibold overflow-hidden text-ellipsis whitespace-nowrap" style={{ letterSpacing: '-0.01em' }}>
            {p.title}
          </div>
          <div className="mono text-xs text-fg-tertiary mt-0.5">
            {p.slug}
          </div>
        </div>
        <VersionStatusBadge status={p.active_version_status} phase={p.active_version_phase} />
      </div>

      {p.note && (
        <p className="m-0 text-sm text-fg-secondary overflow-hidden line-clamp-2">
          {p.note}
        </p>
      )}

      <div className="flex gap-4 text-sm text-fg-secondary mt-auto items-center">
        {/* active version 名（直接版本，无前缀文本） */}
        {p.active_version_label ? (
          <span className="font-mono text-fg-primary">{p.active_version_label}</span>
        ) : (
          <span className="text-fg-tertiary italic text-xs">{t('projects.noActiveVersion')}</span>
        )}
        <span className="flex-1" />
        {archived ? (
          <>
            <button
              onClick={onUnarchive}
              className="bg-transparent border-none px-1.5 py-0.5 rounded-sm text-fg-tertiary text-xs cursor-pointer"
              title={t('projects.unarchiveTitle')}
              aria-label={`${t('projects.unarchiveTitle')} ${p.title}`}
            >
              ↺
            </button>
            <button
              onClick={onDelete}
              className="bg-transparent border-none px-1.5 py-0.5 rounded-sm text-fg-tertiary text-xs cursor-pointer"
              title={t('projects.deleteProjectTitle')}
              aria-label={`${t('projects.deleteProjectTitle')} ${p.title}`}
            >
              ×
            </button>
          </>
        ) : (
          <button
            onClick={onArchive}
            className="bg-transparent border-none px-1.5 py-0.5 rounded-sm text-fg-tertiary text-xs cursor-pointer"
            title={t('projects.archiveProjectTitle')}
            aria-label={`${t('projects.archiveProjectTitle')} ${p.title}`}
          >
            ×
          </button>
        )}
      </div>
    </button>
  )
}

// ── New Project Dialog ──────────────────────────────────────────

interface NewProjectForm {
  title: string
  note: string
  initial_version_label: string
}

function NewProjectDialog({
  busy,
  onCancel,
  onSubmit,
}: {
  busy: boolean
  onCancel: () => void
  onSubmit: (form: NewProjectForm) => void
}) {
  const { t } = useTranslation()
  const [form, setForm] = useState<NewProjectForm>({
    title: '',
    note: '',
    initial_version_label: 'v1',
  })

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.title.trim()) return
    onSubmit(form)
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/45"
      onClick={onCancel}
    >
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
        className="bg-surface border border-dim rounded-xl p-7 flex flex-col gap-[18px] shadow-lg"
        style={{ width: '90%', maxWidth: 440 }}
      >
        <h2 className="m-0 text-xl font-semibold">{t('projects.newProject')}</h2>

        <FieldLabel label={t('projects.newProjectTitle')} hint="title">
          <input
            autoFocus
            className="input"
            value={form.title}
            onChange={(e) => setForm({ ...form, title: e.target.value })}
            placeholder={t('projects.titlePlaceholder')}
          />
        </FieldLabel>

        <FieldLabel label={t('projects.versionLabel')} hint="initial_version_label">
          <input
            className="input input-mono"
            value={form.initial_version_label}
            onChange={(e) => setForm({ ...form, initial_version_label: e.target.value })}
            placeholder={t('projects.versionPlaceholder')}
          />
        </FieldLabel>

        <FieldLabel label={t('common.notes')} hint="note（可选）">
          <textarea
            className="input"
            value={form.note}
            onChange={(e) => setForm({ ...form, note: e.target.value })}
            placeholder={t('projects.notesPlaceholder')}
            rows={3}
            style={{ resize: 'vertical' }}
          />
        </FieldLabel>

        <div className="flex gap-2 justify-end">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>{t('common.cancel')}</button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={busy || !form.title.trim()}
          >
            {busy ? t('projects.creating') : t('common.create')}
          </button>
        </div>
      </form>
    </div>
  )
}

function FieldLabel({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium">
        {label}
        {hint && <span className="ml-2 text-xs text-fg-tertiary font-mono">{hint}</span>}
      </span>
      {children}
    </label>
  )
}
