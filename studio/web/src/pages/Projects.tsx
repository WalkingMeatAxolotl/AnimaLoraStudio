import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { api, type BundleImportResult, type ProjectSummary } from '../api/client'
import PageHeader from '../components/PageHeader'
import PathPicker from '../components/PathPicker'
import VersionStatusBadge from '../components/VersionStatusBadge'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'

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
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()

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
    setShowImportDialog(false)
    await runBundleImport(() => api.importBundleUpload(file))
  }

  const openProject = (p: ProjectSummary) => {
    navigate(`/projects/${p.id}`)
  }

  return (
    <div className="fade-in">
      <PageHeader
        title={t('projects.title')}
        subtitle={t('projects.description')}
        actions={
          <>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setShowImportDialog(true)}
              disabled={importing}
              title={importing ? t('projects.importing') : t('projects.importZipHint')}
            >
              {importing ? t('projects.importing') : t('projects.importZip')}
            </button>
            <button className="btn btn-primary" onClick={() => setCreating(true)}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span>{t('projects.newProject')}</span>
            </button>
          </>
        }
      />

      <div className="p-6">
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
        ) : (
          <div className="grid gap-4 auto-rows-fr" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))' }}>
            {items.map((p) => (
              <ProjectCard
                key={p.id}
                project={p}
                onClick={() => openProject(p)}
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
          onUpload={handleImportUpload}
          onPickPath={() => {
            setShowImportDialog(false)
            setShowImportPicker(true)
          }}
          onCancel={() => setShowImportDialog(false)}
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

function BundleImportDialog({
  importing,
  onUpload,
  onPickPath,
  onCancel,
}: {
  importing: boolean
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
  onClick,
  onDelete,
}: {
  project: ProjectSummary
  onClick: () => void
  onDelete: (e: React.MouseEvent) => void
}) {
  const { t } = useTranslation()
  const [hovered, setHovered] = useState(false)

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`p-[18px] text-left rounded-lg cursor-pointer flex flex-col gap-3.5 relative w-full ${hovered ? 'border-dim shadow-sm bg-surface' : 'border border-subtle bg-surface'}`}
      style={{ transition: 'border-color 0.15s, box-shadow 0.15s' }}
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
        <VersionStatusBadge status={p.active_version_status} />
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
        <button
          onClick={onDelete}
          className="bg-transparent border-none px-1.5 py-0.5 rounded-sm text-fg-tertiary text-xs cursor-pointer"
          title={t('projects.deleteProjectTitle')}
        >
          ×
        </button>
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
