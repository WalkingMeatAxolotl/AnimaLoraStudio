import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type ProjectStage, type ProjectSummary } from '../api/client'
import PageHeader from '../components/PageHeader'
import StageBadge from '../components/StageBadge'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'

function relativeTime(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return '刚刚'
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`
  if (diff < 604800) return `${Math.floor(diff / 86400)} 天前`
  return new Date(ts * 1000).toLocaleDateString('zh-CN')
}

// stage → step path for quick-open nav
const STAGE_STEP: Partial<Record<ProjectStage, string>> = {
  downloading:  'download',
  curating:     'curate',
  tagging:      'tag',
  regularizing: 'reg',
  configured:   'train',
  training:     'train',
}

export default function ProjectsPage() {
  const [items, setItems] = useState<ProjectSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [busy, setBusy] = useState(false)
  const [importing, setImporting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const navigate = useNavigate()
  const { toast } = useToast()

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
      toast(`已创建 ${p.title}`, 'success')
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
    if (!confirm(`移到回收站？\n${p.title} (${p.slug})`)) return
    try {
      await api.deleteProject(p.id)
      toast(`已移到回收站: ${p.title}`, 'success')
      await refresh()
    } catch (err) {
      toast(String(err), 'error')
    }
  }

  const handleImportFile = async (file: File) => {
    setImporting(true)
    try {
      const result = await api.importTrainProject(file)
      const stats = result.stats
      toast(
        `已导入 ${result.project.title}（${stats.image_count} 张图，${stats.tagged_count} 已打标）`,
        'success',
      )
      navigate(`/projects/${result.project.id}`)
    } catch (e) {
      toast(`导入失败: ${e}`, 'error')
    } finally {
      setImporting(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleEmptyTrash = async () => {
    if (!confirm('物理删除所有回收站项目？此操作不可恢复。')) return
    try {
      const r = await api.emptyTrash()
      toast(`清空了 ${r.removed} 个项目`, 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const openProject = (p: ProjectSummary) => {
    navigate(`/projects/${p.id}`)
  }

  return (
    <div className="fade-in">
      <PageHeader
        title="项目"
        subtitle="每个项目对应一个 LoRA 训练目标 — 角色、风格或概念。新建一个项目开始流水线。"
        actions={
          <>
            <button
              className="btn btn-ghost btn-sm"
              onClick={handleEmptyTrash}
              title="物理删除回收站"
            >
              清空回收站
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip,application/zip"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void handleImportFile(f)
              }}
            />
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={importing}
              title={importing ? '上传 + 解压中...' : '上传训练集 zip → 自动新建项目'}
            >
              {importing ? '导入中…' : '导入 zip'}
            </button>
            <button className="btn btn-primary" onClick={() => setCreating(true)}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span>新建项目</span>
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
            <div className="text-lg mb-2">还没有项目</div>
            <div className="text-sm">点右上角「新建项目」开始一个新的 LoRA 训练</div>
          </div>
        ) : (
          <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))' }}>
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
  const [hovered, setHovered] = useState(false)

  const stepPath = p.stage in STAGE_STEP ? STAGE_STEP[p.stage] : undefined

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`p-[18px] text-left rounded-lg cursor-pointer flex flex-col gap-3.5 relative w-full ${hovered ? 'border-dim shadow-sm bg-surface' : 'border border-subtle bg-surface'}`}
      style={{ transition: 'border-color 0.15s, box-shadow 0.15s' }}
    >
      {/* header row */}
      <div className="flex justify-between items-start gap-2">
        <div className="flex-1 min-w-0">
          <div className="text-md font-semibold overflow-hidden text-ellipsis whitespace-nowrap" style={{ letterSpacing: '-0.01em' }}>
            {p.title}
          </div>
          <div className="mono text-xs text-fg-tertiary mt-0.5">
            {p.slug}
          </div>
        </div>
        <StageBadge stage={p.stage} />
      </div>

      {/* note */}
      {p.note && (
        <p className="m-0 text-sm text-fg-secondary overflow-hidden line-clamp-2">
          {p.note}
        </p>
      )}

      {/* footer row */}
      <div className="flex gap-4 text-sm text-fg-secondary mt-auto items-center">
        <StatPair label="下载" value={p.download_image_count ?? 0} />
        <span className="flex-1" />
        {stepPath && (
          <span className="text-xs text-accent font-mono">
            继续 →
          </span>
        )}
        <span className="text-fg-tertiary text-xs">
          {relativeTime(p.updated_at)}
        </span>
        <button
          onClick={onDelete}
          className="bg-transparent border-none px-1.5 py-0.5 rounded-sm text-fg-tertiary text-xs cursor-pointer"
          title="移到回收站"
        >
          ×
        </button>
      </div>
    </button>
  )
}

function StatPair({ label, value }: { label: string; value: number }) {
  return (
    <span className="inline-flex gap-1.5 items-baseline">
      <span className="font-mono font-semibold text-fg-primary">{value}</span>
      <span className="text-xs text-fg-tertiary uppercase tracking-wider">{label}</span>
    </span>
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
        <h2 className="m-0 text-xl font-semibold">新建项目</h2>

        <FieldLabel label="项目名称" hint="title">
          <input
            autoFocus
            className="input"
            value={form.title}
            onChange={(e) => setForm({ ...form, title: e.target.value })}
            placeholder="例：Cosmic Kaguya"
          />
        </FieldLabel>

        <FieldLabel label="初始版本标签" hint="initial_version_label">
          <input
            className="input input-mono"
            value={form.initial_version_label}
            onChange={(e) => setForm({ ...form, initial_version_label: e.target.value })}
            placeholder="v1 / baseline / ..."
          />
        </FieldLabel>

        <FieldLabel label="备注" hint="note（可选）">
          <textarea
            className="input"
            value={form.note}
            onChange={(e) => setForm({ ...form, note: e.target.value })}
            placeholder="训练目标 / 备注"
            rows={3}
            style={{ resize: 'vertical' }}
          />
        </FieldLabel>

        <div className="flex gap-2 justify-end">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>取消</button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={busy || !form.title.trim()}
          >
            {busy ? '创建中…' : '创建'}
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
