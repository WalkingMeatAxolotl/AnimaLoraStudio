import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, type ProjectStage, type ProjectSummary } from '../api/client'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'

const STAGE_LABEL: Record<ProjectStage, string> = {
  created: '已创建',
  downloading: '下载中',
  curating: '筛选中',
  tagging: '打标中',
  regularizing: '正则集中',
  configured: '已配置',
  training: '训练中',
  done: '完成',
}

const STAGE_COLOR: Record<ProjectStage, string> = {
  created: 'bg-slate-700/60 text-slate-300',
  downloading: 'bg-amber-700/40 text-amber-200',
  curating: 'bg-amber-700/40 text-amber-200',
  tagging: 'bg-amber-700/40 text-amber-200',
  regularizing: 'bg-amber-700/40 text-amber-200',
  configured: 'bg-cyan-700/40 text-cyan-200',
  training: 'bg-violet-700/40 text-violet-200',
  done: 'bg-emerald-700/40 text-emerald-200',
}

export default function ProjectsPage() {
  const [items, setItems] = useState<ProjectSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [busy, setBusy] = useState(false)
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

  useEffect(() => {
    void refresh()
  }, [])

  // SSE: project_state_changed → 重拉列表
  useEventStream((evt) => {
    if (evt.type === 'project_state_changed') {
      void refresh()
    }
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

  const handleDelete = async (p: ProjectSummary) => {
    if (!confirm(`移到回收站？\n${p.title} (${p.slug})`)) return
    try {
      await api.deleteProject(p.id)
      toast(`已移到回收站: ${p.title}`, 'success')
      await refresh()
    } catch (e) {
      toast(String(e), 'error')
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

  return (
    <div className="space-y-4">
      <header className="flex items-center gap-3">
        <h1 className="text-2xl font-semibold flex-1">项目</h1>
        <button
          onClick={handleEmptyTrash}
          className="text-xs text-slate-500 hover:text-slate-300"
        >
          清空回收站
        </button>
        <button
          onClick={() => setCreating(true)}
          className="px-3 py-1.5 rounded text-sm bg-cyan-600 hover:bg-cyan-500"
        >
          + 新建项目
        </button>
      </header>

      {error && (
        <div className="p-3 rounded bg-red-900/40 border border-red-700 text-red-300 text-sm font-mono">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-slate-500">加载中...</p>
      ) : items.length === 0 ? (
        <p className="text-slate-500 mt-12 text-center">
          还没有项目。点右上角「+ 新建项目」开始一个新的 LoRA 训练。
        </p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {items.map((p) => (
            <article
              key={p.id}
              className="rounded-xl border border-slate-700 bg-slate-800/40 p-4 flex flex-col gap-2 hover:border-cyan-700 transition"
            >
              <div className="flex items-center gap-2">
                <Link
                  to={`/projects/${p.id}`}
                  className="text-base font-semibold text-slate-100 hover:text-cyan-300 flex-1 truncate"
                >
                  {p.title}
                </Link>
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${STAGE_COLOR[p.stage]}`}
                >
                  {STAGE_LABEL[p.stage]}
                </span>
              </div>
              <div className="text-xs text-slate-500 font-mono truncate">
                {p.slug}
              </div>
              <div className="text-xs text-slate-400 flex gap-3">
                <span>{p.download_image_count ?? 0} 张下载</span>
                <span className="flex-1" />
                <span>
                  {new Date(p.updated_at * 1000).toLocaleString('zh-CN', {
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </span>
              </div>
              {p.note && (
                <p className="text-xs text-slate-400 line-clamp-2">{p.note}</p>
              )}
              <div className="flex gap-2 mt-1">
                <Link
                  to={`/projects/${p.id}`}
                  className="text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-200"
                >
                  打开
                </Link>
                <span className="flex-1" />
                <button
                  onClick={() => handleDelete(p)}
                  className="text-xs px-2 py-1 rounded text-red-400 hover:text-red-300 hover:bg-red-900/30"
                >
                  删除
                </button>
              </div>
            </article>
          ))}
        </div>
      )}

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
      className="fixed inset-0 z-40 bg-black/60 flex items-center justify-center"
      onClick={onCancel}
    >
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
        className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-[90%] max-w-md space-y-4"
      >
        <h2 className="text-lg font-semibold">新建项目</h2>
        <label className="block">
          <span className="text-xs text-slate-400 font-mono">title</span>
          <input
            autoFocus
            value={form.title}
            onChange={(e) => setForm({ ...form, title: e.target.value })}
            className="mt-1 w-full px-2 py-1.5 rounded bg-slate-950 border border-slate-700 text-sm focus:outline-none focus:border-cyan-500"
            placeholder="例：Cosmic Kaguya"
          />
        </label>
        <label className="block">
          <span className="text-xs text-slate-400 font-mono">
            initial_version_label
          </span>
          <input
            value={form.initial_version_label}
            onChange={(e) =>
              setForm({ ...form, initial_version_label: e.target.value })
            }
            className="mt-1 w-full px-2 py-1.5 rounded bg-slate-950 border border-slate-700 text-sm focus:outline-none focus:border-cyan-500"
            placeholder="v1 / baseline / ..."
          />
        </label>
        <label className="block">
          <span className="text-xs text-slate-400 font-mono">note</span>
          <textarea
            value={form.note}
            onChange={(e) => setForm({ ...form, note: e.target.value })}
            className="mt-1 w-full px-2 py-1.5 rounded bg-slate-950 border border-slate-700 text-sm focus:outline-none focus:border-cyan-500 h-20"
            placeholder="可选：训练目标 / 备注"
          />
        </label>
        <div className="flex gap-2 justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1.5 rounded text-sm bg-slate-700 hover:bg-slate-600"
          >
            取消
          </button>
          <button
            type="submit"
            disabled={busy || !form.title.trim()}
            className="px-3 py-1.5 rounded text-sm bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500"
          >
            {busy ? '创建中...' : '创建'}
          </button>
        </div>
      </form>
    </div>
  )
}
