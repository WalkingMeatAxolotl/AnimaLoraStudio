import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Outlet, useNavigate, useParams } from 'react-router-dom'
import { api, downloadBlob, type ProjectDetail } from '../../api/client'
import { useProjectCtxSetter } from '../../context/ProjectContext'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'

export default function ProjectLayout() {
  const { pid } = useParams()
  const projectId = pid ? Number(pid) : NaN
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const setCtx = useProjectCtxSetter()
  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [creatingBusy, setCreatingBusy] = useState(false)
  const [exporting, setExporting] = useState(false)
  const projectRef = useRef<ProjectDetail | null>(null)
  projectRef.current = project

  const reload = useCallback(async () => {
    if (!Number.isFinite(projectId)) return
    try {
      const p = await api.getProject(projectId)
      setProject(p)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [projectId])

  useEffect(() => {
    void reload()
  }, [reload])

  useEventStream((evt) => {
    if (
      (evt.type === 'project_state_changed' && evt.project_id === projectId) ||
      (evt.type === 'version_state_changed' && evt.project_id === projectId)
    ) {
      void reload()
    }
  })

  const activeVersion = useMemo(() => {
    if (!project) return null
    const aid = project.active_version_id
    return project.versions.find((v) => v.id === aid) ?? project.versions[0] ?? null
  }, [project])

  const handleSelectVersion = useCallback(async (vid: number) => {
    if (!projectRef.current) return
    if (projectRef.current.active_version_id === vid) return
    try {
      const updated = await api.activateVersion(projectRef.current.id, vid)
      setProject(updated)
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [toast])

  const handleExportTrain = useCallback(async () => {
    if (!projectRef.current || exporting) return
    const av = projectRef.current.versions.find(
      (v) => v.id === projectRef.current!.active_version_id
    ) ?? projectRef.current.versions[0] ?? null
    if (!av) return
    setExporting(true)
    try {
      const filename = `${projectRef.current.slug}-${av.label}.train.zip`
      await downloadBlob(api.versionTrainZipUrl(projectRef.current.id, av.id), filename)
    } catch (e) {
      toast(`导出失败: ${e}`, 'error')
    } finally {
      setExporting(false)
    }
  }, [exporting, toast])

  const handleDeleteVersion = useCallback(async (vid: number) => {
    if (!projectRef.current) return
    const v = projectRef.current.versions.find((x) => x.id === vid)
    if (!v) return
    if (!(await confirm(`删除版本 ${v.label}？目录将移到回收站。`, { tone: 'warn', okText: '删版本' }))) return
    const pid = projectRef.current.id
    try {
      await api.deleteVersion(pid, vid)
      await reload()
      toast(`已删除版本 ${v.label}`, 'success')
      navigate(`/projects/${pid}`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [reload, toast, navigate, confirm])

  const handleCreateVersion = useCallback(async (label: string, forkFromVersionId: number | null) => {
    if (!projectRef.current || creatingBusy) return
    setCreatingBusy(true)
    try {
      const body: { label: string; fork_from_version_id?: number } = { label }
      if (forkFromVersionId !== null) body.fork_from_version_id = forkFromVersionId
      const v = await api.createVersion(projectRef.current.id, body)
      await api.activateVersion(projectRef.current.id, v.id)
      await reload()
      setCreating(false)
      toast(
        forkFromVersionId !== null
          ? `已从副本创建版本 ${label}`
          : `已创建版本 ${label}`,
        'success',
      )
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setCreatingBusy(false)
    }
  }, [creatingBusy, reload, toast])

  // Push context up to App-level so Sidebar (sibling of <main>) can read it
  useEffect(() => {
    if (!project || !setCtx) return
    setCtx({
      project,
      activeVersion,
      reload,
      onSelectVersion: handleSelectVersion,
      onCreateVersion: () => setCreating(true),
      onExportTrain: handleExportTrain,
      onDeleteVersion: handleDeleteVersion,
      exporting,
    })
  }, [project, activeVersion, reload, handleSelectVersion, handleExportTrain, handleDeleteVersion, exporting, setCtx])

  // Clear context when leaving project route
  useEffect(() => {
    return () => { setCtx?.(null) }
  }, [setCtx])

  if (error) {
    return (
      <div className="m-4 p-3 rounded-md border border-err bg-err-soft text-err font-mono text-sm">
        {error}
      </div>
    )
  }
  if (!project) {
    return <p className="p-6 text-fg-tertiary">加载项目...</p>
  }

  return (
    <div className="flex flex-col h-full">
      {/* outlet context 透传 onCreateVersion + busy,Overview 的「新版本」按钮
          复用同一个 NewVersionDialog,避免 Overview 单独维护 window.prompt 版本。 */}
      <Outlet context={{
        project,
        activeVersion,
        reload,
        onCreateVersion: () => setCreating(true),
        creatingVersionBusy: creatingBusy,
      }} />
      {creating && (
        <NewVersionDialog
          existingLabels={project.versions.map((v) => v.label)}
          existingVersions={project.versions.map((v) => ({ id: v.id, label: v.label }))}
          busy={creatingBusy}
          onCancel={() => { if (creatingBusy) return; setCreating(false) }}
          onSubmit={handleCreateVersion}
        />
      )}
    </div>
  )
}

export function NewVersionDialog({
  existingLabels,
  existingVersions,
  busy = false,
  onCancel,
  onSubmit,
}: {
  existingLabels: string[]
  existingVersions: { id: number; label: string }[]
  busy?: boolean
  onCancel: () => void
  onSubmit: (label: string, forkFromVersionId: number | null) => void
}) {
  const [label, setLabel] = useState('')
  const [forkFrom, setForkFrom] = useState<string>('')
  const [err, setErr] = useState<string | null>(null)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (busy) return
    const l = label.trim()
    if (!l) return setErr('label 不能为空')
    if (!/^[A-Za-z0-9_.-]+$/.test(l))
      return setErr('label 只允许字母 / 数字 / 下划线 / 连字符 / 点')
    if (existingLabels.includes(l)) return setErr('label 已存在')
    const fid = forkFrom === '' ? null : Number(forkFrom)
    onSubmit(l, fid)
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
      onClick={onCancel}
    >
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
        className="bg-elevated border border-dim rounded-lg w-[90%] max-w-[440px] p-6 flex flex-col gap-4 shadow-xl"
      >
        <h2 className="m-0 text-lg font-semibold">新建版本</h2>
        <label className="flex flex-col gap-1">
          <span className="text-xs text-fg-tertiary font-mono">label</span>
          <input
            autoFocus
            value={label}
            onChange={(e) => { setLabel(e.target.value); setErr(null) }}
            className="input input-mono"
            placeholder="例：baseline / high-lr"
          />
        </label>
        {existingVersions.length > 0 && (
          <label className="flex flex-col gap-1">
            <span className="text-xs text-fg-tertiary font-mono">从…创建</span>
            <select
              value={forkFrom}
              onChange={(e) => setForkFrom(e.target.value)}
              className="input"
            >
              <option value="">从空白开始</option>
              {existingVersions.map((v) => (
                <option key={v.id} value={String(v.id)}>
                  从 {v.label} 复制
                </option>
              ))}
            </select>
            {forkFrom !== '' && (
              <p className="m-0 text-xs text-fg-tertiary">
                将复制 train/、reg/、训练配置、解锁状态（output/、samples/ 不复制）
              </p>
            )}
          </label>
        )}
        {err && <p className="m-0 text-sm text-err">{err}</p>}
        <div className="flex gap-2 justify-end">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="btn btn-secondary"
          >
            取消
          </button>
          <button
            type="submit"
            disabled={busy}
            className="btn btn-primary"
          >
            {busy ? '创建中...' : '创建'}
          </button>
        </div>
      </form>
    </div>
  )
}

export interface ProjectLayoutContext {
  project: ProjectDetail
  activeVersion: ReturnType<typeof Object.assign>
  reload: () => Promise<void>
}
