import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Outlet, useNavigate, useParams } from 'react-router-dom'
import { api, type ProjectDetail } from '../../api/client'
import { useProjectCtxSetter } from '../../context/ProjectContext'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'

export default function ProjectLayout() {
  const { t } = useTranslation()
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
    } else if (
      // train.zip 打包完成 / 失败 —— 后端 publish 后清 app-side "打包中..."
      // 状态。<a> 直链已经把传输交给浏览器原生下载条；这里只管 prep 阶段反馈。
      (evt.type === 'version_train_zip_ready' || evt.type === 'version_train_zip_failed') &&
      evt.project_id === projectId
    ) {
      setExporting(false)
      if (evt.type === 'version_train_zip_failed') {
        const err = typeof evt.error === 'string' ? evt.error : '?'
        toast(t('layout.exportFailed', { error: err }), 'error')
      }
    }
  })

  // 兜底：SSE 事件丢失 / 后端进程挂了的时候,60s 后强制清 exporting 不让按钮卡死。
  useEffect(() => {
    if (!exporting) return
    const tid = window.setTimeout(() => setExporting(false), 60_000)
    return () => window.clearTimeout(tid)
  }, [exporting])

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

  const handleExportTrain = useCallback(() => {
    if (!projectRef.current || exporting) return
    const av = projectRef.current.versions.find(
      (v) => v.id === projectRef.current!.active_version_id
    ) ?? projectRef.current.versions[0] ?? null
    if (!av) return
    setExporting(true)
    // <a download> 直链 —— 浏览器原生接管下载（进度条 / 暂停 / 切 tab 不中断）。
    // app-side "打包中..." 由 version_train_zip_ready/_failed SSE 清。
    // download 属性是兜底,后端 Content-Disposition.filename 优先。
    const filename = `${projectRef.current.slug}-${av.label}.train.zip`
    const a = document.createElement('a')
    a.href = api.versionTrainZipUrl(projectRef.current.id, av.id)
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }, [exporting])

  const handleDeleteVersion = useCallback(async (vid: number) => {
    if (!projectRef.current) return
    const v = projectRef.current.versions.find((x) => x.id === vid)
    if (!v) return
    if (!(await confirm(t('layout.deleteVersionConfirm', { label: v.label }), { tone: 'danger', okText: t('layout.deleteVersionOk') }))) return
    const pid = projectRef.current.id
    try {
      await api.deleteVersion(pid, vid)
      await reload()
      toast(t('layout.deleteVersionDone', { label: v.label }), 'success')
      navigate(`/projects/${pid}`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [reload, toast, navigate, confirm, t])

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
          ? t('layout.versionCreatedFromFork', { label })
          : t('layout.versionCreated', { label }),
        'success',
      )
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setCreatingBusy(false)
    }
  }, [creatingBusy, reload, toast, t])

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
    return <p className="p-6 text-fg-tertiary">{t('layout.loading')}</p>
  }

  return (
    <div className="flex flex-col h-full">
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
  const { t } = useTranslation()
  const [label, setLabel] = useState('')
  const [forkFrom, setForkFrom] = useState<string>('')
  const [err, setErr] = useState<string | null>(null)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (busy) return
    const l = label.trim()
    if (!l) return setErr(t('layout.labelEmpty'))
    if (!/^[A-Za-z0-9_.-]+$/.test(l))
      return setErr(t('layout.labelInvalid'))
    if (existingLabels.includes(l)) return setErr(t('layout.labelExists'))
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
        <h2 className="m-0 text-lg font-semibold">{t('layout.newVersionTitle')}</h2>
        <label className="flex flex-col gap-1">
          <span className="text-xs text-fg-tertiary font-mono">label</span>
          <input
            autoFocus
            value={label}
            onChange={(e) => { setLabel(e.target.value); setErr(null) }}
            className="input input-mono"
            placeholder={t('layout.labelPlaceholder')}
          />
        </label>
        {existingVersions.length > 0 && (
          <label className="flex flex-col gap-1">
            <span className="text-xs text-fg-tertiary font-mono">{t('layout.forkFrom')}</span>
            <select
              value={forkFrom}
              onChange={(e) => setForkFrom(e.target.value)}
              className="input"
            >
              <option value="">{t('layout.forkBlank')}</option>
              {existingVersions.map((v) => (
                <option key={v.id} value={String(v.id)}>
                  {t('layout.forkFromVersion', { label: v.label })}
                </option>
              ))}
            </select>
            {forkFrom !== '' && (
              <p className="m-0 text-xs text-fg-tertiary">
                {t('layout.forkNote')}
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
            {t('common.cancel')}
          </button>
          <button
            type="submit"
            disabled={busy}
            className="btn btn-primary"
          >
            {busy ? t('layout.creatingBtn') : t('common.create')}
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
