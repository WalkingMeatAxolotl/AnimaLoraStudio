import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Outlet, useMatch, useNavigate, useParams } from 'react-router-dom'
import { api, type ProjectDetail } from '../../api/client'
import { useProjectCtxSetter, useSelectedProjectSetter } from '../../context/ProjectContext'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import ExportBundleDialog, { type BundleExportOpts } from '../../components/ExportBundleDialog'

export default function ProjectLayout() {
  const { t } = useTranslation()
  const { pid } = useParams()
  const projectId = pid ? Number(pid) : NaN
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const setCtx = useProjectCtxSetter()
  const setSelected = useSelectedProjectSetter()
  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState<{ forkFrom: number | null } | null>(null)
  const [creatingBusy, setCreatingBusy] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [showExportDialog, setShowExportDialog] = useState(false)
  const projectRef = useRef<ProjectDetail | null>(null)
  projectRef.current = project
  // 版本切换请求序号：快速连切时只认最后一次切换的结果，防止先发后至的响应/回滚覆盖新选择。
  const switchSeqRef = useRef(0)
  // 版本切换守卫：步骤页有需确认才能丢弃的状态时（如 TagEdit 未保存编辑）注册，
  // 返回 false 取消切换。切版本重挂载步骤页不走路由导航，useBlocker 拦不住，
  // 需要这条独立通道。
  const switchGuardRef = useRef<(() => Promise<boolean>) | null>(null)
  const setVersionSwitchGuard = useCallback(
    (g: (() => Promise<boolean>) | null) => { switchGuardRef.current = g },
    [],
  )
  // 版本作用域路由（v/:vid/*）下 Outlet 以 activeVersion.id 为 key：切版本强制
  // 步骤页重挂载，本地 state / 缓存全部换代，杜绝「挂着新版本显示旧数据」
  //（如 Curation 的 view 缓存守卫不会因 vid 变化重拉）。Overview / Download 是
  // project 作用域（Overview 另有自己的 selectedVid 本地态），不跟切。
  const inVersionScope = useMatch('/projects/:pid/v/:vid/*') != null

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
      (
        evt.type === 'version_train_zip_ready' ||
        evt.type === 'version_train_zip_failed' ||
        evt.type === 'version_bundle_zip_ready' ||
        evt.type === 'version_bundle_zip_failed'
      ) &&
      evt.project_id === projectId
    ) {
      setExporting(false)
      if (evt.type === 'version_train_zip_failed' || evt.type === 'version_bundle_zip_failed') {
        const err = typeof evt.error === 'string' ? evt.error : '?'
        toast(t('layout.exportFailed', { error: err }), 'error')
      }
    }
  })

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
    const prev = projectRef.current
    if (!prev || prev.active_version_id === vid) return
    const guard = switchGuardRef.current
    if (guard && !(await guard())) return
    const prevVid = prev.active_version_id
    const seq = ++switchSeqRef.current
    // 乐观更新：先本地切换再等后端。activate 往返期间 activeVersion 若停在旧值，
    // 「切完版本马上点开始训练」会把旧版本入队（#386）。
    setProject((cur) => (cur ? { ...cur, active_version_id: vid } : cur))
    try {
      await api.activateVersion(prev.id, vid)
      // 成功不应用响应（瘦响应）：乐观值即服务端新状态，全量数据由
      // project_state_changed → reload 收敛。
    } catch (e) {
      if (seq === switchSeqRef.current) {
        // 只回滚 active_version_id 字段，不整包回退——避免吞掉在途 reload 带来的其他更新。
        setProject((cur) => (cur ? { ...cur, active_version_id: prevVid } : cur))
        toast(String(e), 'error')
      }
    }
  }, [toast])

  const handleExportTrain = useCallback(() => {
    if (!projectRef.current || exporting) return
    setShowExportDialog(true)
  }, [exporting])

  const handleExportBundleConfirm = useCallback(async (opts: BundleExportOpts) => {
    setShowExportDialog(false)
    if (!projectRef.current) return
    const av = projectRef.current.versions.find(
      (v) => v.id === projectRef.current!.active_version_id
    ) ?? projectRef.current.versions[0] ?? null
    if (!av) return
    setExporting(true)
    const bundleOpts = {
      train: opts.train,
      trainCaptions: opts.trainCaptions,
      reg: opts.reg,
      regCaptions: opts.regCaptions,
      includeConfig: opts.includeConfig,
      latentCache: opts.latentCache,
    }
    if (opts.destination === 'download') {
      const filename = `${projectRef.current.slug}-${av.label}.bundle.zip`
      const a = document.createElement('a')
      a.href = api.versionBundleZipUrl(projectRef.current.id, av.id, bundleOpts)
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      return
    }
    try {
      const result = await api.exportBundleToDataExports(projectRef.current.id, av.id, bundleOpts)
      toast(t('layout.exportSavedToDataExports', { filename: result.filename, path: result.path }), 'success')
      setExporting(false)
    } catch (e) {
      setExporting(false)
      toast(t('layout.exportFailed', { error: String(e) }), 'error')
    }
  }, [t, toast])

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
    // 建新版本会激活它 → 步骤页重挂载，同样要过切换守卫（取消则对话框留在原地）。
    const guard = switchGuardRef.current
    if (guard && !(await guard())) return
    setCreatingBusy(true)
    try {
      const body: { label: string; fork_from_version_id?: number } = { label }
      if (forkFromVersionId !== null) body.fork_from_version_id = forkFromVersionId
      const v = await api.createVersion(projectRef.current.id, body)
      await api.activateVersion(projectRef.current.id, v.id)
      await reload()
      setCreating(null)
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
      onCreateVersion: (forkFromVid?: number) => setCreating({ forkFrom: forkFromVid ?? null }),
      onExportTrain: handleExportTrain,
      onDeleteVersion: handleDeleteVersion,
      exporting,
    })
  }, [project, activeVersion, reload, handleSelectVersion, handleExportTrain, handleDeleteVersion, exporting, setCtx])


  useEffect(() => {
    return () => { setCtx?.(null) }
  }, [setCtx])

  // 粘性快照：加载/刷新时写入，离开项目页时**不清**（见 ProjectContext 注释），
  // 让侧边栏跨页保留选中项目用于导航。打开另一个项目会覆盖这份快照。
  useEffect(() => {
    if (project) setSelected?.({ project, activeVersion })
  }, [project, activeVersion, setSelected])

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
      <Outlet key={inVersionScope ? activeVersion?.id ?? -1 : 'project'} context={{
        project,
        activeVersion,
        reload,
        onCreateVersion: (forkFromVid?: number) => setCreating({ forkFrom: forkFromVid ?? null }),
        creatingVersionBusy: creatingBusy,
        setVersionSwitchGuard,
      }} />
      {creating && (
        <NewVersionDialog
          existingLabels={project.versions.map((v) => v.label)}
          existingVersions={project.versions.map((v) => ({ id: v.id, label: v.label }))}
          initialForkFrom={creating.forkFrom}
          busy={creatingBusy}
          onCancel={() => { if (creatingBusy) return; setCreating(null) }}
          onSubmit={handleCreateVersion}
        />
      )}
      {showExportDialog && (
        <ExportBundleDialog
          onConfirm={handleExportBundleConfirm}
          onCancel={() => setShowExportDialog(false)}
        />
      )}
    </div>
  )
}

export function NewVersionDialog({
  existingLabels,
  existingVersions,
  initialForkFrom = null,
  busy = false,
  onCancel,
  onSubmit,
}: {
  existingLabels: string[]
  existingVersions: { id: number; label: string }[]
  /** 打开对话框时预填的 forkFrom version id（null = 不预填，user 自己选）。 */
  initialForkFrom?: number | null
  busy?: boolean
  onCancel: () => void
  onSubmit: (label: string, forkFromVersionId: number | null) => void
}) {
  const { t } = useTranslation()
  const [label, setLabel] = useState('')
  const [forkFrom, setForkFrom] = useState<string>(initialForkFrom != null ? String(initialForkFrom) : '')
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
