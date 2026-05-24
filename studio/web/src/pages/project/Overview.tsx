import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useOutletContext } from 'react-router-dom'
import { api, type ProjectDetail, type Task, type Version } from '../../api/client'
import VersionStatusBadge from '../../components/VersionStatusBadge'

type OverviewTab = 'details' | 'tasks' | 'output'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  onCreateVersion: () => void
  creatingVersionBusy: boolean
}

// ── DatasetDetailGrid (ADR-0007 §11.8-C [详情] tab) ───────────────────

/** 数据集统计 5 格 grid card：每格 empty state 链向关联 phase 页面。 */
function DatasetDetailGrid({
  project, version,
}: { project: ProjectDetail; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const stats = version?.stats
  const trainCount = stats?.train_image_count ?? 0
  const taggedCount = stats?.tagged_image_count ?? 0
  const regCount = stats?.reg_image_count ?? 0
  const folders = stats?.train_folders ?? []
  const vid = version?.id
  const goPhase = (key: string) => () => vid && navigate(`/projects/${project.id}/v/${vid}/${key}`)

  const CardShell = ({ title, children, action }: { title: string; children: React.ReactNode; action?: { label: string; onClick: () => void } }) => (
    <div className="card flex flex-col gap-2" style={{ padding: 16 }}>
      <div className="flex items-center">
        <h3 className="text-sm font-semibold flex-1 m-0">{title}</h3>
        {action && (
          <button className="btn btn-ghost btn-xs" onClick={action.onClick}>{action.label}</button>
        )}
      </div>
      <div className="text-sm text-fg-secondary">{children}</div>
    </div>
  )

  const EmptyHint = ({ k }: { k: string }) => (
    <p className="m-0 text-fg-tertiary text-xs italic">{t(k)}</p>
  )

  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
      <CardShell
        title={t('overview.detail.folders')}
        action={vid ? { label: t('overview.detail.goCurate'), onClick: goPhase('curate') } : undefined}
      >
        {folders.length === 0 ? (
          <EmptyHint k="overview.detail.emptyCurate" />
        ) : (
          <>
            <ul className="m-0 pl-4 font-mono text-xs flex flex-col gap-0.5">
              {folders.map((f) => (
                <li key={f.name}>{f.name} · {f.image_count}</li>
              ))}
            </ul>
            <p className="mt-1.5 m-0 text-xs text-fg-tertiary">
              {t('overview.detail.foldersTotal', { n: trainCount })}
            </p>
          </>
        )}
      </CardShell>

      <CardShell
        title={t('overview.detail.tagDist')}
        action={vid ? { label: t('overview.detail.goEdit'), onClick: goPhase('edit') } : undefined}
      >
        {trainCount === 0 || taggedCount === 0 ? (
          <EmptyHint k="overview.detail.emptyTag" />
        ) : (
          <p className="m-0 text-xs text-fg-tertiary">
            {t('overview.detail.tagCoverage', { tagged: taggedCount, total: trainCount })}
          </p>
        )}
      </CardShell>

      <CardShell
        title={t('overview.detail.resolutionDist')}
        action={{ label: t('overview.detail.goUpscale'), onClick: () => navigate(`/projects/${project.id}/preprocess?tool=upscale`) }}
      >
        <EmptyHint k="overview.detail.emptyResolution" />
      </CardShell>

      <CardShell
        title={t('overview.detail.aspectDist')}
        action={{ label: t('overview.detail.goCrop'), onClick: () => navigate(`/projects/${project.id}/preprocess?tool=crop`) }}
      >
        <EmptyHint k="overview.detail.emptyAspect" />
      </CardShell>

      <CardShell
        title={t('overview.detail.regSet')}
        action={vid ? { label: t('overview.detail.goReg'), onClick: goPhase('reg') } : undefined}
      >
        {regCount === 0 ? (
          <EmptyHint k="overview.detail.emptyReg" />
        ) : (
          <p className="m-0 text-xs text-fg-tertiary">{t('overview.detail.regCount', { n: regCount })}</p>
        )}
      </CardShell>
    </div>
  )
}

// ── VersionTasksPanel (ADR-0007 §11.8-C [Tasks] tab — version scope) ──

function VersionTasksPanel({ projectId, versionId }: { projectId: number; versionId: number | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    void api.listQueue()
      .then((items) => {
        if (cancelled) return
        const filtered = items
          .filter((tk) => tk.project_id === projectId && (versionId == null || tk.version_id === versionId))
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
        setTasks(filtered)
      })
      .catch(() => { if (!cancelled) setTasks([]) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [projectId, versionId])

  if (loading) {
    return <div className="p-6 text-fg-tertiary text-sm">{t('common.loading')}</div>
  }

  if (tasks.length === 0) {
    return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.tasksEmpty')}</div>
  }

  const fmtTime = (ts: number | null) => ts ? new Date(ts * 1000).toLocaleString() : '—'

  return (
    <div className="p-6">
      <table className="w-full text-sm">
        <thead className="text-fg-tertiary text-xs">
          <tr className="border-b border-subtle">
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.name')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.status')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.started')}</th>
            <th className="text-left py-2 px-3 font-normal">{t('overview.tasksTable.finished')}</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((tk) => (
            <tr
              key={tk.id}
              className="border-b border-subtle cursor-pointer hover:bg-overlay"
              onClick={() => navigate(`/queue/${tk.id}`)}
            >
              <td className="py-2 px-3 font-mono">#{tk.id} {tk.name}</td>
              <td className="py-2 px-3"><span className={`badge badge-${TASK_STATUS_BADGE[tk.status] ?? 'neutral'}`}>{tk.status}</span></td>
              <td className="py-2 px-3 text-fg-tertiary text-xs">{fmtTime(tk.started_at ?? null)}</td>
              <td className="py-2 px-3 text-fg-tertiary text-xs">{fmtTime(tk.finished_at ?? null)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const TASK_STATUS_BADGE: Record<string, string> = {
  pending: 'neutral', running: 'accent', paused: 'warn',
  done: 'ok', failed: 'err', canceled: 'neutral',
}

// ── VersionOutputPanel (ADR-0007 §11.8-C [Output] tab — 单 version) ─────

function VersionOutputPanel({ projectId, version }: { projectId: number; version: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()

  if (!version) {
    return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.outputEmpty')}</div>
  }
  if (!version.output_lora_path && !version.stats?.has_output) {
    return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.outputEmptyVersion')}</div>
  }

  return (
    <div className="p-6 flex flex-col gap-3">
      <div className="card" style={{ padding: 16 }}>
        <div className="flex items-center mb-2">
          <span className="font-mono font-semibold flex-1">{version.label}</span>
          <VersionStatusBadge status={version.status} />
        </div>
        {version.output_lora_path && (
          <p className="m-0 text-xs text-fg-tertiary font-mono break-all">{version.output_lora_path}</p>
        )}
        <div className="mt-2 flex gap-2">
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => navigate(`/projects/${projectId}/v/${version.id}/train`)}
          >
            {t('overview.outputOpenTrain')}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Overview (ADR-0007 §11.8-C 重排：项目级上半 + version-select + 3 tab) ──

export default function ProjectOverview() {
  const { t } = useTranslation()
  const { project, activeVersion } = useOutletContext<Ctx>()
  const navigate = useNavigate()

  // dropdown 独立 selection：初值 = active_version_id，改它不动 sidebar active。
  // 切 sidebar active version 重渲染会触发 project 重读，此时若 selected 还是合法
  // 就保留；非法或不存在则回退到新的 active_version_id。
  const [selectedVid, setSelectedVid] = useState<number | null>(
    project.active_version_id ?? activeVersion?.id ?? null,
  )
  useEffect(() => {
    const stillExists = project.versions.some((v) => v.id === selectedVid)
    if (!stillExists) setSelectedVid(project.active_version_id ?? null)
  }, [project.versions, project.active_version_id, selectedVid])

  const selectedVersion: Version | null =
    project.versions.find((v) => v.id === selectedVid) ?? null

  const [activeTab, setActiveTab] = useState<OverviewTab>('details')

  const tabBtnCls = (tab: OverviewTab) => [
    'px-4 py-2 text-sm border-none bg-transparent cursor-pointer border-b-2 transition-colors',
    activeTab === tab
      ? 'text-fg-primary font-semibold border-accent'
      : 'text-fg-secondary border-transparent hover:text-fg-primary',
  ].join(' ')

  const created = project.created_at
    ? new Date(project.created_at * 1000).toLocaleDateString()
    : '—'

  return (
    <div className="fade-in">
      {/* ── 项目级 header — 不随 dropdown 变 ────────────────────────── */}
      <div className="px-6 pt-5 pb-4 bg-canvas border-b border-subtle">
        <h1 className="m-0 text-2xl font-semibold tracking-tight leading-[1.15]">{project.title}</h1>
        <p className="mt-1.5 m-0 text-xs text-fg-tertiary font-mono">
          {t('overview.header.slug', { slug: project.slug })}
          <span className="mx-2">·</span>
          {t('overview.header.created', { date: created })}
        </p>
        {project.note && (
          <p className="mt-2 mb-0 text-sm text-fg-secondary max-w-[720px]">{project.note}</p>
        )}
      </div>

      {/* ── 项目级 meta bar：数据集 + 版本数 ───────────────────────── */}
      <div className="px-6 py-3 bg-sunken border-b border-subtle flex items-center gap-6 flex-wrap text-sm">
        <div className="flex items-center gap-2">
          <span className="text-fg-tertiary">{t('overview.header.dataset')}:</span>
          <span className="font-mono">
            {t('overview.header.downloadCount', { n: project.download_image_count ?? 0 })}
          </span>
          <span className="text-fg-tertiary">·</span>
          <span className="font-mono">
            {t('overview.header.preprocessCount', { n: project.preprocess_image_count ?? 0 })}
          </span>
          <button
            className="btn btn-ghost btn-xs ml-1"
            onClick={() => navigate(`/projects/${project.id}/download`)}
          >
            {t('overview.header.manage')}
          </button>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-fg-tertiary">{t('overview.header.versions')}:</span>
          <span className="font-mono">{project.versions.length}</span>
        </div>
      </div>

      {/* ── version 选择 — 独立于 sidebar active ─────────────────────── */}
      <div className="px-6 py-3 bg-canvas border-b border-subtle flex items-center gap-3 flex-wrap">
        <label htmlFor="version-select" className="text-sm text-fg-secondary">
          {t('overview.versionSelector.label')}:
        </label>
        <select
          id="version-select"
          className="input input-mono"
          style={{ width: 'auto', minWidth: 160 }}
          value={selectedVid ?? ''}
          onChange={(e) => setSelectedVid(e.target.value ? Number(e.target.value) : null)}
        >
          {project.versions.length === 0 ? (
            <option value="">{t('overview.versionSelector.empty')}</option>
          ) : (
            project.versions.map((v) => (
              <option key={v.id} value={v.id}>{v.label}</option>
            ))
          )}
        </select>
        {selectedVersion && (
          <div className="ml-auto">
            <VersionStatusBadge status={selectedVersion.status} />
          </div>
        )}
      </div>

      {/* ── version scope tabs ────────────────────────────────────── */}
      <div className="border-b border-subtle px-6">
        <div className="flex gap-1">
          <button className={tabBtnCls('details')} onClick={() => setActiveTab('details')}>
            {t('overview.tabDetails')}
          </button>
          <button className={tabBtnCls('tasks')} onClick={() => setActiveTab('tasks')}>
            {t('overview.tabTasks')}
          </button>
          <button className={tabBtnCls('output')} onClick={() => setActiveTab('output')}>
            {t('overview.tabOutput')}
          </button>
        </div>
      </div>

      {activeTab === 'details' && (
        <div className="p-6">
          <DatasetDetailGrid project={project} version={selectedVersion} />
        </div>
      )}

      {activeTab === 'tasks' && (
        <VersionTasksPanel projectId={project.id} versionId={selectedVid} />
      )}

      {activeTab === 'output' && (
        <VersionOutputPanel projectId={project.id} version={selectedVersion} />
      )}
    </div>
  )
}
