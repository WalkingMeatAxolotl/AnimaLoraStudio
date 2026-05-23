import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useOutletContext } from 'react-router-dom'
import { api, type ProjectDetail, type Task, type Version } from '../../api/client'
import PageHeader from '../../components/PageHeader'
import StageBadge from '../../components/StageBadge'
import VersionStatusBadge from '../../components/VersionStatusBadge'
import { useToast } from '../../components/Toast'

type OverviewTab = 'details' | 'tasks' | 'output'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  onCreateVersion: () => void
  creatingVersionBusy: boolean
}

// ── StatCard ────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  sub,
  tone,
  mono = true,
}: {
  label: string
  value: string | number
  sub?: string
  tone?: 'ok' | 'warn' | 'err' | 'accent'
  mono?: boolean
}) {
  const colorCls =
    tone === 'ok'     ? 'text-ok'
    : tone === 'warn' ? 'text-warn'
    : tone === 'err'  ? 'text-err'
    : tone === 'accent' ? 'text-accent'
    : 'text-fg-primary'
  return (
    <div className="card" style={{ padding: 18 }}>
      <div className="caption mb-2.5">{label}</div>
      <div
        className={`text-2xl ${colorCls} ${mono ? 'font-mono' : ''}`}
        style={{ fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1.05 }}
      >{value}</div>
      {sub && <div className="mt-1.5 text-sm text-fg-tertiary">{sub}</div>}
    </div>
  )
}

// ── PipelineTimeline ─────────────────────────────────────────────

type StepStatus = 'done' | 'active' | 'pending'

interface PipelineStep {
  idx: number
  label: string
  status: StepStatus
  meta: string
}

function deriveTimeline(
  project: ProjectDetail,
  activeVersion: Version | null,
  t: (k: string, o?: Record<string, unknown>) => string,
): PipelineStep[] {
  const stage = activeVersion?.stage ?? project.stage
  const stageOrder = ['downloading', 'preprocessing', 'curating', 'tagging', 'regularizing', 'configured', 'training', 'done']
  const stageIdx = stageOrder.indexOf(stage)

  const steps: Array<{ label: string; stages: string[]; meta: () => string }> = [
    {
      label: t('overview.stepDownload'),
      stages: ['downloading'],
      meta: () => t('overview.nImages', { n: project.download_image_count ?? 0 }),
    },
    {
      label: t('overview.stepPreprocess'),
      stages: ['preprocessing'],
      meta: () => {
        const n = project.preprocess_image_count ?? 0
        return n > 0 ? t('overview.nImages', { n }) : '—'
      },
    },
    {
      label: t('overview.stepCurate'),
      stages: ['curating'],
      meta: () => {
        const n = activeVersion?.stats?.train_image_count ?? 0
        return n > 0 ? t('overview.nImages', { n }) : '—'
      },
    },
    {
      label: t('overview.stepTag'),
      stages: ['tagging'],
      meta: () => {
        const n = activeVersion?.stats?.train_image_count ?? 0
        return n > 0 ? t('overview.nImages', { n }) : '—'
      },
    },
    {
      label: t('overview.stepTagEdit'),
      stages: ['regularizing'],
      meta: () => '—',
    },
    {
      label: t('overview.stepReg'),
      stages: ['configured'],
      meta: () => {
        const n = activeVersion?.stats?.reg_image_count ?? 0
        return n > 0 ? t('overview.nImages', { n }) : '—'
      },
    },
    {
      label: t('overview.stepTrain'),
      stages: ['training', 'done'],
      meta: () => activeVersion?.stats?.has_output ? t('overview.hasOutput') : '—',
    },
  ]

  return steps.map((s, i) => {
    const stepFirstStageIdx = stageOrder.indexOf(s.stages[0])
    let status: StepStatus = 'pending'
    if (stage === 'done' || s.stages.some(st => st === 'done')) {
      status = stage === 'done' ? 'done' : 'pending'
    }
    if (stageIdx > stepFirstStageIdx) status = 'done'
    else if (s.stages.includes(stage)) status = 'active'

    return { idx: i + 1, label: s.label, status, meta: s.meta() }
  })
}

function PipelineTimeline({ steps }: { steps: PipelineStep[] }) {
  return (
    <div className="grid" style={{ gridTemplateColumns: `repeat(${steps.length}, 1fr)` }}>
      {steps.map((s, i) => (
        <div key={i} className="relative px-1 min-w-0">
          {i > 0 && (
            <div
              className={`absolute top-[15px] left-0 h-0.5 ${s.status !== 'pending' ? 'bg-ok' : 'bg-border-subtle'}`}
              style={{ width: 'calc(50% - 15px)' }}
            />
          )}
          {i < steps.length - 1 && (
            <div
              className={`absolute top-[15px] right-0 h-0.5 ${s.status === 'done' ? 'bg-ok' : 'bg-border-subtle'}`}
              style={{ width: 'calc(50% - 15px)' }}
            />
          )}
          <div className="flex flex-col items-center text-center relative min-w-0">
            <div
              className={`w-[30px] h-[30px] rounded-full grid place-items-center font-mono font-bold text-xs shrink-0 ${
                s.status === 'done'   ? 'bg-ok text-fg-inverse'
                : s.status === 'active' ? 'bg-accent text-fg-inverse ring-[3px] ring-accent-soft'
                : 'bg-overlay text-fg-tertiary'
              }`}
            >
              {s.status === 'done' ? '✓' : s.idx}
            </div>
            <div className={`mt-2 text-sm font-medium leading-tight max-w-full overflow-hidden text-ellipsis whitespace-nowrap ${
              s.status === 'pending' ? 'text-fg-tertiary' : 'text-fg-primary'
            }`}>
              {s.label}
            </div>
            <div className={`text-xs mt-0.5 max-w-full overflow-hidden text-ellipsis whitespace-nowrap ${
              s.meta === '—' ? 'text-fg-disabled' : 'text-fg-tertiary'
            }`}>
              {s.meta}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── DatasetDetailGrid (ADR-0007 §11.8-C) ─────────────────────────

/** 数据集统计 5 格 grid card：每格 empty state 链向关联 phase 页面。 */
function DatasetDetailGrid({
  project, activeVersion,
}: { project: ProjectDetail; activeVersion: Version | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const stats = activeVersion?.stats
  const trainCount = stats?.train_image_count ?? 0
  const taggedCount = stats?.tagged_image_count ?? 0
  const regCount = stats?.reg_image_count ?? 0
  const folders = stats?.train_folders ?? []
  const vid = activeVersion?.id
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
      {/* 1. 文件夹/repeat */}
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

      {/* 2. tag 分布 — 暂置 placeholder + 链到 ⑤ 编辑页 */}
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

      {/* 3. 分辨率分布 — 链到 ② 放大页 */}
      <CardShell
        title={t('overview.detail.resolutionDist')}
        action={{ label: t('overview.detail.goUpscale'), onClick: () => navigate(`/projects/${project.id}/preprocess?tool=upscale`) }}
      >
        <EmptyHint k="overview.detail.emptyResolution" />
      </CardShell>

      {/* 4. 长宽比分布 — 链到 ② 裁剪页 */}
      <CardShell
        title={t('overview.detail.aspectDist')}
        action={{ label: t('overview.detail.goCrop'), onClick: () => navigate(`/projects/${project.id}/preprocess?tool=crop`) }}
      >
        <EmptyHint k="overview.detail.emptyAspect" />
      </CardShell>

      {/* 5. 正则集 */}
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

// ── ProjectTasksPanel (ADR-0007 §11.8-C) ─────────────────────────

/** [Tasks] tab：列本项目的训练任务（按 created_at 倒序）。点行跳 /queue/:tid。 */
function ProjectTasksPanel({ projectId }: { projectId: number }) {
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
          .filter((t) => t.project_id === projectId)
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
        setTasks(filtered)
      })
      .catch(() => { if (!cancelled) setTasks([]) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [projectId])

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

// ── ProjectOutputPanel (ADR-0007 §11.8-C) ────────────────────────

/** [Output] tab：按 version 列出主 LoRA artifact + 链 /queue/:tid#outputs 看 step/epoch ckpts。 */
function ProjectOutputPanel({ project }: { project: ProjectDetail }) {
  const { t } = useTranslation()
  const navigate = useNavigate()

  const withOutput = project.versions.filter((v) => v.output_lora_path || v.stats?.has_output)
  if (withOutput.length === 0) {
    return <div className="p-6 text-fg-tertiary text-sm italic">{t('overview.outputEmpty')}</div>
  }

  return (
    <div className="p-6 flex flex-col gap-3">
      {withOutput.map((v) => (
        <div key={v.id} className="card" style={{ padding: 16 }}>
          <div className="flex items-center mb-2">
            <span className="font-mono font-semibold flex-1">{v.label}</span>
            <VersionStatusBadge status={v.status} />
          </div>
          {v.output_lora_path && (
            <p className="m-0 text-xs text-fg-tertiary font-mono break-all">{v.output_lora_path}</p>
          )}
          <div className="mt-2 flex gap-2">
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => navigate(`/projects/${project.id}/v/${v.id}/train`)}
            >
              {t('overview.outputOpenTrain')}
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Overview ─────────────────────────────────────────────────────

export default function ProjectOverview() {
  const { t } = useTranslation()
  const { project, activeVersion, reload, onCreateVersion, creatingVersionBusy } = useOutletContext<Ctx>()
  const navigate = useNavigate()
  const { toast } = useToast()
  const [relatedTasks, setRelatedTasks] = useState<Task[]>([])
  // ADR-0007 §11.8-C: 三 tab 框架。后续 commit 把 details 改成 grid 布局 +
  // 实装 tasks / output 面板内容。
  const [activeTab, setActiveTab] = useState<OverviewTab>('details')

  useEffect(() => {
    let cancelled = false
    void api.listQueue('done', { includeGenerate: true })
      .then((items) => {
        if (cancelled) return
        setRelatedTasks(items.filter(
          (t) => t.project_id === project.id && t.config_name === 'generate',
        ))
      })
      .catch(() => {
        if (!cancelled) setRelatedTasks([])
      })
    return () => { cancelled = true }
  }, [project.id])

  const handleActivate = async (v: Version) => {
    try {
      await api.activateVersion(project.id, v.id)
      await reload()
      navigate(`/projects/${project.id}/download`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const stats = [
    {
      label: 'download images',
      value: project.download_image_count ?? 0,
      sub: t('overview.totalDownload'),
    },
    {
      label: 'train images',
      value: activeVersion?.stats?.train_image_count ?? 0,
      sub: t('overview.currentVersion', { label: activeVersion?.label ?? '—' }),
    },
    {
      label: 'reg images',
      value: activeVersion?.stats?.reg_image_count ?? 0,
      sub: activeVersion?.stats?.has_output ? t('overview.hasCkpt') : t('overview.noTrained'),
      tone: activeVersion?.stats?.has_output ? 'ok' as const : undefined,
    },
    {
      label: t('overview.versionCount'),
      value: project.versions.length,
      sub: t('overview.activeVersion', { label: activeVersion?.label ?? '—' }),
      tone: 'accent' as const,
      mono: false,
    },
  ]

  const steps = deriveTimeline(project, activeVersion, t)

  const latestOutputTaskByVersion = useMemo(() => {
    const out = new Map<number, Task>()
    const byFinished = [...relatedTasks].sort(
      (a, b) => (b.finished_at ?? 0) - (a.finished_at ?? 0),
    )
    for (const task of byFinished) {
      if (task.version_id == null) continue
      if (!out.has(task.version_id)) out.set(task.version_id, task)
    }
    return out
  }, [relatedTasks])

  const nextStep = steps.find(s => s.status === 'active')
  const nextStepPaths: Record<string, string> = {
    [t('overview.stepDownload')]:  'download',
    [t('overview.stepCurate')]:    `v/${activeVersion?.id}/curate`,
    [t('overview.stepTag')]:       `v/${activeVersion?.id}/tag`,
    [t('overview.stepTagEdit')]:   `v/${activeVersion?.id}/edit`,
    [t('overview.stepReg')]:       `v/${activeVersion?.id}/reg`,
    [t('overview.stepTrain')]:     `v/${activeVersion?.id}/train`,
  }
  const nextPath = nextStep ? nextStepPaths[nextStep.label] : undefined

  // ADR-0007 §11.8-C 右上角 = 当前 version 的 status badge
  const headerActions = (
    <div className="flex items-center gap-3">
      {activeVersion && (
        <VersionStatusBadge status={activeVersion.status} />
      )}
      {nextPath ? (
        <button
          className="btn btn-primary"
          onClick={() => navigate(`/projects/${project.id}/${nextPath}`)}
        >
          {t('overview.continueStep', { label: nextStep?.label })}
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </button>
      ) : null}
    </div>
  )

  const tabBtnCls = (tab: OverviewTab) => [
    'px-4 py-2 text-sm border-none bg-transparent cursor-pointer border-b-2 transition-colors',
    activeTab === tab
      ? 'text-fg-primary font-semibold border-accent'
      : 'text-fg-secondary border-transparent hover:text-fg-primary',
  ].join(' ')

  return (
    <div className="fade-in">
      <PageHeader
        title={`${project.title}${activeVersion ? ` / ${activeVersion.label}` : ''}`}
        subtitle={project.note || t('overview.subtitle', { n: project.download_image_count ?? 0, v: project.versions.length })}
        actions={headerActions}
      />

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

      {activeTab === 'tasks' && (
        <ProjectTasksPanel projectId={project.id} />
      )}

      {activeTab === 'output' && (
        <ProjectOutputPanel project={project} />
      )}

      {activeTab === 'details' && (
      <div className="p-6 flex flex-col gap-5">
        {/* ADR-0007 §11.8-C [详情] tab grid 布局：5 个 card 复用关联 phase 页面的统计风格 */}
        <DatasetDetailGrid project={project} activeVersion={activeVersion} />

        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
          {stats.map((s, i) => (
            <StatCard key={i} {...s} />
          ))}
        </div>

        <div className="card" style={{ padding: 18 }}>
          <div className="flex items-center mb-3.5">
            <h2 className="text-md font-semibold flex-1" style={{ margin: 0 }}>{t('overview.pipelineProgress')}</h2>
            <span className="caption">{t('overview.stages')}</span>
          </div>
          <div>
            <PipelineTimeline steps={steps} />
          </div>
        </div>

        <div className="card" style={{ padding: 18 }}>
          <div className="flex items-center mb-3.5">
            <h2 className="text-md font-semibold flex-1" style={{ margin: 0 }}>{t('overview.versions')}</h2>
            <button
              className="btn btn-ghost btn-sm border border-dashed border-dim"
              onClick={onCreateVersion}
              disabled={creatingVersionBusy}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
              {creatingVersionBusy ? t('overview.creating') : t('overview.newVersion')}
            </button>
          </div>

          <div className="flex flex-col gap-2">
            {project.versions.map((v) => {
              const isActive = v.id === project.active_version_id
              return (
                <div
                  key={v.id}
                  className={`p-3.5 rounded-md ${
                    isActive ? 'border border-accent bg-accent-soft' : 'border border-subtle'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-mono font-semibold">{v.label}</span>
                    <StageBadge stage={v.stage} />
                  </div>
                  <div className="mt-1.5 flex gap-3.5 text-sm text-fg-secondary">
                    <span>{t('overview.trainImages', { n: v.stats?.train_image_count ?? 0 })}</span>
                    <span>{t('overview.regImages', { n: v.stats?.reg_image_count ?? 0 })}</span>
                    {v.stats?.has_output && (
                      <span className="text-ok">{t('overview.trained')}</span>
                    )}
                  </div>
                  {v.note && (
                    <p className="mt-1.5 text-sm text-fg-secondary">{v.note}</p>
                  )}
                  <div className="mt-2.5">
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleActivate(v)}
                    >
                      {isActive ? t('overview.open') : t('overview.activateAndOpen')}
                    </button>
                    {latestOutputTaskByVersion.has(v.id) && (
                      <button
                        className="btn btn-ghost btn-sm ml-2"
                        onClick={() => navigate(`/queue/${latestOutputTaskByVersion.get(v.id)!.id}#outputs`)}
                        title={t('overview.viewOutput')}
                      >
                        {t('overview.viewOutput')}
                      </button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
      )}
    </div>
  )
}
