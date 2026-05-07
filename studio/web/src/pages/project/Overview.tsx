import { useState } from 'react'
import { useNavigate, useOutletContext } from 'react-router-dom'
import { api, type ProjectDetail, type Version } from '../../api/client'
import PageHeader from '../../components/PageHeader'
import StageBadge from '../../components/StageBadge'
import { useToast } from '../../components/Toast'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
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
        style={{
          fontWeight: 600,
          letterSpacing: '-0.02em',
          lineHeight: 1.05,
        }}
      >{value}</div>
      {sub && (
        <div className="mt-1.5 text-sm text-fg-tertiary">{sub}</div>
      )}
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

function deriveTimeline(project: ProjectDetail, activeVersion: Version | null): PipelineStep[] {
  const stage = activeVersion?.stage ?? project.stage
  const stageOrder = ['downloading', 'curating', 'tagging', 'regularizing', 'configured', 'training', 'done']
  const stageIdx = stageOrder.indexOf(stage)

  const steps: Array<{ label: string; stages: string[]; meta: () => string }> = [
    {
      label: '下载',
      stages: ['downloading'],
      meta: () => `${project.download_image_count ?? 0} 张`,
    },
    {
      label: '筛选',
      stages: ['curating'],
      meta: () => {
        const n = activeVersion?.stats?.train_image_count ?? 0
        return n > 0 ? `${n} 张` : '—'
      },
    },
    {
      label: '打标',
      stages: ['tagging'],
      meta: () => {
        const n = activeVersion?.stats?.train_image_count ?? 0
        return n > 0 ? `${n} 张` : '—'
      },
    },
    {
      label: '标签编辑',
      stages: ['regularizing'],
      meta: () => '—',
    },
    {
      label: '正则集',
      stages: ['configured'],
      meta: () => {
        const n = activeVersion?.stats?.reg_image_count ?? 0
        return n > 0 ? `${n} 张` : '—'
      },
    },
    {
      label: '训练',
      stages: ['training', 'done'],
      meta: () => activeVersion?.stats?.has_output ? '已出模型' : '—',
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
          {/* left connector */}
          {i > 0 && (
            <div
              className={`absolute top-[15px] left-0 h-0.5 ${s.status !== 'pending' ? 'bg-ok' : 'bg-border-subtle'}`}
              style={{ width: 'calc(50% - 15px)' }}
            />
          )}
          {/* right connector */}
          {i < steps.length - 1 && (
            <div
              className={`absolute top-[15px] right-0 h-0.5 ${s.status === 'done' ? 'bg-ok' : 'bg-border-subtle'}`}
              style={{ width: 'calc(50% - 15px)' }}
            />
          )}
          <div className="flex flex-col items-center text-center relative min-w-0">
            {/* 步骤圆点 */}
            <div
              className={`w-[30px] h-[30px] rounded-full grid place-items-center font-mono font-bold text-xs shrink-0 ${
                s.status === 'done'   ? 'bg-ok text-fg-inverse'
                : s.status === 'active' ? 'bg-accent text-fg-inverse ring-[3px] ring-accent-soft'
                : 'bg-overlay text-fg-tertiary'
              }`}
            >
              {s.status === 'done' ? '✓' : s.idx}
            </div>
            {/* 步骤标签 */}
            <div className={`mt-2 text-sm font-medium leading-tight max-w-full overflow-hidden text-ellipsis whitespace-nowrap ${
              s.status === 'pending' ? 'text-fg-tertiary' : 'text-fg-primary'
            }`}>
              {s.label}
            </div>
            {/* 元信息 */}
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

// ── Overview ─────────────────────────────────────────────────────

export default function ProjectOverview() {
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const navigate = useNavigate()
  const { toast } = useToast()
  const [newVersionBusy, setNewVersionBusy] = useState(false)

  const handleActivate = async (v: Version) => {
    try {
      await api.activateVersion(project.id, v.id)
      await reload()
      navigate(`/projects/${project.id}/v/${v.id}/curate`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const handleNewVersion = async () => {
    const label = prompt('版本标签', `v${project.versions.length + 1}`)
    if (!label) return
    setNewVersionBusy(true)
    try {
      await api.createVersion(project.id, { label })
      await reload()
      toast(`已创建版本 ${label}`, 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setNewVersionBusy(false)
    }
  }

  const stats = [
    {
      label: 'download images',
      value: project.download_image_count ?? 0,
      sub: '总下载量',
    },
    {
      label: 'train images',
      value: activeVersion?.stats?.train_image_count ?? 0,
      sub: `当前版本: ${activeVersion?.label ?? '—'}`,
    },
    {
      label: 'reg images',
      value: activeVersion?.stats?.reg_image_count ?? 0,
      sub: activeVersion?.stats?.has_output ? '✓ 已出 checkpoint' : '尚未训练',
      tone: activeVersion?.stats?.has_output ? 'ok' as const : undefined,
    },
    {
      label: '版本数',
      value: project.versions.length,
      sub: `活跃: ${activeVersion?.label ?? '—'}`,
      tone: 'accent' as const,
      mono: false,
    },
  ]

  const steps = deriveTimeline(project, activeVersion)

  const nextStep = steps.find(s => s.status === 'active')
  const nextStepPaths: Record<string, string> = {
    '下载': 'download',
    '筛选': `v/${activeVersion?.id}/curate`,
    '打标': `v/${activeVersion?.id}/tag`,
    '标签编辑': `v/${activeVersion?.id}/edit`,
    '正则集': `v/${activeVersion?.id}/reg`,
    '训练': `v/${activeVersion?.id}/train`,
  }
  const nextPath = nextStep ? nextStepPaths[nextStep.label] : undefined

  return (
    <div className="fade-in">
      <PageHeader
        eyebrow={`项目 · ${project.slug}`}
        title={project.title}
        subtitle={project.note || `${project.download_image_count ?? 0} 张下载 · ${project.versions.length} 个版本`}
        actions={
          nextPath ? (
            <button
              className="btn btn-primary"
              onClick={() => navigate(`/projects/${project.id}/${nextPath}`)}
            >
              继续 → {nextStep?.label}
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </button>
          ) : undefined
        }
      />

      <div className="p-6 flex flex-col gap-5">
        {/* Stat cards */}
        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
          {stats.map((s, i) => (
            <StatCard key={i} {...s} />
          ))}
        </div>

        {/* Pipeline timeline */}
        <div className="card p-0 overflow-hidden">
          <div className="px-[18px] py-3.5 border-b border-subtle flex items-center justify-between">
            <h2 className="text-md font-semibold" style={{ margin: 0 }}>流水线进度</h2>
            <span className="caption">stages</span>
          </div>
          <div style={{ padding: 18 }}>
            <PipelineTimeline steps={steps} />
          </div>
        </div>

        {/* Versions panel */}
        <div className="card" style={{ padding: 18 }}>
          <div className="flex items-center mb-3.5">
            <h2 className="text-md font-semibold flex-1" style={{ margin: 0 }}>版本</h2>
            <button
              className="btn btn-ghost btn-sm border border-dashed border-dim"
              onClick={handleNewVersion}
              disabled={newVersionBusy}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
              {newVersionBusy ? '创建中…' : '新版本'}
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
                    <span>{v.stats?.train_image_count ?? 0} 训练图</span>
                    <span>{v.stats?.reg_image_count ?? 0} 正则图</span>
                    {v.stats?.has_output && (
                      <span className="text-ok">✓ 已训练</span>
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
                      {isActive ? '打开' : '激活并打开'}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
