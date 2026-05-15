import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useOutletContext } from 'react-router-dom'
import { api, type ProjectDetail, type Task, type Version } from '../../api/client'
import PageHeader from '../../components/PageHeader'
import StageBadge from '../../components/StageBadge'
import { useToast } from '../../components/Toast'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  /** Layout 透传:复用侧边栏 NewVersionDialog,避免 Overview 重复实现 window.prompt 版本。 */
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
  const stageOrder = ['downloading', 'preprocessing', 'curating', 'tagging', 'regularizing', 'configured', 'training', 'done']
  const stageIdx = stageOrder.indexOf(stage)

  const steps: Array<{ label: string; stages: string[]; meta: () => string }> = [
    {
      label: '下载',
      stages: ['downloading'],
      meta: () => `${project.download_image_count ?? 0} 张`,
    },
    {
      label: '预处理',
      stages: ['preprocessing'],
      meta: () => {
        const n = project.preprocess_image_count ?? 0
        return n > 0 ? `${n} 张` : '—'
      },
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
  const { project, activeVersion, reload, onCreateVersion, creatingVersionBusy } = useOutletContext<Ctx>()
  const navigate = useNavigate()
  const { toast } = useToast()
  const [relatedTasks, setRelatedTasks] = useState<Task[]>([])

  useEffect(() => {
    let cancelled = false
    // 只拉 done generate task（测试出图）：
    //  - status='done' 让后端 db.list_tasks 直接过滤，避免老用户队列上千条全量传
    //  - includeGenerate=true 因为 /api/queue 默认隐藏 generate；"查看输出" 链
    //    向的就是 generate sample 目录，train task 的 .safetensors 用版本卡片
    //    自带 ✓ 已训练 / 激活并打开 走另一条路径，不混在 "查看输出" 按钮里
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
      // 选版本 → 跳项目级 download(不是直接跳 curate)。download 是工作流真起点,
      // 用户从这里决定要不要重新下,还是直接往下走 curate/tag/...。Sidebar 切
      // 版本不 navigate(只 activate),两边语义分开:Overview 卡片点击 = 进入版本
      // 工作流,Sidebar 版本切换 = 改上下文不离当前页。
      navigate(`/projects/${project.id}/download`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 「新版本」按钮调 onCreateVersion → Layout 弹 NewVersionDialog(label + 可选
  // fork from + 自动 activate)。Overview 不再自己维护创建逻辑。

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

  const latestOutputTaskByVersion = useMemo(() => {
    const out = new Map<number, Task>()
    // finished_at 排比 id 排准确：priority 调度 / retry 会让 id 顺序 ≠ 完成顺序。
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
          <div className="px-4.5 py-3.5 border-b border-subtle flex items-center justify-between">
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
              onClick={onCreateVersion}
              disabled={creatingVersionBusy}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
              {creatingVersionBusy ? '创建中…' : '新版本'}
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
                    {latestOutputTaskByVersion.has(v.id) && (
                      <button
                        className="btn btn-ghost btn-sm ml-2"
                        onClick={() => navigate(`/queue/${latestOutputTaskByVersion.get(v.id)!.id}#outputs`)}
                        title="打开该版本最近一次完成任务的 output"
                      >
                        查看输出
                      </button>
                    )}
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
