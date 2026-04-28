import { Link, useLocation } from 'react-router-dom'
import type { ProjectDetail, Version } from '../api/client'

interface Step {
  key: string
  label: string
  scope: 'project' | 'version'
}

const STEPS: Step[] = [
  { key: 'download', label: '① 下载', scope: 'project' },
  { key: 'curate', label: '② 筛选', scope: 'version' },
  { key: 'tag', label: '③ 打标', scope: 'version' },
  { key: 'edit', label: '④ 标签编辑', scope: 'version' },
  { key: 'reg', label: '⑤ 正则集', scope: 'version' },
  { key: 'train', label: '⑥ 训练', scope: 'version' },
]

/** 根据 project.stage 推断各步状态：✓ 完成 / ● 当前 / ○ 未开始 */
function statusFor(
  step: Step,
  project: ProjectDetail,
  version: Version | null
): 'done' | 'active' | 'pending' {
  // 极简：项目 stage > 该步对应 stage 阈值 → done；等于 → active；小于 → pending
  // tag / edit 都映射到后端 stage "tagging"（编辑只是 tagging 阶段的子步骤）
  const order = ['created', 'downloading', 'curating', 'tagging', 'regularizing', 'configured', 'training', 'done']
  const stepIdx: Record<string, number> = {
    download: 1, // downloading
    curate: 2,
    tag: 3,
    edit: 3,
    reg: 4,
    train: 6, // configured/training
  }
  const projIdx = order.indexOf(project.stage)
  const target = stepIdx[step.key] ?? 0
  let status: 'done' | 'active' | 'pending' = 'pending'
  if (projIdx > target) status = 'done'
  else if (projIdx === target) status = 'active'

  // version 级 stage 也参考（active version 进入 tagging 时 step3 标 done）
  if (status !== 'done' && step.scope === 'version' && version) {
    const vorder = ['curating', 'tagging', 'regularizing', 'ready', 'training', 'done']
    const vstepIdx: Record<string, number> = {
      curate: 0,
      tag: 1,
      edit: 1,
      reg: 2,
      train: 3,
    }
    const vIdx = vorder.indexOf(version.stage)
    const vt = vstepIdx[step.key] ?? -1
    if (vt >= 0) {
      if (vIdx > vt) status = 'done'
      else if (vIdx === vt && status === 'pending') status = 'active'
    }
  }

  // tag / edit 派生覆盖：train 有图 && 全部已打标 → done
  if ((step.key === 'tag' || step.key === 'edit') && version?.stats) {
    const s = version.stats
    if (s.train_image_count > 0 && s.tagged_image_count >= s.train_image_count) {
      return 'done'
    }
  }
  // reg 派生覆盖：reg/1_general/ 已生成（meta 存在 + 至少 1 张图）→ done
  if (step.key === 'reg' && version?.stats) {
    const s = version.stats
    if (s.reg_meta_exists && s.reg_image_count > 0) {
      return 'done'
    }
  }
  return status
}

export default function ProjectStepper({
  project,
  version,
}: {
  project: ProjectDetail
  version: Version | null
}) {
  const loc = useLocation()
  return (
    <ul className="space-y-1" aria-label="pipeline-stepper">
      {STEPS.map((s) => {
        const status = statusFor(s, project, version)
        const icon = status === 'done' ? '✓' : status === 'active' ? '●' : '○'
        const path =
          s.scope === 'project'
            ? `/projects/${project.id}/${s.key}`
            : version
              ? `/projects/${project.id}/v/${version.id}/${s.key}`
              : null
        const isActiveRoute =
          path !== null && loc.pathname.startsWith(path)
        const baseCls =
          'flex items-center gap-2 px-3 py-1.5 rounded text-sm transition'
        const stateCls =
          status === 'done'
            ? 'text-emerald-300'
            : status === 'active'
              ? 'text-cyan-300'
              : 'text-slate-500'
        const activeCls = isActiveRoute ? 'bg-slate-800/80' : 'hover:bg-slate-800/50'
        if (path === null) {
          return (
            <li key={s.key}>
              <span
                className={`${baseCls} ${stateCls} cursor-not-allowed opacity-50`}
                title="先选择 / 创建一个版本"
              >
                <span className="font-mono w-4 text-center">{icon}</span>
                {s.label}
              </span>
            </li>
          )
        }
        return (
          <li key={s.key}>
            <Link to={path} className={`${baseCls} ${stateCls} ${activeCls}`}>
              <span className="font-mono w-4 text-center">{icon}</span>
              {s.label}
            </Link>
          </li>
        )
      })}
    </ul>
  )
}
