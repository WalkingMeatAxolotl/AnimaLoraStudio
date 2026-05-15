import { Link, useLocation } from 'react-router-dom'
import type { ProjectDetail, Version } from '../api/client'

interface Step {
  key: string
  label: string
  scope: 'project' | 'version'
}

const STEPS: Step[] = [
  { key: 'download', label: '① 下载', scope: 'project' },
  { key: 'preprocess', label: '② 预处理', scope: 'project' },
  { key: 'curate', label: '③ 筛选', scope: 'version' },
  { key: 'tag', label: '④ 打标', scope: 'version' },
  { key: 'edit', label: '⑤ 标签编辑', scope: 'version' },
  { key: 'reg', label: '⑥ 正则集', scope: 'version' },
  { key: 'train', label: '⑦ 训练', scope: 'version' },
]

/** 根据 project.stage 推断各步状态：✓ 完成 / ● 当前 / ○ 未开始 */
function statusFor(
  step: Step,
  project: ProjectDetail,
  version: Version | null
): 'done' | 'active' | 'pending' {
  // 极简：项目 stage > 该步对应 stage 阈值 → done；等于 → active；小于 → pending
  // tag / edit 都映射到后端 stage "tagging"（编辑只是 tagging 阶段的子步骤）
  // preprocess 是可选阶段：用户没启动也不卡 curate（派生覆盖见下）。
  const order = ['created', 'downloading', 'preprocessing', 'curating', 'tagging', 'regularizing', 'configured', 'training', 'done']
  const stepIdx: Record<string, number> = {
    download: 1, // downloading
    preprocess: 2, // preprocessing（可选）
    curate: 3,
    tag: 4,
    edit: 4,
    reg: 5,
    train: 7, // configured/training
  }
  const projIdx = order.indexOf(project.stage)
  const target = stepIdx[step.key] ?? 0
  let status: 'done' | 'active' | 'pending' = 'pending'
  if (projIdx > target) status = 'done'
  else if (projIdx === target) status = 'active'

  // preprocess 是可选阶段，状态完全派生自 preprocess_image_count：
  //   >0 → done（确实做了）； =0 → pending（即使用户已经走到 curating 之后）。
  // 这样跳过预处理的项目不会被误标 ✓，回头还能进来跑增量。
  if (step.key === 'preprocess') {
    const cnt = (project as ProjectDetail & { preprocess_image_count?: number }).preprocess_image_count ?? 0
    return cnt > 0 ? 'done' : 'pending'
  }

  // version 级 stage 也参考（active version 进入 tagging 时 tag step 标 done）
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
  // reg 派生覆盖：reg/ 已生成（meta 存在 + 至少 1 张图）→ done
  if (step.key === 'reg' && version?.stats) {
    const s = version.stats
    if (s.reg_meta_exists && s.reg_image_count > 0) {
      return 'done'
    }
  }
  // train 派生：output_lora_path 存在 → done；version.stage=training → active
  if (step.key === 'train' && version) {
    if (version.output_lora_path) return 'done'
    if (version.stage === 'training') return 'active'
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
            ? 'text-ok'
            : status === 'active'
              ? 'text-accent'
              : 'text-fg-tertiary'
        const activeCls = isActiveRoute ? 'bg-surface' : 'hover:bg-overlay'
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
