/** 测试页面（Generate）共用本地类型 + 常量。 */

import type { VersionStatus } from '../../../api/client'

/** 训练好的 LoRA 视图（InlineLoraPicker / SidebarLoras 共用）。 */
export interface ProjectLora {
  projectId: number
  projectTitle: string
  versionId: number
  versionLabel: string
  status: VersionStatus
  /** output_lora_path —— 必有值（hook 侧已过滤 null） */
  path: string
  createdAt: number
}

export const DEFAULT_NEG =
  'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, bad hands, bad feet, missing fingers, extra fingers, text, watermark, logo, signature'

/** 采样器 / 调度器选项 —— 与后端 GenerateRequest 的 Literal 保持一致
 *  （studio/api/schemas/generate.py）；新增值两边要同步。
 *  按族白名单（多模型 P4-4）：镜像 studio/domain/common.py FAMILY_SAMPLING
 *  （首项 = 族默认值），越族值后端 422。 */
export type GenerateFamily = 'anima' | 'krea2'

export const SAMPLER_OPTIONS_BY_FAMILY = {
  anima: ['er_sde', 'dpmpp_3m_sde'],
  krea2: ['euler'],
} as const satisfies Record<GenerateFamily, readonly string[]>

export const SCHEDULER_OPTIONS_BY_FAMILY = {
  anima: ['simple', 'sgm_uniform'],
  krea2: ['krea2_shift'],
} as const satisfies Record<GenerateFamily, readonly string[]>

/** 切族时应用的生成参数默认值（steps/cfg 对齐族 SamplingDefaults）。 */
export const FAMILY_GENERATE_DEFAULTS = {
  anima: { steps: 25, cfgScale: 4.0 },
  krea2: { steps: 28, cfgScale: 4.5 },
} as const satisfies Record<GenerateFamily, { steps: number; cfgScale: number }>

/** Turbo（蒸馏推理 variant）选中时的参数默认：8 步、无 CFG。 */
export const DISTILLED_GENERATE_DEFAULTS = { steps: 8, cfgScale: 0.0 } as const

// 全量集合（快照校验 / 类型用）；单页下拉请用 *_BY_FAMILY
export const SAMPLER_OPTIONS = ['er_sde', 'dpmpp_3m_sde', 'euler'] as const
export type SamplerName = (typeof SAMPLER_OPTIONS)[number]
export const DEFAULT_SAMPLER: SamplerName = 'er_sde'

export const SCHEDULER_OPTIONS = ['simple', 'sgm_uniform', 'krea2_shift'] as const
export type SchedulerName = (typeof SCHEDULER_OPTIONS)[number]
export const DEFAULT_SCHEDULER: SchedulerName = 'simple'
