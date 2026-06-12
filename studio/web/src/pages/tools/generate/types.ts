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

/** 采样器 / 调度器选项 —— 与后端 GenerateParams 的 Literal 保持一致
 *  （studio/api/schemas/generate.py）；新增值两边要同步。 */
export const SAMPLER_OPTIONS = ['er_sde', 'dpmpp_3m_sde'] as const
export type SamplerName = (typeof SAMPLER_OPTIONS)[number]
export const DEFAULT_SAMPLER: SamplerName = 'er_sde'

export const SCHEDULER_OPTIONS = ['simple', 'sgm_uniform'] as const
export type SchedulerName = (typeof SCHEDULER_OPTIONS)[number]
export const DEFAULT_SCHEDULER: SchedulerName = 'simple'
