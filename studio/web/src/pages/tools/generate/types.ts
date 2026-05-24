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
