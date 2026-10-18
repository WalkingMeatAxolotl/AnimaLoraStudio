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
export type GenerateFamily = 'anima' | 'krea2' | 'moss'

export const SAMPLER_OPTIONS_BY_FAMILY = {
  anima: ['er_sde', 'dpmpp_3m_sde'],
  krea2: ['euler'],
  moss: ['euler', 'er_sde'],
} as const satisfies Record<GenerateFamily, readonly string[]>

export const SCHEDULER_OPTIONS_BY_FAMILY = {
  anima: ['simple', 'sgm_uniform'],
  // krea2 的 simple 含固定 shift 口径（曾名 krea2_shift，命名对齐 ComfyUI）
  krea2: ['simple'],
  // moss 的 simple = ComfyUI ModelSamplingDiscreteFlow shift=5
  moss: ['simple'],
} as const satisfies Record<GenerateFamily, readonly string[]>

/** 切族时应用的生成参数默认值（steps/cfg 对齐族 SamplingDefaults）。 */
export const FAMILY_GENERATE_DEFAULTS = {
  anima: { steps: 25, cfgScale: 4.0 },
  krea2: { steps: 28, cfgScale: 4.5 },
  moss: { steps: 50, cfgScale: 3.0 },
} as const satisfies Record<GenerateFamily, { steps: number; cfgScale: number }>

/** 老 prefs / 快照里的族字符串 → 合法族（缺省 / 未知 = anima）。 */
export function coerceGenerateFamily(v: unknown): GenerateFamily {
  return v === 'krea2' || v === 'moss' ? v : 'anima'
}

/** Skip Layer Guidance（uncond 分支在 [start, end] 采样进度内跳过这些 DiT 层，
 *  对齐 ComfyUI SkipLayerGuidanceDiTSimple）。layers 是用户输入的原文（逗号分隔）。 */
export interface SkipLayerGuidance { layers: string; start: number; end: number }

/** 支持 SLG 的族及其默认值（镜像后端能力位 skip_layer_guidance + 族 sample_image
 *  默认：MOSS 官方 pipeline 第 22 层、全程）。没列出的族不显示、不发送这三个参数。 */
export const FAMILY_SKIP_LAYER_DEFAULTS: Partial<Record<GenerateFamily, SkipLayerGuidance>> = {
  moss: { layers: '22', start: 0, end: 1 },
}

/** SLG 层号原文 → 非负整数列表；空串 = 关闭（[]）；含非法项返回 null。 */
export function parseSkipLayers(raw: string): number[] | null {
  const parts = raw.split(/[,\s，]+/).filter((s) => s !== '')
  const out: number[] = []
  for (const p of parts) {
    if (!/^\d+$/.test(p)) return null
    out.push(Number(p))
  }
  return out
}

/** Turbo（蒸馏推理 variant）选中时的参数默认：8 步、无 CFG。 */
export const DISTILLED_GENERATE_DEFAULTS = { steps: 8, cfgScale: 0.0 } as const

// 全量集合（快照校验 / 类型用）；单页下拉请用 *_BY_FAMILY
export const SAMPLER_OPTIONS = ['er_sde', 'dpmpp_3m_sde', 'euler'] as const
export type SamplerName = (typeof SAMPLER_OPTIONS)[number]
export const DEFAULT_SAMPLER: SamplerName = 'er_sde'

export const SCHEDULER_OPTIONS = ['simple', 'sgm_uniform'] as const
export type SchedulerName = (typeof SCHEDULER_OPTIONS)[number]
export const DEFAULT_SCHEDULER: SchedulerName = 'simple'
