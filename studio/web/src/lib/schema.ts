// schema.ts —— 把 FastAPI 返回的 JSON Schema 解释成前端表单需要的形态。
import type { TFunction } from 'i18next'
import type { SchemaProperty } from '../api/client'

export type ControlKind =
  | 'bool'
  | 'select'
  | 'tristate'
  | 'int'
  | 'float'
  | 'string'
  | 'path'
  | 'textarea'
  | 'code'
  | 'string-list'
  | 'int-list'

/**
 * 推断字段的控件类型。优先用 schema 里 control 自定义元字段，否则按
 * JSON Schema 的 type / enum / anyOf 推断。
 */
export function controlKind(prop: SchemaProperty): ControlKind {
  if (prop.control && prop.control !== 'auto') {
    if (
      prop.control === 'path' ||
      prop.control === 'textarea' ||
      prop.control === 'code' ||
      prop.control === 'string-list' ||
      prop.control === 'tristate'
    )
      return prop.control
  }

  if (prop.enum && prop.enum.length > 0) return 'select'

  // 解开 anyOf: [X, null] 的可空类型
  let type = prop.type
  if (!type && prop.anyOf) {
    const hasNull = prop.anyOf.some((a) => a.type === 'null')
    const nonNull = prop.anyOf.find((a) => a.type && a.type !== 'null')
    if (hasNull && nonNull?.type === 'boolean') return 'tristate'
    type = nonNull?.type
  }

  if (type === 'boolean') return 'bool'
  if (type === 'integer') return 'int'
  if (type === 'number') return 'float'
  if (type === 'array') {
    const itemType = prop.items?.type
    return itemType === 'integer' || itemType === 'number' ? 'int-list' : 'string-list'
  }
  return 'string'
}

/**
 * show_when / disable_when 表达式解析器。
 * 支持：`key==value` / `key!=value` / `key>value` / `key>=value` / `key<value` / `key<=value`
 * 组合：`||`（任一为真）/ `&&`（全部为真）。
 */
export function evalShowWhen(
  expr: string | undefined,
  values: Record<string, unknown>
): boolean {
  if (!expr) return true
  const branches = expr.split('||').map((part) => part.trim()).filter(Boolean)
  if (branches.length > 1) {
    return branches.some((branch) => evalShowWhen(branch, values))
  }
  const ands = expr.split('&&').map((part) => part.trim()).filter(Boolean)
  if (ands.length > 1) {
    return ands.every((clause) => evalShowWhen(clause, values))
  }
  const eq = expr.split('==')
  if (eq.length === 2) {
    return String(values[eq[0].trim()]) === eq[1].trim()
  }
  const ne = expr.split('!=')
  if (ne.length === 2) {
    return String(values[ne[0].trim()]) !== ne[1].trim()
  }
  const cmp = expr.match(/^(.+?)(>=|<=|>|<)(.+)$/)
  if (cmp) {
    const lhs = Number(values[cmp[1].trim()])
    const rhs = Number(cmp[3].trim())
    if (Number.isNaN(lhs) || Number.isNaN(rhs)) return false
    switch (cmp[2]) {
      case '>': return lhs > rhs
      case '>=': return lhs >= rhs
      case '<': return lhs < rhs
      case '<=': return lhs <= rhs
    }
  }
  return true
}

/** 字段的人类可读 label：首字母大写 + 下划线变空格。 */
export function fieldLabel(name: string): string {
  return name
    .split('_')
    .map((w) => (w.length > 0 ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ')
}

export const SCHEMA_GROUP_LABEL_KEYS: Record<string, string> = {
  model: 'schema.groups.model',
  dataset: 'schema.groups.dataset',
  caption: 'schema.groups.caption',
  lora: 'schema.groups.lora',
  training: 'schema.groups.training',
  noise_augmentation: 'schema.groups.noiseAugmentation',
  timestep_sampling: 'schema.groups.timestepSampling',
  loss: 'schema.groups.loss',
  system: 'schema.groups.system',
  output: 'schema.groups.output',
  sample: 'schema.groups.sample',
  eval_validation: 'schema.groups.evalValidation',
  monitor: 'schema.groups.monitor',
  wandb: 'schema.groups.wandb',
}

export const SCHEMA_ENUM_LABEL_KEYS: Record<string, Record<string, string>> = {
  lora_type: {
    lora: 'schema.enums.loraType.lora',
    lokr: 'schema.enums.loraType.lokr',
    loha: 'schema.enums.loraType.loha',
    ortho: 'schema.enums.loraType.ortho',
    tlora: 'schema.enums.loraType.tlora',
  },
  lr_scheduler: {
    none: 'schema.enums.lrScheduler.none',
    cosine: 'schema.enums.lrScheduler.cosine',
    cosine_with_restart: 'schema.enums.lrScheduler.cosineWithRestart',
    cosine_with_warmup: 'schema.enums.lrScheduler.cosineWithWarmup',
  },
  optimizer_type: {
    adamw: 'schema.enums.optimizerType.adamw',
    automagic: 'schema.enums.optimizerType.automagic',
    lion: 'schema.enums.optimizerType.lion',
    prodigy: 'schema.enums.optimizerType.prodigy',
    prodigy_plus_schedulefree: 'schema.enums.optimizerType.prodigyPlusSchedulefree',
  },
  timestep_sampling: {
    logit_normal: 'schema.enums.timestepSampling.logitNormal',
    uniform: 'schema.enums.timestepSampling.uniform',
    logit_normal_low: 'schema.enums.timestepSampling.logitNormalLow',
    mode: 'schema.enums.timestepSampling.mode',
  },
  loss_weighting: {
    none: 'schema.enums.lossWeighting.none',
    min_snr: 'schema.enums.lossWeighting.minSnr',
    detail_inv_t: 'schema.enums.lossWeighting.detailInvT',
    cosmap: 'schema.enums.lossWeighting.cosmap',
  },
  leap_variant: {
    original: 'schema.enums.leapVariant.original',
    sparse: 'schema.enums.leapVariant.sparse',
    bridge: 'schema.enums.leapVariant.bridge',
    lagrange: 'schema.enums.leapVariant.lagrange',
  },
  mixed_precision: {
    bf16: 'schema.enums.mixedPrecision.bf16',
    fp16: 'schema.enums.mixedPrecision.fp16',
    no: 'schema.enums.mixedPrecision.no',
  },
  attention_backend: {
    none: 'schema.enums.attentionBackend.none',
    xformers: 'schema.enums.attentionBackend.xformers',
    flash_attn: 'schema.enums.attentionBackend.flashAttn',
  },
  noise_enhancement_type: {
    none: 'schema.enums.noiseEnhancementType.none',
    offset: 'schema.enums.noiseEnhancementType.offset',
    pyramid: 'schema.enums.noiseEnhancementType.pyramid',
  },
  sample_sampler_name: {
    er_sde: 'schema.enums.sampler.erSde',
    dpmpp_3m_sde: 'schema.enums.sampler.dpmpp3mSde',
  },
  sample_scheduler: {
    simple: 'schema.enums.scheduler.simple',
    sgm_uniform: 'schema.enums.scheduler.sgmUniform',
  },
  wandb_mode: {
    '': 'field.useGlobal',
    online: 'schema.enums.wandbMode.online',
    offline: 'schema.enums.wandbMode.offline',
    disabled: 'schema.enums.wandbMode.disabled',
  },
  wandb_upload_model_policy: {
    '': 'field.useGlobal',
    all: 'schema.enums.wandbPolicy.all',
    last: 'schema.enums.wandbPolicy.last',
  },
  wandb_upload_state_manual_policy: {
    '': 'field.useGlobal',
    all: 'schema.enums.wandbPolicy.all',
    last: 'schema.enums.wandbPolicy.last',
  },
  wandb_upload_state_auto_policy: {
    '': 'field.useGlobal',
    all: 'schema.enums.wandbPolicy.all',
    last: 'schema.enums.wandbPolicy.last',
  }
}

export function schemaGroupLabel(key: string, fallback: string, t: TFunction): string {
  const labelKey = SCHEMA_GROUP_LABEL_KEYS[key]
  return labelKey ? t(labelKey) : fallback
}

export function schemaEnumLabel(fieldName: string, value: unknown, t: TFunction): string {
  const raw = String(value)
  const labelKey = SCHEMA_ENUM_LABEL_KEYS[fieldName]?.[raw]
  return labelKey ? t(labelKey) : raw
}

export function schemaDescription(name: string, fallback: string | undefined, t: TFunction): string | undefined {
  const translated = t(`schema.descriptions.${name}`, { defaultValue: '' })
  return translated || fallback
}

export function schemaAltDescription(name: string, fallback: string | undefined, t: TFunction): string | undefined {
  const translated = t(`schema.altDescriptions.${name}`, { defaultValue: '' })
  return translated || fallback
}

export function schemaDisableHint(name: string, fallback: string | undefined, t: TFunction): string | undefined {
  const translated = t(`schema.disableHints.${name}`, { defaultValue: '' })
  return translated || fallback
}
