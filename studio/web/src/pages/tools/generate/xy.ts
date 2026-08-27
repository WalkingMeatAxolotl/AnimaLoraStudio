import type { LoraEntry, XYAxisSpec, XYAxisType, XYMatrixSpec } from '../../../api/client'
import i18n from '../../../i18n'

export interface XYAxisDraft {
  axis: XYAxisType
  raw: string
  /** 旧版持久化字段；新 UI 不把它当作长期身份。 */
  loraIndex?: number | null
  /** lora_ckpt 轴实际加载的动态 LoRA，运行时生成 lora_index。 */
  checkpointAnchor?: LoraEntry | null
}

export const AXIS_VALUE_TYPE: Record<XYAxisType, 'int' | 'float' | 'string'> = {
  steps: 'int', cfg_scale: 'float', lora_scale: 'float', lora_ckpt: 'string',
}
export const AXIS_LABEL_KEYS: Record<XYAxisType, string> = {
  steps: 'generate.axisSteps', cfg_scale: 'generate.axisCfgScale',
  lora_scale: 'generate.axisLoraScale', lora_ckpt: 'generate.axisLora',
}
export function axisLabel(axis: XYAxisType): string { return i18n.t(AXIS_LABEL_KEYS[axis]) }
export const REQUIRES_LORA_INDEX: Set<XYAxisType> = new Set(['lora_ckpt'])

export function splitAxisRaw(raw: string): string[] {
  return raw.split(/[,，]+/).map((value) => value.trim()).filter(Boolean)
}

export function parseAxisValues(axis: XYAxisType, raw: string): Array<number | string> {
  const parts = splitAxisRaw(raw)
  if (!parts.length) throw i18n.t('generate.axisValueRequired', { axis: axisLabel(axis) })
  if (AXIS_VALUE_TYPE[axis] === 'string') return parts
  const out: number[] = []
  for (const p of parts) {
    const n = Number(p)
    if (!Number.isFinite(n)) throw i18n.t('generate.axisValueInvalidNumber', { axis: axisLabel(axis), value: p })
    if (AXIS_VALUE_TYPE[axis] === 'int' && !Number.isInteger(n)) {
      throw i18n.t('generate.axisValueMustBeInteger', { axis: axisLabel(axis), value: p })
    }
    out.push(n)
  }
  return out
}

export function draftToSpec(draft: XYAxisDraft, loras: LoraEntry[]): XYAxisSpec {
  const values = parseAxisValues(draft.axis, draft.raw)
  const spec: XYAxisSpec = { axis: draft.axis, values }
  if (REQUIRES_LORA_INDEX.has(draft.axis)) {
    const index = draft.loraIndex
    if (index == null) throw i18n.t('generate.axisRequiresLora', { axis: axisLabel(draft.axis) })
    if (index < 0 || index >= loras.length || !loras[index]?.path.trim()) {
      throw i18n.t('generate.axisLoraMissing', { axis: axisLabel(draft.axis), n: index + 1 })
    }
    spec.lora_index = index
  }
  return spec
}

function normalizedPath(path: string): string { return path.replace(/\\/g, '/').toLocaleLowerCase() }

/** 构建 wire contract。第四个参数是新 UI 的固定 LoRA；第三个参数保留旧调用兼容。 */
export function buildXYMatrix(
  xDraft: XYAxisDraft, yDraft: XYAxisDraft | null,
  legacyLoras: LoraEntry[] = [], fixedLoras?: LoraEntry[],
  fixedUi?: Array<{ enabled: boolean }>,
): { xy_matrix: XYMatrixSpec; loraConfigs: LoraEntry[] } {
  if (yDraft && xDraft.axis === yDraft.axis) {
    throw i18n.t('generate.axisDuplicateType')
  }
  if (fixedLoras === undefined) {
    const remap = new Map<number, number>()
    const loraConfigs: LoraEntry[] = []
    const remapDraft = (draft: XYAxisDraft): XYAxisDraft => {
      if (draft.axis !== 'lora_ckpt' || draft.loraIndex == null) return draft
      const entry = legacyLoras[draft.loraIndex]
      if (!entry || !entry.path.trim()) return draft
      if (!remap.has(draft.loraIndex)) {
        remap.set(draft.loraIndex, loraConfigs.length)
        loraConfigs.push(entry)
      }
      return { ...draft, loraIndex: remap.get(draft.loraIndex) ?? null }
    }
    const x = draftToSpec(remapDraft(xDraft), loraConfigs)
    const y = yDraft ? draftToSpec(remapDraft(yDraft), loraConfigs) : null
    return { xy_matrix: { x, y }, loraConfigs }
  }

  const base = fixedLoras.filter((entry, i) => entry.path.trim() && fixedUi?.[i]?.enabled !== false).map((entry) => ({ ...entry }))
  const configs = [...base]
  const anchorIndices = new Map<string, number>()

  for (const draft of [xDraft, yDraft].filter((d): d is XYAxisDraft => Boolean(d))) {
    if (draft.axis !== 'lora_ckpt') continue
    const entry = draft.checkpointAnchor
      ?? (draft.loraIndex != null ? legacyLoras[draft.loraIndex] : undefined)
    if (!entry?.path.trim()) {
      // 走旧 API 时保留原始错误文案；新 API 也拒绝没有动态 LoRA 的 checkpoint 轴。
      throw i18n.t('generate.axisRequiresLora', { axis: axisLabel(draft.axis) })
    }
    if (configs.some((item) => normalizedPath(item.path) === normalizedPath(entry.path))) {
      throw i18n.t('generate.axisDuplicateLora')
    }
    configs.push({ ...entry })
    anchorIndices.set(normalizedPath(entry.path), configs.length - 1)
  }
  const remap = (draft: XYAxisDraft): XYAxisDraft => {
    if (draft.axis !== 'lora_ckpt') return draft
    const entry = draft.checkpointAnchor
      ?? (draft.loraIndex != null ? legacyLoras[draft.loraIndex] : undefined)
    const idx = entry ? anchorIndices.get(normalizedPath(entry.path)) : undefined
    return { ...draft, loraIndex: idx ?? draft.loraIndex }
  }
  const x = draftToSpec(remap(xDraft), configs)
  const y = yDraft ? draftToSpec(remap(yDraft), configs) : null
  if ((x.axis === 'lora_scale' || y?.axis === 'lora_scale') && configs.length === 0) {
    throw i18n.t('generate.axisLoraScaleRequiresLora')
  }
  // legacy tests and old callers intentionally keep their previous orphan-filter behavior.
  return {
    xy_matrix: { x, y },
    loraConfigs: configs,
  }
}

export function cellCount(xLen: number, yLen: number | null): number { return xLen * (yLen ?? 1) }
export function ckptStemFromPath(path: string): string {
  const filename = path.split(/[\\/]/).pop() ?? path
  return filename.replace(/\.safetensors$/i, '')
}
export function formatAxisValue(axis: XYAxisType, value: string): string {
  return axis === 'lora_ckpt' ? ckptStemFromPath(value) : value
}
export interface XYAxisView { label: string; values: string[]; format?: (value: string) => string; title?: (value: string) => string }
export function axisView(draft: XYAxisDraft): XYAxisView {
  return { label: axisLabel(draft.axis), values: splitAxisRaw(draft.raw), format: (v) => formatAxisValue(draft.axis, v) }
}
export function axisText(axis: XYAxisView, value: string | null): string { return value == null ? '' : axis.format ? axis.format(value) : value }
export function axisTitle(axis: XYAxisView, value: string | null): string { return value == null ? '' : axis.title ? axis.title(value) : value }
