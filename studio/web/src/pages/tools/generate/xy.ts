/** XY 模式本地 state + values 解析/校验工具。
 *
 * 后端 schema 要求 values 类型按 axis 派生（int / float / string）；前端
 * 把用户输入的逗号字符串解析成正确类型，并在解析失败时给出错误信息。 */

import type { LoraEntry, XYAxisSpec, XYAxisType, XYMatrixSpec } from '../../../api/client'
import i18n from '../../../i18n'

/** UI 侧 axis 状态：raw 是用户输入的逗号字符串（不实时解析便于编辑）。 */
export interface XYAxisDraft {
  axis: XYAxisType
  raw: string
  loraIndex: number | null
}

/** XYAxisSpec 配套的字段类型映射（与 schema._check_axis_values 同源）。 */
export const AXIS_VALUE_TYPE: Record<XYAxisType, 'int' | 'float' | 'string'> = {
  steps: 'int',
  cfg_scale: 'float',
  lora_scale: 'float',
  lora_ckpt: 'string',  // ckpt 路径
}

export const AXIS_LABEL_KEYS: Record<XYAxisType, string> = {
  steps: 'generate.axisSteps',
  cfg_scale: 'generate.axisCfgScale',
  lora_scale: 'generate.axisLoraScale',
  lora_ckpt: 'generate.axisLora',
}

export function axisLabel(axis: XYAxisType): string {
  return i18n.t(AXIS_LABEL_KEYS[axis])
}

/** 仅 lora_ckpt 需要 loraIndex（指 cell 内 mutate 哪条 LoRA 的 path）。
 *  lora_scale 改成全局轴（所有 LoRA 共用 cell 值），不再绑特定 LoRA。 */
export const REQUIRES_LORA_INDEX: Set<XYAxisType> = new Set(['lora_ckpt'])

/** 解析逗号分隔的 raw 字符串成 axis values。失败抛 string error。 */
export function parseAxisValues(axis: XYAxisType, raw: string): Array<number | string> {
  const parts = raw.split(',').map((s) => s.trim()).filter((s) => s.length > 0)
  if (parts.length === 0) {
    throw i18n.t('generate.axisValueRequired', { axis: axisLabel(axis) })
  }
  const t = AXIS_VALUE_TYPE[axis]
  if (t === 'string') {
    return parts
  }
  const out: number[] = []
  for (const p of parts) {
    const n = Number(p)
    if (!Number.isFinite(n)) {
      throw i18n.t('generate.axisValueInvalidNumber', { axis: axisLabel(axis), value: p })
    }
    if (t === 'int' && !Number.isInteger(n)) {
      throw i18n.t('generate.axisValueMustBeInteger', { axis: axisLabel(axis), value: p })
    }
    out.push(n)
  }
  return out
}

/** 把 draft 转成 XYAxisSpec —— schema 校验前的客户端 sanity check。 */
export function draftToSpec(
  draft: XYAxisDraft,
  loras: LoraEntry[],
): XYAxisSpec {
  const values = parseAxisValues(draft.axis, draft.raw)
  const spec: XYAxisSpec = { axis: draft.axis, values }
  if (REQUIRES_LORA_INDEX.has(draft.axis)) {
    if (draft.loraIndex === null) {
      throw i18n.t('generate.axisRequiresLora', { axis: axisLabel(draft.axis) })
    }
    if (draft.loraIndex >= loras.length) {
      throw i18n.t('generate.axisLoraMissing', { axis: axisLabel(draft.axis), n: draft.loraIndex + 1 })
    }
    spec.lora_index = draft.loraIndex
  }
  return spec
}

/** XY 提交时构建 xy_matrix + 实际要发的 lora_configs。
 *
 * **为什么不能直接整桶发 loras**：XY 模式下唯一能往 loras（xyLoras）加 anchor
 * 的入口是 lora_ckpt 轴 picker，且它只 push 不 prune（SidebarXYAxes.commitPicks）
 * —— 切项目/版本、切轴类型、删 Y 轴都会把旧 anchor 留下成「孤儿」。后端把
 * lora_configs 全部当 base LoRA 叠到每个 cell（anima_daemon._run_xy），孤儿就会
 * 混进每张图（「选 chenbin V3.4 却带上没选过的 v3.2 / 跨 mode 的 hoshi」的根因）。
 *
 * 这里只保留被当前 X/Y 轴 loraIndex 引用的 anchor，按出现顺序重映射索引，孤儿
 * 一律丢弃。非 lora_ckpt 轴不贡献任何 anchor → lora_configs 为空。
 * 校验失败（轴值缺失 / loraIndex 越界）沿用 draftToSpec 抛 string error。 */
export function buildXYMatrix(
  xDraft: XYAxisDraft,
  yDraft: XYAxisDraft | null,
  loras: LoraEntry[],
): { xy_matrix: XYMatrixSpec; loraConfigs: LoraEntry[] } {
  const remap = new Map<number, number>()
  const loraConfigs: LoraEntry[] = []
  const remapDraft = (d: XYAxisDraft): XYAxisDraft => {
    if (d.axis !== 'lora_ckpt' || d.loraIndex == null) return d
    const entry = loras[d.loraIndex]
    if (!entry || !entry.path.trim()) return d // draftToSpec 会抛 axisLoraMissing
    if (!remap.has(d.loraIndex)) {
      remap.set(d.loraIndex, loraConfigs.length)
      loraConfigs.push(entry)
    }
    return { ...d, loraIndex: remap.get(d.loraIndex) ?? null }
  }
  const x = draftToSpec(remapDraft(xDraft), loraConfigs)
  const y = yDraft ? draftToSpec(remapDraft(yDraft), loraConfigs) : null
  return { xy_matrix: { x, y }, loraConfigs }
}

/** 计算 cell 总数（y=null 时退化成 1×N）。 */
export function cellCount(xLen: number, yLen: number | null): number {
  return xLen * (yLen ?? 1)
}

/** path → 不带目录前缀和 .safetensors 后缀的"短名"（XY 标头 / LoRA 卡片用）。 */
export function ckptStemFromPath(path: string): string {
  const filename = path.split(/[\\/]/).pop() ?? path
  return filename.replace(/\.safetensors$/i, '')
}

/** 如果 axis 是 lora_ckpt（值是 path），用 stem 显示；其他类型原样返回。 */
export function formatAxisValue(axis: XYAxisType, value: string): string {
  if (axis === 'lora_ckpt') return ckptStemFromPath(value)
  return value
}
