/** XY 模式本地 state + values 解析/校验工具。
 *
 * 后端 schema 要求 values 类型按 axis 派生（int / float / string）；前端
 * 把用户输入的逗号字符串解析成正确类型，并在解析失败时给出错误信息。 */

import type { LoraEntry, XYAxisSpec, XYAxisType } from '../../../api/client'

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

export const AXIS_LABELS: Record<XYAxisType, string> = {
  steps: '步数',
  cfg_scale: 'CFG Scale',
  lora_scale: '权重',
  lora_ckpt: 'LoRA',
}

export const REQUIRES_LORA_INDEX: Set<XYAxisType> = new Set(['lora_scale', 'lora_ckpt'])

/** 解析逗号分隔的 raw 字符串成 axis values。失败抛 string error。 */
export function parseAxisValues(axis: XYAxisType, raw: string): Array<number | string> {
  const parts = raw.split(',').map((s) => s.trim()).filter((s) => s.length > 0)
  if (parts.length === 0) {
    throw `${AXIS_LABELS[axis]} 的值不能为空`
  }
  const t = AXIS_VALUE_TYPE[axis]
  if (t === 'string') {
    return parts
  }
  const out: number[] = []
  for (const p of parts) {
    const n = Number(p)
    if (!Number.isFinite(n)) {
      throw `${AXIS_LABELS[axis]} 的值「${p}」不是合法数字`
    }
    if (t === 'int' && !Number.isInteger(n)) {
      throw `${AXIS_LABELS[axis]} 必须是整数（收到 ${p}）`
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
      throw `${AXIS_LABELS[draft.axis]} 必须绑定一个 LoRA`
    }
    if (draft.loraIndex >= loras.length) {
      throw `${AXIS_LABELS[draft.axis]} 绑定的 LoRA #${draft.loraIndex + 1} 不存在`
    }
    spec.lora_index = draft.loraIndex
  }
  return spec
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
