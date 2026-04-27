// schema.ts —— 把 FastAPI 返回的 JSON Schema 解释成前端表单需要的形态。
import type { SchemaProperty } from '../api/client'

export type ControlKind =
  | 'bool'
  | 'select'
  | 'int'
  | 'float'
  | 'string'
  | 'path'
  | 'textarea'
  | 'string-list'

/**
 * 推断字段的控件类型。优先用 schema 里 control 自定义元字段，否则按
 * JSON Schema 的 type / enum / anyOf 推断。
 */
export function controlKind(prop: SchemaProperty): ControlKind {
  if (prop.control && prop.control !== 'auto') {
    if (
      prop.control === 'path' ||
      prop.control === 'textarea' ||
      prop.control === 'string-list'
    )
      return prop.control
  }

  if (prop.enum && prop.enum.length > 0) return 'select'

  // 解开 anyOf: [X, null] 的可空类型
  let type = prop.type
  if (!type && prop.anyOf) {
    const nonNull = prop.anyOf.find((a) => a.type && a.type !== 'null')
    type = nonNull?.type
  }

  if (type === 'boolean') return 'bool'
  if (type === 'integer') return 'int'
  if (type === 'number') return 'float'
  if (type === 'array') return 'string-list'
  return 'string'
}

/**
 * show_when 简单解析器：支持 `key==value` / `key!=value`。
 * 复杂表达式以后再加。
 */
export function evalShowWhen(
  expr: string | undefined,
  values: Record<string, unknown>
): boolean {
  if (!expr) return true
  const eq = expr.split('==')
  if (eq.length === 2) {
    return String(values[eq[0].trim()]) === eq[1].trim()
  }
  const ne = expr.split('!=')
  if (ne.length === 2) {
    return String(values[ne[0].trim()]) !== ne[1].trim()
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
