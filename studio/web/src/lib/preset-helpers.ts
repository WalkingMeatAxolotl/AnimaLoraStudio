// preset-helpers.ts —— Presets 页面 / Train 页面共享的预设相关工具。
// 拆出来避免 Train.tsx 加内联「新建预设」表单时把这几样复制一份。
import type { ConfigData, SchemaResponse } from '../api/client'

/** 预设名合法字符：字母 / 数字 / _ / -。/api/presets/<name> 路由对名字
 * 也按这个集合校验。 */
export const PRESET_NAME_RE = /^[A-Za-z0-9_\-]+$/

const DESC_KEY = 'studio.preset.descriptions'

/** 预设副标题（"描述"）走 localStorage 按 name 索引存。后端 schema 里 preset
 * 没有 description 字段；这是纯前端展示用的辅助文案。 */
export function loadPresetDescriptions(): Record<string, string> {
  try {
    const raw = localStorage.getItem(DESC_KEY)
    return raw ? (JSON.parse(raw) as Record<string, string>) : {}
  } catch {
    return {}
  }
}

export function savePresetDescriptions(d: Record<string, string>) {
  try {
    localStorage.setItem(DESC_KEY, JSON.stringify(d))
  } catch {
    /* ignore quota errors */
  }
}

/** 从 schema 抽默认值字典。新建预设时表单的初始内容。 */
export function defaultsFromSchema(schema: SchemaResponse | null): ConfigData {
  if (!schema) return {}
  const out: ConfigData = {}
  for (const [name, prop] of Object.entries(schema.schema.properties)) {
    if (prop.default !== undefined) out[name] = prop.default
  }
  return out
}

/** `base` 不撞 `existing` 任何一个名字时直接返回 `base`；否则尝试 `base_1`、
 * `base_2`…… 找到第一个未被占用的返回。
 *
 * 用在「项目页一键新建预设」自动命名上 —— base 形如 `<project_slug>_<version_label>`，
 * 重名（用户之前为同 version 创建过）时加下划线后缀，PRESET_NAME_RE 兼容。
 * 上限 999 兜底；999 个同名场景不现实，超过返回带 timestamp 后缀避免死循环。
 */
export function generateUniquePresetName(
  base: string,
  existing: ReadonlyArray<{ name: string }>,
): string {
  const taken = new Set(existing.map((p) => p.name))
  if (!taken.has(base)) return base
  for (let i = 1; i < 1000; i++) {
    const candidate = `${base}_${i}`
    if (!taken.has(candidate)) return candidate
  }
  return `${base}_${Date.now()}`
}
