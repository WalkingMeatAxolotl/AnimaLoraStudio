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
