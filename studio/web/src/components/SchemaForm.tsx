import { useState } from 'react'
import type { SchemaResponse, ConfigData } from '../api/client'
import { evalShowWhen } from '../lib/schema'
import Field from './Field'

interface Props {
  schema: SchemaResponse
  values: ConfigData
  onChange: (values: ConfigData) => void
  /** 这些字段名将以 readonly / disabled 渲染（项目特定 / 全局控制）。 */
  disabledFields?: string[]
  /** 每个 disabled 字段的徽章文字；缺省走 Field 默认「自动 · 项目控制」。 */
  disabledHints?: Record<string, string>
}

/**
 * 按 schema.groups 分区渲染表单；分组可折叠。
 * show_when 用 evalShowWhen 做条件显示，依赖当前 values。
 */
export default function SchemaForm({
  schema, values, onChange, disabledFields, disabledHints,
}: Props) {
  const disabledSet = new Set(disabledFields ?? [])
  const hints = disabledHints ?? {}
  // 用 schema.groups[].default_collapsed 决定初始折叠状态；用户手动改后保留状态。
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>(() => {
    const out: Record<string, boolean> = {}
    for (const g of schema.groups) {
      if (g.default_collapsed) out[g.key] = true
    }
    return out
  })
  const setField = (name: string, v: unknown) =>
    onChange({ ...values, [name]: v })

  const props = schema.schema.properties

  // 按 group 分桶
  const buckets = new Map<string, string[]>()
  for (const [name, prop] of Object.entries(props)) {
    const g = prop.group ?? 'misc'
    if (!buckets.has(g)) buckets.set(g, [])
    buckets.get(g)!.push(name)
  }

  return (
    <div className="space-y-3">
      {schema.groups.map(({ key, label }) => {
        const fields = buckets.get(key) ?? []
        if (fields.length === 0) return null
        const isCollapsed = collapsed[key]
        return (
          <section
            key={key}
            className="border border-slate-700 rounded-lg bg-slate-800/40"
          >
            <button
              type="button"
              onClick={() =>
                setCollapsed({ ...collapsed, [key]: !isCollapsed })
              }
              className="w-full flex items-center justify-between
                px-4 py-3 text-sm font-semibold text-slate-200
                hover:bg-slate-800/60 transition-colors"
            >
              <span>{label}</span>
              <span className="text-slate-500 text-xs">
                {fields.length} 项 {isCollapsed ? '▸' : '▾'}
              </span>
            </button>
            {!isCollapsed && (
              <div className="px-4 pb-3 space-y-1">
                {fields.map((name) => {
                  const prop = props[name]
                  if (!evalShowWhen(prop.show_when, values)) return null
                  return (
                    <Field
                      key={name}
                      name={name}
                      prop={prop}
                      value={values[name]}
                      onChange={(v) => setField(name, v)}
                      disabled={disabledSet.has(name)}
                      disabledHint={hints[name]}
                    />
                  )
                })}
              </div>
            )}
          </section>
        )
      })}
    </div>
  )
}
