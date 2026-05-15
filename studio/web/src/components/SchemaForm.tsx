import { useEffect, useState } from 'react'
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
  /** 字段不 disabled 但要挂个徽章（如「自动 · 项目设置」表示项目预填了，
   * 但仍允许用户修改）。优先级：disabledHints > autoHints。 */
  autoHints?: Record<string, string>
  /** false（默认）= 简单模式，隐藏 advanced=true 的字段。 */
  advancedMode?: boolean
}

/**
 * 按 schema.groups 分区渲染表单；分组可折叠。
 * show_when 用 evalShowWhen 做条件显示，依赖当前 values。
 */
export default function SchemaForm({
  schema, values, onChange, disabledFields, disabledHints, autoHints, advancedMode = false,
}: Props) {
  const disabledSet = new Set(disabledFields ?? [])
  const dHints = disabledHints ?? {}
  const aHints = autoHints ?? {}
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

  // disable_when 触发时把字段值强制回到 default。避免「切换 optimizer 到
  // prodigy_plus_schedulefree 之后 lr_scheduler 还停在 cosine，保存时被 pydantic
  // model_validator 拒绝」这种死锁 UX。
  useEffect(() => {
    let nextValues = values
    let changed = false
    for (const [name, prop] of Object.entries(props)) {
      if (!prop.disable_when) continue
      if (!evalShowWhen(prop.disable_when, values)) continue
      const def = prop.default
      if (def !== undefined && values[name] !== def) {
        nextValues = { ...nextValues, [name]: def }
        changed = true
      }
    }
    if (changed) onChange(nextValues)
    // 故意只监听 values；onChange / props 引用稳定，加进去会无限循环。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values])

  // 按 group 分桶。hidden=true 的字段直接跳过：值仍由 ConfigData 透传（PUT 时不丢），
  // 只是不在 UI 上渲染。如果一个组所有字段都 hidden，下面 `fields.length === 0`
  // 会让整个 section 自动消失。
  const buckets = new Map<string, string[]>()
  for (const [name, prop] of Object.entries(props)) {
    if (prop.hidden) continue
    if (prop.advanced && !advancedMode) continue
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
            className="rounded-md border border-subtle bg-surface"
          >
            <button
              type="button"
              onClick={() =>
                setCollapsed({ ...collapsed, [key]: !isCollapsed })
              }
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-fg-primary bg-transparent border-none cursor-pointer"
            >
              <span>{label}</span>
              <span className="text-fg-tertiary text-xs">
                {fields.length} 项 {isCollapsed ? '▸' : '▾'}
              </span>
            </button>
            {!isCollapsed && (
              <div className="px-4 pb-3 space-y-1">
                {fields.map((name) => {
                  const prop = props[name]
                  if (!evalShowWhen(prop.show_when, values)) return null
                  // disable_when（schema 驱动条件 disable，如 PPSF → lr_scheduler）
                  // 优先级低于全局 disabledFields（项目预填）。
                  const conditionallyDisabled =
                    !!prop.disable_when &&
                    evalShowWhen(prop.disable_when, values)
                  const isDisabled =
                    disabledSet.has(name) || conditionallyDisabled
                  const hint = disabledSet.has(name)
                    ? dHints[name]
                    : conditionallyDisabled
                      ? prop.disable_hint
                      : aHints[name]
                  const descriptionOverride =
                    prop.alt_description_when &&
                    prop.alt_description &&
                    evalShowWhen(prop.alt_description_when, values)
                      ? prop.alt_description
                      : undefined
                  return (
                    <Field
                      key={name}
                      name={name}
                      prop={prop}
                      value={values[name]}
                      onChange={(v) => setField(name, v)}
                      disabled={isDisabled}
                      hint={hint}
                      descriptionOverride={descriptionOverride}
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
