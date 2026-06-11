import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { SchemaResponse, ConfigData } from '../api/client'
import { evalShowWhen, schemaAltDescription, schemaDisableHint, schemaDescription, schemaGroupLabel } from '../lib/schema'
import Field from './Field'

interface Props {
  schema: SchemaResponse
  values: ConfigData
  onChange: (values: ConfigData) => void
  /** 这些字段名将以 readonly / disabled 渲染（项目特定 / 全局控制）。 */
  disabledFields?: string[]
  /** 每个 disabled 字段的徽章；缺省走 Field 默认「自动 · 项目控制」。
   * 支持 ReactNode 以便嵌入可点击链接（如跳到 Settings 对应区段）。 */
  disabledHints?: Record<string, React.ReactNode>
  /** 字段不 disabled 但要挂个徽章（如「自动 · 项目设置」表示项目预填了，
   * 但仍允许用户修改）。优先级：disabledHints > autoHints。 */
  autoHints?: Record<string, React.ReactNode>
  /** 字段右侧额外按钮槽（如「↺ 重置为全局默认」）。仅对 string/path 字段
   * 生效；按字段名查表。 */
  fieldSuffixes?: Record<string, React.ReactNode>
  /** false（默认）= 简单模式，隐藏 advanced=true 的字段。 */
  advancedMode?: boolean
}

/**
 * 按 schema.groups 分区渲染表单；分组可折叠。
 * show_when 用 evalShowWhen 做条件显示，依赖当前 values。
 */
export default function SchemaForm({
  schema, values, onChange, disabledFields, disabledHints, autoHints, fieldSuffixes, advancedMode = false,
}: Props) {
  const { t } = useTranslation()
  const disabledSet = new Set(disabledFields ?? [])
  const dHints = disabledHints ?? {}
  const aHints = autoHints ?? {}
  const suffixes = fieldSuffixes ?? {}
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
  const isAutoLrOptimizer = values.optimizer_type === 'prodigy' || values.optimizer_type === 'prodigy_plus_schedulefree' || values.optimizer_type === 'automagic'
  const shouldDisableField = (name: string, prop: typeof props[string]) => {
    if (name === 'lr_scheduler' && isAutoLrOptimizer) return true
    return !!prop.disable_when && evalShowWhen(prop.disable_when, values)
  }
  const takeoverValueForField = (name: string, prop: typeof props[string]) => {
    if (name === 'lr_scheduler' && isAutoLrOptimizer) return 'none'
    return prop.disable_value ?? prop.default
  }

  // disable_when 触发时把字段值强制回到 default。避免「切换 optimizer 到
  // prodigy_plus_schedulefree 之后 lr_scheduler 还停在 cosine，保存时被 pydantic
  // model_validator 拒绝」这种死锁 UX。
  useEffect(() => {
    let nextValues = values
    let changed = false
    for (const [name, prop] of Object.entries(props)) {
      if (!shouldDisableField(name, prop)) continue
      const takeoverValue = takeoverValueForField(name, prop)
      if (takeoverValue !== undefined && values[name] !== takeoverValue) {
        nextValues = { ...nextValues, [name]: takeoverValue }
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
        const groupLabel = schemaGroupLabel(key, label, t)
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
              <span>{groupLabel}</span>
              <span className="text-fg-tertiary text-xs">
                {t('schema.fieldCount', { n: fields.length })} {isCollapsed ? '▸' : '▾'}
              </span>
            </button>
            {!isCollapsed && (
              <div className="px-4 pb-3 space-y-1">
                {fields.map((name) => {
                  const prop = props[name]
                  if (!evalShowWhen(prop.show_when, values)) return null
                  // disable_when（schema 驱动条件 disable，如 Prodigy → lr_scheduler）
                  // 优先级低于全局 disabledFields（项目预填）。
                  const conditionallyDisabled = shouldDisableField(name, prop)
                  const isDisabled =
                    disabledSet.has(name) || conditionallyDisabled
                  const hint = disabledSet.has(name)
                    ? dHints[name]
                    : conditionallyDisabled
                      ? schemaDisableHint(name, prop.disable_hint, t)
                      : aHints[name]
                  const descriptionOverride =
                    prop.alt_description_when &&
                    evalShowWhen(prop.alt_description_when, values)
                      ? schemaAltDescription(name, prop.alt_description, t)
                      : schemaDescription(name, prop.description, t)
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
                      suffix={suffixes[name]}
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
