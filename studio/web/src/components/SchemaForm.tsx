import { useEffect, useRef, useState } from 'react'
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

/** 计算当前 advancedMode 下哪些 group 至少有一个可见字段（用于侧栏锚点导航）。
 * 与下面的 buckets 逻辑保持一致：跳过 hidden / 跳过 advanced（简单模式下）。
 * 不考虑 show_when —— 那是 per-field 动态，section header 仍按 bucket 渲染。 */
export function visibleSchemaGroups(
  schema: SchemaResponse,
  advancedMode: boolean,
): Array<{ key: string; label: string }> {
  const counts = new Map<string, number>()
  for (const [, prop] of Object.entries(schema.schema.properties)) {
    if (prop.hidden) continue
    if (prop.advanced && !advancedMode) continue
    const g = prop.group ?? 'misc'
    counts.set(g, (counts.get(g) ?? 0) + 1)
  }
  return schema.groups
    .filter((g) => (counts.get(g.key) ?? 0) > 0)
    .map((g) => ({ key: g.key, label: g.label }))
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
  const isProdigyOptimizer = values.optimizer_type === 'prodigy' || values.optimizer_type === 'prodigy_plus_schedulefree'
  const shouldDisableField = (name: string, prop: typeof props[string]) => {
    if (name === 'lr_scheduler' && isProdigyOptimizer) return true
    return !!prop.disable_when && evalShowWhen(prop.disable_when, values)
  }
  const takeoverValueForField = (name: string, prop: typeof props[string]) => {
    if (name === 'lr_scheduler' && isProdigyOptimizer) return 'none'
    return prop.disable_value ?? prop.default
  }

  // disable_when 触发 → 字段灰显 + 值 reset 到 disable_value（缺省回到 default）。
  // 「先开的赢」语义：当 InfoNoise 与 loss_weighting / loss_type / schedule_shift /
  // noise_enhancement_type 任一互斥，先把状态切到非默认的那一侧赢，另一侧灰显且
  // reset 到 default。两侧都装 disable_when 形成对称锁。
  //
  // 老 config 同开（infonoise=on + 互斥字段非默认）由后端 _tolerant_validate 反向
  // 处理：关掉 infonoise 保留用户原投入的 weighting / huber / shift / enhancement，
  // 把 "infonoise_enabled" 写进 defaulted_fields，前端顶部 banner 提示。
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

  // Automagic 推荐 init lr=1e-6（upstream ostris/ai-toolkit + diffusion-pipe 默认）；
  // 用 AdamW 量级 lr (1e-4) 起跑会让 sign-agreement 自适应慢 ~100× 才收敛回工作区间。
  // 用户从其他 optimizer 切到 automagic 时，自动改写为 1e-6；不 disable，用户可继续调整。
  // 初次 mount（含加载 saved config）不触发，避免覆盖用户已保存值——saved config 路径由
  // training runtime 的 logger.warning 兜底。
  const prevOptimizerType = useRef(values.optimizer_type)
  useEffect(() => {
    const prev = prevOptimizerType.current
    const curr = values.optimizer_type
    if (prev !== curr) {
      prevOptimizerType.current = curr
      if (curr === 'automagic' && Number(values.learning_rate) > 1e-5) {
        onChange({ ...values, learning_rate: 1e-6 })
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values.optimizer_type])

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
            id={`schema-group-${key}`}
            className="rounded-md border border-subtle bg-surface scroll-mt-4"
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
