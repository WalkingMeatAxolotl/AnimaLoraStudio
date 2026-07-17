import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { SchemaResponse, ConfigData } from '../api/client'
import { evalShowWhen, schemaAltDescription, schemaDisableHint, schemaDescription, schemaGroupLabel } from '../lib/schema'
import Field from './Field'
import RuleImpactDialog, { type RuleImpactChange } from './RuleImpactDialog'

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
  const props = schema.schema.properties
  const shouldDisableField = (prop: typeof props[string]) =>
    !!prop.disable_when && evalShowWhen(prop.disable_when, values)
  const takeoverValueForField = (prop: typeof props[string]) =>
    prop.disable_value ?? prop.default

  /** R6 确认弹窗的待决改动（非空时渲染 RuleImpactDialog）。 */
  const [pendingImpact, setPendingImpact] = useState<{
    trigger: { field: string; from: unknown; to: unknown }
    next: ConfigData
    writes: RuleImpactChange[]
  } | null>(null)

  /** 某次改动落地后规则会写哪些字段（有损清单：目标值 ≠ 当前值才列入）。
   * 来源：① disable_when takeover（reset 到 disable_value / default）；
   * ② advisory 改写 —— 切到 automagic 时 learning_rate 建议 1e-6
   *（upstream ostris/ai-toolkit + diffusion-pipe 默认；AdamW 量级 lr 起跑
   * 会让 sign-agreement 自适应慢 ~100× 才收敛）。 */
  const computeRuleWrites = (
    next: ConfigData, triggerField: string,
  ): RuleImpactChange[] => {
    const writes: RuleImpactChange[] = []
    for (const [name, prop] of Object.entries(props)) {
      if (!prop.disable_when || !evalShowWhen(prop.disable_when, next)) continue
      const target = takeoverValueForField(prop)
      if (target !== undefined && next[name] !== target) {
        writes.push({
          field: name, from: next[name], to: target,
          reason: schemaDisableHint(name, prop.disable_hint, t),
        })
      }
    }
    if (
      triggerField === 'optimizer_type' && next.optimizer_type === 'automagic'
      && Number(next.learning_rate) > 1e-5
    ) {
      writes.push({
        field: 'learning_rate', from: next.learning_rate, to: 1e-6,
        reason: t('ruleImpact.automagicLr'),
      })
    }
    return writes
  }

  // setField 入口拦截（R6，D6）：违反态不进表单 state —— 有损联动改值先弹
  // 确认（确认 = 触发改动 + 全部联动写值一次性提交；取消 = 什么都不发生），
  // 无损（联动目标本来就在钉值上）静默应用。model_family 不在任何 disable_when
  // 里出现，族切换仍由调用方（Train/Presets 的 onFormChange）拦去 FamilySwitchDialog。
  const setField = (name: string, v: unknown) => {
    const next = { ...values, [name]: v }
    const writes = computeRuleWrites(next, name)
    if (writes.length === 0) {
      onChange(next)
      return
    }
    setPendingImpact({
      trigger: { field: name, from: values[name], to: v },
      next,
      writes,
    })
  }

  // 兜底静默 takeover：外部写路径（config 载入 / disabledFields 变化）带进来的
  // 违反态直接修正，不弹窗（用户没有触发动作，弹窗无从「取消」）。正常交互
  // 路径经 setField 拦截后不会走到这里。老 config 同开的互斥由后端
  // _tolerant_validate 的 gate-first 修复 + defaulted_fields banner 处理。
  useEffect(() => {
    let nextValues = values
    let changed = false
    for (const [name, prop] of Object.entries(props)) {
      if (!shouldDisableField(prop)) continue
      const takeoverValue = takeoverValueForField(prop)
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
                  const conditionallyDisabled = shouldDisableField(prop)
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
                  // option_show_when：按当前 values 过滤下拉选项（多模型 P4-2）。
                  // 当前已选中的值即使被门控也保留——表单如实反映 config，
                  // 越族值由后端校验报错，不在 UI 里凭空消失。
                  const gates = prop.option_show_when
                  const enumOptions = gates
                    ? (prop.enum ?? []).filter(
                        (opt) =>
                          evalShowWhen(gates[String(opt)], values) ||
                          String(opt) === String(values[name] ?? '')
                      )
                    : undefined
                  // option_disable_when：命中的选项灰显不可选（D4：不隐藏，
                  // 用户能看见为什么不可选——title 显示 disable_hint）。
                  const dGates = prop.option_disable_when
                  const disabledEnumOptions = dGates
                    ? Object.keys(dGates).filter((opt) =>
                        evalShowWhen(dGates[opt], values)
                      )
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
                      suffix={suffixes[name]}
                      enumOptions={enumOptions}
                      disabledEnumOptions={disabledEnumOptions}
                      disabledOptionHint={schemaDisableHint(name, prop.disable_hint, t)}
                    />
                  )
                })}
              </div>
            )}
          </section>
        )
      })}
      {pendingImpact && (
        <RuleImpactDialog
          trigger={pendingImpact.trigger}
          changes={pendingImpact.writes}
          onApply={() => {
            const applied = { ...pendingImpact.next }
            for (const w of pendingImpact.writes) applied[w.field] = w.to
            setPendingImpact(null)
            onChange(applied)
          }}
          onCancel={() => setPendingImpact(null)}
        />
      )}
    </div>
  )
}
