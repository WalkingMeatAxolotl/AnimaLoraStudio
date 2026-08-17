import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type BaseModelArch, type ModelsCatalog } from '../api/client'

/** 底模下拉的分组：官方 variant / 第三方内置条目 / 用户注册（本地或下载）。 */
export type BaseModelGroup = 'official' | 'community' | 'custom'

/** 底模下拉的一个选项：value = 官方 variant key 或本地 custom 绝对路径。 */
export interface BaseModelOption {
  value: string
  label: string
  group: BaseModelGroup
  /** 官方 variant 的用途声明（krea2：raw=training / turbo=inference）；
   *  custom 权重无此元数据。页面可据此应用蒸馏推理默认参数。 */
  purpose?: 'training' | 'inference'
  /** 底模架构（层数等，后端 header 探测）；未知为 null。层数决定 LoRA 能否互换。 */
  arch: BaseModelArch | null
}

/** 支持底模选择的模型族。catalog section 键 = `${family}_main`。 */
export type BaseModelFamily = 'anima' | 'krea2'

interface FamilyMainSection {
  variants: Array<{
    variant: string
    label?: string
    group?: 'official' | 'community'
    exists: boolean
    /** krea2 起 variant 带用途声明（raw=training / turbo=inference）。 */
    purpose?: 'training' | 'inference'
    arch?: BaseModelArch | null
  }>
  custom: Array<{ path: string; name: string; exists: boolean; arch?: BaseModelArch | null }>
  selected: string
}

/** 「N 层」后缀（arch 未知不显示）。层数是 LoRA 互换性的判据，所以进 label 而非 tooltip。 */
export function archSuffix(arch: BaseModelArch | null | undefined, t: (k: string, o?: Record<string, unknown>) => string): string {
  return arch?.num_blocks ? ` · ${t('baseModel.layers', { n: arch.num_blocks })}` : ''
}

function mainSection(
  catalog: ModelsCatalog | null, family: BaseModelFamily,
): FamilyMainSection | null {
  if (!catalog) return null
  const section = family === 'krea2' ? catalog.krea2_main : catalog.anima_main
  return (section as FamilyMainSection | undefined) ?? null
}

/** 从模型 catalog 拉「已下载的指定族主模型」列表 + 设置页当前选定值。
 *
 *  options 只含磁盘上存在的官方 variant + 注册的本地 custom（未下载的不出现，
 *  避免选了拉不到权重）；defaultValue = 设置页该族当前选中底模，作为下拉的
 *  初始 / 回退值。krea2 的 variant 带 purpose 徽标（raw=训练底模 /
 *  turbo=推理底模，两者都可选——A1 不加白名单）。 */
export function useBaseModelOptions(family: BaseModelFamily = 'anima'): {
  options: BaseModelOption[]
  defaultValue: string | null
  loaded: boolean
} {
  const { t } = useTranslation()
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  useEffect(() => {
    let alive = true
    api.getModelsCatalog().then((c) => { if (alive) setCatalog(c) }).catch(() => {})
    return () => { alive = false }
  }, [])
  const options = useMemo<BaseModelOption[]>(() => {
    const section = mainSection(catalog, family)
    if (!section) return []
    const out: BaseModelOption[] = []
    for (const v of section.variants) {
      if (!v.exists) continue
      const badge = v.purpose
        ? ` · ${t(`baseModel.purpose.${v.purpose}`)}`
        : ''
      out.push({
        value: v.variant,
        label: `${v.label ?? v.variant}${badge}${archSuffix(v.arch, t)}`,
        group: v.group ?? 'official',
        purpose: v.purpose,
        arch: v.arch ?? null,
      })
    }
    for (const c of section.custom) {
      if (c.exists) {
        out.push({
          value: c.path, label: `${c.name}${archSuffix(c.arch, t)}`,
          group: 'custom', arch: c.arch ?? null,
        })
      }
    }
    return out
  }, [catalog, family, t])
  return {
    options,
    defaultValue: mainSection(catalog, family)?.selected ?? null,
    loaded: catalog !== null,
  }
}

/** krea2 TE 选项状态：fp8 目录是否就绪（权重 + config 已下载，决定测试页
 *  下拉里 fp8 的可选性）+ 下载中心选中的默认 variant（下拉初值）。 */
export function useKrea2TeOptions(): { fp8Ready: boolean; selected: 'bf16' | 'fp8' } {
  const [state, setState] = useState<{ fp8Ready: boolean; selected: 'bf16' | 'fp8' }>({
    fp8Ready: false, selected: 'bf16',
  })
  useEffect(() => {
    let alive = true
    api.getModelsCatalog().then((c) => {
      if (!alive) return
      const files = c.krea2_text_encoder_fp8?.files ?? []
      const existing = new Set(files.filter((f) => f.exists).map((f) => f.name))
      const fp8Ready = (
        files.some((f) => f.name.endsWith('.safetensors') && f.exists)
        && existing.has('config.json')
        && existing.has('tokenizer.json')
      )
      const sel = c.krea2_text_encoder?.selected
      setState({ fp8Ready, selected: sel === 'fp8' ? 'fp8' : 'bf16' })
    }).catch(() => {})
    return () => { alive = false }
  }, [])
  return state
}

function basename(p: string): string {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  return i >= 0 ? p.slice(i + 1) : p
}

/** 底模下拉。受控：`value` 是「本次临时覆盖」（null = 跟随设置页默认）。
 *
 *  `family` 决定列哪个族的主模型（默认 anima，向后兼容既有调用方）。
 *  `className` 让各页面把 select 样式对齐自己页面里的其它 input
 *  （正则集用 "select input"，测试页用 "input text-xs w-full"）。 */
export default function BaseModelSelect({
  value, onChange, family = 'anima', className = 'select input', style, ariaLabel,
}: {
  value: string | null
  onChange: (v: string) => void
  family?: BaseModelFamily
  className?: string
  /** 内联样式透传（正则集页用它对齐训练配置页控件视觉）。 */
  style?: React.CSSProperties
  ariaLabel?: string
}) {
  const { t } = useTranslation()
  const { options, defaultValue } = useBaseModelOptions(family)
  // 有效值：显式覆盖优先，否则跟随设置页默认。
  const effective = value ?? defaultValue ?? ''
  // effective 不在 options 里（例如设置页选的 variant 还没下载）时补一项，
  // 避免 select 落到列表首项造成「显示的不是实际生效的」。
  const missing = effective !== '' && !options.some((o) => o.value === effective)
  // 分组：官方 / 第三方 / 自定义；只有一组时不渲染 optgroup（保持既有单列观感）
  const groups = (['official', 'community', 'custom'] as const)
    .map((g) => ({ key: g, items: options.filter((o) => o.group === g) }))
    .filter((g) => g.items.length > 0)
  const renderOptions = (items: BaseModelOption[]) => items.map((o) => (
    <option key={o.value} value={o.value}>{o.label}</option>
  ))
  return (
    <select
      className={className}
      style={style}
      value={effective}
      onChange={(e) => onChange(e.target.value)}
      aria-label={ariaLabel}
    >
      {missing && <option value={effective}>{basename(effective)}</option>}
      {groups.length <= 1
        ? renderOptions(options)
        : groups.map((g) => (
          <optgroup key={g.key} label={t(`modelPicker.group.${g.key}`)}>
            {renderOptions(g.items)}
          </optgroup>
        ))}
    </select>
  )
}
