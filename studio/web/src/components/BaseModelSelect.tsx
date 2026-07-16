import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ModelsCatalog } from '../api/client'

/** 底模下拉的一个选项：value = 官方 variant key 或本地 custom 绝对路径。 */
export interface BaseModelOption {
  value: string
  label: string
  /** 官方 variant 的用途声明（krea2：raw=training / turbo=inference）；
   *  custom 权重无此元数据。页面可据此应用蒸馏推理默认参数。 */
  purpose?: 'training' | 'inference'
}

/** 支持底模选择的模型族。catalog section 键 = `${family}_main`。 */
export type BaseModelFamily = 'anima' | 'krea2'

interface FamilyMainSection {
  variants: Array<{
    variant: string
    exists: boolean
    /** krea2 起 variant 带用途声明（raw=training / turbo=inference）。 */
    purpose?: 'training' | 'inference'
  }>
  custom: Array<{ path: string; name: string; exists: boolean }>
  selected: string
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
        label: `${v.variant}${badge}`,
        purpose: v.purpose,
      })
    }
    for (const c of section.custom) {
      if (c.exists) out.push({ value: c.path, label: c.name })
    }
    return out
  }, [catalog, family, t])
  return {
    options,
    defaultValue: mainSection(catalog, family)?.selected ?? null,
    loaded: catalog !== null,
  }
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
  const { options, defaultValue } = useBaseModelOptions(family)
  // 有效值：显式覆盖优先，否则跟随设置页默认。
  const effective = value ?? defaultValue ?? ''
  // effective 不在 options 里（例如设置页选的 variant 还没下载）时补一项，
  // 避免 select 落到列表首项造成「显示的不是实际生效的」。
  const missing = effective !== '' && !options.some((o) => o.value === effective)
  return (
    <select
      className={className}
      style={style}
      value={effective}
      onChange={(e) => onChange(e.target.value)}
      aria-label={ariaLabel}
    >
      {missing && <option value={effective}>{basename(effective)}</option>}
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  )
}
