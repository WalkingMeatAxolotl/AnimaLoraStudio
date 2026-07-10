import { useEffect, useMemo, useState } from 'react'
import { api, type ModelsCatalog } from '../api/client'

/** 底模下拉的一个选项：value = 官方 variant key 或本地 custom 绝对路径。 */
export interface BaseModelOption {
  value: string
  label: string
}

/** 从模型 catalog 拉「已下载的 Anima 主模型」列表 + 设置页当前选定值。
 *
 *  options 只含磁盘上存在的官方 variant + 注册的本地 custom（未下载的不出现，
 *  避免选了拉不到权重）；defaultValue = secrets.models.selected_anima（即设置页
 *  当前底模），作为下拉的初始 / 回退值。 */
export function useBaseModelOptions(): {
  options: BaseModelOption[]
  defaultValue: string | null
  loaded: boolean
} {
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  useEffect(() => {
    let alive = true
    api.getModelsCatalog().then((c) => { if (alive) setCatalog(c) }).catch(() => {})
    return () => { alive = false }
  }, [])
  const options = useMemo<BaseModelOption[]>(() => {
    const am = catalog?.anima_main
    if (!am) return []
    const out: BaseModelOption[] = []
    for (const v of am.variants) {
      if (v.exists) out.push({ value: v.variant, label: v.variant })
    }
    for (const c of am.custom) {
      if (c.exists) out.push({ value: c.path, label: c.name })
    }
    return out
  }, [catalog])
  return {
    options,
    defaultValue: catalog?.anima_main.selected ?? null,
    loaded: catalog !== null,
  }
}

function basename(p: string): string {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  return i >= 0 ? p.slice(i + 1) : p
}

/** 底模下拉。受控：`value` 是「本次临时覆盖」（null = 跟随设置页默认）。
 *
 *  `className` 让各页面把 select 样式对齐自己页面里的其它 input
 *  （正则集用 "select input"，测试页用 "input text-xs w-full"）。 */
export default function BaseModelSelect({
  value, onChange, className = 'select input', style, ariaLabel,
}: {
  value: string | null
  onChange: (v: string) => void
  className?: string
  /** 内联样式透传（正则集页用它对齐训练配置页控件视觉）。 */
  style?: React.CSSProperties
  ariaLabel?: string
}) {
  const { options, defaultValue } = useBaseModelOptions()
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
