import { useEffect, useRef, useState, type RefObject } from 'react'
import { useTranslation } from 'react-i18next'
import { InfoButton } from '../../../components/InfoButton'
import { MASK, textInputClass } from './constants'

// ── Section / Field ────────────────────────────────────────────────────────

export function SettingsSection({
  id, title, headerExtras, children,
}: {
  id?: string
  title: string
  headerExtras?: React.ReactNode  // 可选 slot：渲染在 h2 右侧（紧贴），给 ⓘ tooltip 之类用
  children: React.ReactNode
}) {
  const titleEl = <h2 className="text-sm font-semibold text-fg-primary">{title}</h2>
  return (
    <section id={id} className="rounded-md border border-subtle bg-surface p-4 flex flex-col gap-3 scroll-mt-24">
      {headerExtras ? (
        <div className="flex items-center gap-2 mb-0.5">
          {titleEl}
          {headerExtras}
        </div>
      ) : (
        <div className="mb-0.5">{titleEl}</div>
      )}
      {children}
    </section>
  )
}

/**
 * 右侧 sticky section 目录。基于 IntersectionObserver 在 scrollContainer 视口内
 * 跟踪当前可见 section，并提供点击平滑滚动。
 *
 * rootMargin 调整为顶部 -20%、底部 -70%：让"当前可见"判定集中在视口偏上区域，
 * 滚动时高亮跟随更自然（用户视线在 viewport 上 1/3 处）。
 */
export function SectionIndex({
  sections,
  scrollContainer,
}: {
  sections: { id: string; labelKey: string }[]
  scrollContainer: RefObject<HTMLDivElement>
}) {
  const { t } = useTranslation()
  const [active, setActive] = useState<string>(sections[0]?.id ?? '')

  useEffect(() => {
    // 切换 tab 后重置 active 到第一条
    setActive(sections[0]?.id ?? '')
  }, [sections])

  useEffect(() => {
    const root = scrollContainer.current
    if (!root || sections.length === 0) return
    // jsdom（vitest 环境）没有 IntersectionObserver；非浏览器环境直接跳过。
    if (typeof IntersectionObserver === 'undefined') return
    const observers: IntersectionObserver[] = []
    // 收集 (id, top) 用来在 onIntersect 时挑当前最靠上的可见 section
    const visible = new Set<string>()
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) visible.add(e.target.id)
          else visible.delete(e.target.id)
        }
        // 按 sections 顺序取第一个可见的作为 active
        const next = sections.find((s) => visible.has(s.id))
        if (next) setActive(next.id)
      },
      { root, rootMargin: '-20% 0px -70% 0px', threshold: 0 },
    )
    sections.forEach((s) => {
      const el = document.getElementById(s.id)
      if (el) obs.observe(el)
    })
    observers.push(obs)
    return () => observers.forEach((o) => o.disconnect())
  }, [sections, scrollContainer])

  const onJump = (id: string) => {
    const el = document.getElementById(id)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setActive(id)
  }

  return (
    <aside className="hidden lg:block">
      <nav className="sticky top-4 flex flex-col gap-0.5">
        <div className="caption mb-2 px-2">{t('settings.pageIndex')}</div>
        {sections.map((s) => (
          <button
            key={s.id}
            onClick={() => onJump(s.id)}
            className={`text-left text-xs px-2 py-1.5 rounded-sm transition-colors border-l-2 ${
              active === s.id
                ? 'border-accent text-accent bg-accent-soft/40'
                : 'border-transparent text-fg-tertiary hover:text-fg-secondary hover:bg-overlay/40'
            }`}
          >
            {t(s.labelKey)}
          </button>
        ))}
      </nav>
    </aside>
  )
}

export function SettingsField({ label, desc, helpTooltip, children }: {
  label: string
  desc?: string
  /** 可选 ⓘ tooltip slot，渲染在 label 旁边。中长说明（≥20 字 / 详细用法）
   *  适合放这里，避免 inline desc 把字段名行撑得过长。一般和 desc 二选一。 */
  helpTooltip?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="grid grid-cols-[240px_1fr] gap-3 items-start">
      <div className="flex flex-col gap-0.5 pt-1.5">
        <div className="flex items-center gap-2 min-w-0">
          <label className="text-xs text-fg-secondary font-mono leading-none">{label}</label>
          {helpTooltip && <InfoButton>{helpTooltip}</InfoButton>}
        </div>
        {desc && <p className="text-[10px] text-fg-tertiary m-0 leading-snug">{desc}</p>}
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  )
}

export function Bool({ value, onChange, disabled }: { value: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  const { t } = useTranslation()
  return (
    <select
      value={value ? 'on' : 'off'}
      onChange={(e) => onChange(e.target.value === 'on')}
      disabled={disabled}
      className={`${textInputClass} max-w-32 disabled:opacity-60`}
    >
      <option value="on">{t('settings.boolEnabled')}</option>
      <option value="off">{t('settings.boolDisabled')}</option>
    </select>
  )
}

export function SensitiveInput({ value, serverValue, onChange }: {
  value: string; serverValue: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const [localValue, setLocalValue] = useState(value)

  useEffect(() => {
    setLocalValue(value)
  }, [value])

  const masked = localValue === MASK

  const handleBlur = () => {
    if (localValue !== value) {
      onChange(localValue)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.currentTarget.blur()
    }
  }

  return (
    <input
      type="password"
      value={masked ? '' : localValue}
      placeholder={serverValue === MASK ? t('settings.sensitiveSavedPlaceholder') : ''}
      onChange={(e) => setLocalValue(e.target.value || MASK)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
      autoComplete="new-password"
      data-lpignore="true"
      data-1p-ignore
      data-form-type="other"
      className={textInputClass}
    />
  )
}

export interface SettingsInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'> {
  value: string | number
  onChange: (v: string) => void
}

export function SettingsInput({ value, onChange, type = 'text', ...props }: SettingsInputProps) {
  const [localValue, setLocalValue] = useState(value)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    setLocalValue(value)
  }, [value])

  // 卸载时清掉未触发的 debounce，避免对已卸载组件 setState / 提交。
  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current) }, [])

  const commit = (v: string) => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null }
    if (String(v) !== String(value)) onChange(v)
  }

  // 数字框（spinner 点箭头 / 输入）停手 ~500ms 自动提交，不必主动失焦也有反馈；
  // 文本框保持失焦 / Enter 提交（避免输入中途半截 PUT）。
  const debounced = type === 'number'

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value
    setLocalValue(v)
    if (!debounced) return
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => commit(v), 500)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.currentTarget.blur()
    }
  }

  return (
    <input
      type={type}
      {...props}
      value={localValue}
      onChange={handleChange}
      onBlur={() => commit(String(localValue))}
      onKeyDown={handleKeyDown}
    />
  )
}

/** 互斥选项 pill 组（样式同「系统 → 版本更新通道」的 .pill-radio）。
 *  显示设置的语言 / 主题 / 密度等少选项切换用。 */
export function PillRadioGroup<T extends string>({ options, value, onChange }: {
  options: { id: T; label: string }[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          role="radio"
          aria-checked={value === o.id}
          className={`pill-radio${value === o.id ? ' on' : ''}`}
          onClick={() => { if (value !== o.id) onChange(o.id) }}
        >
          <span className="pill-radio-dot" />{o.label}
        </button>
      ))}
    </div>
  )
}
